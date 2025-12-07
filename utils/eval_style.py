#============================================
'''
CUDA_VISIBLE_DEVICES=1 python utils/eval.py \
  --pred_dir <예측 npy 폴더> \
  --gt_dir <GT npy 폴더> \
  --mel_dir <mel npy 폴더>
'''
#============================================

'''
예시:

CUDA_VISIBLE_DEVICES=1 python utils/eval_style.py \
  --pred_dir /workspace/RL-VOCASET_my_copy_check_3/checkpoints/faceformer_hj/styledependant/results \
  --gt_dir /workspace/RL-VOCASET_my_copy/vocaset/vertices_npy \
  --mel_dir /workspace/RL-VOCASET_my_copy/vocaset/wav_npy

CUDA_VISIBLE_DEVICES=1 python utils/eval.py \
  --pred_dir /workspace/RL-VOCASET_my_copy/checkpoints/faceformer_base/styleIndependant/results \
  --gt_dir /workspace/RL-VOCASET_my_copy/vocaset/vertices_npy \
  --mel_dir /workspace/RL-VOCASET_my_copy/vocaset/wav_npy
'''

import os
import argparse
from typing import List, Tuple
import numpy as np
import pickle
from evaluate_PLRS import load_state_dict
from timm.models import create_model
import sys
sys.path.append('/workspace')
from models.reward.models.modeling import speech_mesh_rep
import torch

np.set_printoptions(threshold=np.inf)

# VOCASET mouth region vertex indices
mouth_map = np.array([
    1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1590, 1590, 1591, 1593, 1593,
    1657, 1658, 1661, 1662, 1663, 1667, 1668, 1669, 1670, 1686, 1687, 1691, 1693,
    1694, 1695, 1696, 1697, 1700, 1702, 1703, 1704, 1709, 1710, 1711, 1712, 1713,
    1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1728, 1729, 1730,
    1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1740, 1743, 1748, 1749, 1750,
    1751, 1758, 1763, 1765, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779,
    1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796, 1801,
    1802, 1803, 1804, 1826, 1827, 1836, 1846, 1847, 1848, 1849, 1850, 1865, 1866,
    2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2726, 2726, 2727, 2729, 2729,
    2774, 2775, 2778, 2779, 2780, 2784, 2785, 2786, 2787, 2803, 2804, 2808, 2810,
    2811, 2812, 2813, 2814, 2817, 2819, 2820, 2821, 2826, 2827, 2828, 2829, 2830,
    2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2843, 2844, 2845,
    2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2855, 2858, 2863, 2864, 2865,
    2866, 2869, 2871, 2873, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886,
    2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2899, 2904,
    2905, 2906, 2907, 2928, 2929, 2934, 2935, 2936, 2937, 2938, 2939, 2948, 2949,
    3503, 3504, 3506, 3509, 3511, 3512, 3513, 3531, 3533, 3537, 3541, 3543, 3546,
    3547, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801,
    3802, 3803, 3804, 3805, 3806, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921,
    3922, 3923, 3924, 3925, 3926, 3927, 3928
])

# 평가에 사용할 test subject 두 명만 필터링
TEST_SUBJECTS = {
    "FaceTalk_170809_00138_TA",
    "FaceTalk_170731_00024_TA",
}


def _list_npy_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(".npy")])


def compute_folder_metrics(
    pred_dir: str,
    gt_dir: str,
    mel_dir: str,
    model: torch.nn.Module,
) -> Tuple[float, float, float, float, int]:
    """
    pred_dir와 gt_dir의 파일(.npy)을 매칭하여
    전체 LVE / FDD / MVE / PLRS를 계산합니다.

    - style-dependent 결과:
      예) FaceTalk_170731_00024_TA_sentence01_condition_FaceTalk_170725_00137_TA.npy
      -> GT / mel: FaceTalk_170731_00024_TA_sentence01.npy

    - TEST_SUBJECTS 에 포함된 subject만 사용.
    """

    pred_files = _list_npy_files(pred_dir)
    model.eval()

    # FLAME upper-face mask 로딩 (eye / forehead / nose)
    with open("vocaset/FLAME_masks.pkl", "rb") as f:
        fb = pickle.load(f, encoding="latin")
    output = []
    for r in ["eye_region", "forehead", "nose"]:
        output.extend(fb[r])
    upper_map = list(set(output))

    # subject별 템플릿 로딩
    with open("vocaset/templates.pkl", "rb") as f:
        templates = pickle.load(f, encoding="latin1")

    pred_vertices_all = []
    gt_vertices_all = []
    motion_std_difference = []
    cosine_similarity = 0.0
    valid_file_count = 0  # 실제로 사용된 파일 개수(PLRS 분모용)

    for fname in pred_files:
        stem = os.path.splitext(fname)[0]

        # 1) style-dependent vs independent 이름 분리
        #    "_condition_" 앞부분 = GT / mel / subject 기준 이름
        if "_condition_" in stem:
            base = stem.split("_condition_")[0]
        else:
            base = stem

        # base 예시: "FaceTalk_170731_00024_TA_sentence01"
        if "_sentence" not in base:
            # 예상치 못한 이름이면 스킵
            continue

        # subject_id: "FaceTalk_170731_00024_TA"
        subject_id = base.split("_sentence")[0]

        # test subject 두 명만 사용
        if subject_id not in TEST_SUBJECTS:
            continue

        # (중요) sentence 번호로 자르지 않고, sentence01~… 전부 사용

        # GT / mel 파일 이름
        gt_fname = base + ".npy"
        mel_fname = base + ".npy"

        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, gt_fname)
        mel_path = os.path.join(mel_dir, mel_fname)

        # 파일 없으면 스킵
        if not (os.path.isfile(pred_path)
                and os.path.isfile(gt_path)
                and os.path.isfile(mel_path)):
            continue

        # 템플릿
        if subject_id not in templates:
            # 템플릿이 없으면 스킵
            continue
        template = templates[subject_id].reshape(1, 5023, 3)

        # ====== 여기부터는 기존 계산 로직 그대로 ======
        audios = np.load(mel_path)
        pred_vertices = np.load(pred_path)
        gt_vertices = np.load(gt_path)

        if gt_vertices.ndim == 2:
            gt_vertices = gt_vertices.reshape(-1, 5023, 3)

        if pred_vertices.ndim == 2:
            pred_vertices = pred_vertices.reshape(-1, 5023, 3)

        # 프레임 수 맞추기 (GT를 pred 길이에 맞게 리샘플)
        if gt_vertices.shape[0] != pred_vertices.shape[0]:
            idxs = np.linspace(
                0, pred_vertices.shape[0] - 1,
                num=pred_vertices.shape[0]
            )
            idxs = np.round(idxs).astype(int)
            gt_vertices = gt_vertices[idxs]

        # PLRS 계산용 클립 구성
        clip_num = min(audios.shape[0], pred_vertices.shape[0] - 4)
        audios = audios[:clip_num]

        audio_batch = []
        vertice_batch = []
        for i in range(clip_num // 5):
            audio_batch.append(torch.from_numpy(audios[i * 5]))
            vertice_batch.append(
                torch.from_numpy(pred_vertices[i * 5:i * 5 + 5])
            )

        if len(audio_batch) > 0 and len(vertice_batch) > 0:
            vertice_batch_tensor = torch.stack(vertice_batch, axis=0).cuda()
            audio_batch_tensor = torch.stack(audio_batch, dim=0).cuda()

            with torch.no_grad():
                vertex_feature, audio_feature = model(
                    vertice_batch_tensor.float(),
                    audio_batch_tensor.unsqueeze(1)
                )

            cosine_similarity += (
                vertex_feature @ audio_feature.t()
            ).diag().mean().item()

        # LVE / MVE / FDD 계산용으로 누적
        pred_vertices_all.append(pred_vertices)
        gt_vertices_all.append(gt_vertices)

        motion_pred = pred_vertices - template
        motion_gt = gt_vertices - template

        # ----- FDD: upper-face motion std 차이 -----
        L2_dis_upper = np.array(
            [np.square(motion_gt[:, v, :]) for v in upper_map]
        )
        L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
        L2_dis_upper = np.sum(L2_dis_upper, axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        gt_motion_std = np.mean(L2_dis_upper)

        L2_dis_upper = np.array(
            [np.square(motion_pred[:, v, :]) for v in upper_map]
        )
        L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
        L2_dis_upper = np.sum(L2_dis_upper, axis=2)
        L2_dis_upper = np.std(L2_dis_upper, axis=0)
        pred_motion_std = np.mean(L2_dis_upper)

        motion_std_difference.append(gt_motion_std - pred_motion_std)

        valid_file_count += 1

    # 유효한 파일이 하나도 없으면 0 반환 (에러 방지)
    if valid_file_count == 0 or len(pred_vertices_all) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    # 모든 예측/GT를 프레임 방향으로 concat
    pred_vertices_all = np.concatenate(pred_vertices_all, axis=0)
    gt_vertices_all = np.concatenate(gt_vertices_all, axis=0)

    # ----- MVE (전체 얼굴 max vertex error per frame 평균) -----
    l2_all_dists = np.array(np.square(pred_vertices_all - gt_vertices_all))
    l2_all_dists = np.transpose(l2_all_dists, (1, 0, 2))   # (5023, T_all, 3)
    l2_all_dists = np.sum(l2_all_dists, axis=2)            # (5023, T_all)
    l2_all_dists = np.max(l2_all_dists, axis=0)            # (T_all,)
    mve = np.mean(l2_all_dists)

    # ----- LVE (입술 영역만) -----
    l2_lip_dists = np.array([
        np.square(pred_vertices_all[:, v, :] - gt_vertices_all[:, v, :])
        for v in mouth_map
    ])  # (num_lip, T_all, 3)
    l2_lip_dists = np.transpose(l2_lip_dists, (1, 0, 2))   # (T_all, num_lip, 3)
    l2_lip_dists = np.sum(l2_lip_dists, axis=2)            # (T_all, num_lip)
    l2_lip_dists = np.max(l2_lip_dists, axis=1)            # (T_all,)
    lve = np.mean(l2_lip_dists)

    # ----- FDD 평균 -----
    fdd = sum(motion_std_difference) / len(motion_std_difference)

    # ----- PLRS: cosine similarity 평균 -----
    plrs = cosine_similarity / valid_file_count

    return lve, fdd, mve, plrs, valid_file_count


def main():
    ap = argparse.ArgumentParser(description="폴더 단위 LVE/FDD/MVE/PLRS 집계")
    ap.add_argument("--pred_dir", required=True, help="예측 .npy 폴더")
    ap.add_argument(
        "--gt_dir",
        default=os.path.join("/data/", "vocaset", "vertices_npy"),
        help="GT .npy 폴더",
    )
    ap.add_argument(
        "--mel_dir",
        default=os.path.join("/data/", "vocaset", "wav_npy"),
        help="Mel 폴더",
    )
    ap.add_argument(
        "--mask",
        default=os.path.join("vocaset", "FLAME_masks.pkl"),
        help="VOCASET FLAME_masks.pkl 경로 (기본: vocaset/FLAME_masks.pkl)",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="device to use for training / testing",
    )
    ap.add_argument(
        "--model_path",
        default="checkpoints/reward/model_loss.pth",
        help="model checkpoint path",
    )
    args = ap.parse_args()
    device = torch.device(args.device)

    # reward backbone 모델 로드
    model = create_model(
        "speech_mesh_rep",
        pretrained=False,
        depth=10,
        depth_audio=10,
    )

    if model.encoder is not None:
        patch_size = 254 * 3
        print("Patch size = %s" % str(patch_size))
        args.window_size = (5 // 1, patch_size)
        args.patch_size = patch_size

    if model.encoder_audio is not None:
        patch_size_audio = model.encoder_audio.patch_embed.patch_size  # (16,16)
        print("Patch size (audio) = %s" % str(patch_size_audio))
        args.window_size_audio = (
            64 // patch_size_audio[0],
            128 // patch_size_audio[1],
        )
        args.patch_size_audio = patch_size_audio

    checkpoint = torch.load(
        args.model_path, map_location="cpu", weights_only=False
    )

    print("Load ckpt from %s" % args.model_path)
    checkpoint_model = None
    for model_key in ["model", "module"]:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    load_state_dict(model, checkpoint_model, prefix="")

    model.to(device)

    lve, fdd, mve, plrs, file_count = compute_folder_metrics(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        mel_dir=args.mel_dir,
        model=model,
    )

    print(
        f"pred_dir: {args.pred_dir}, "
        f"LVE: {lve:.6e}, "
        f"FDD: {fdd:.6e}, "
        f"MVE: {mve:.6e}, "
        f"PLRS: {plrs:.6e}, "
        f"file_count: {file_count}"
    )


if __name__ == "__main__":
    main()
