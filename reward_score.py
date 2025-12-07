import csv
import os
from pathlib import Path
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from models.reward.models.modeling import SpeechMeshTransformer
from models.reward.models.head_v2 import ScoreHead
from src import utils


DEVICE = "cuda:1"
PRED_DIR = Path("checkpoints/faceformer_hj/styledependant/results")
GT_DIR = Path("vocaset/vertices_npy")
MEL_DIR = Path("vocaset/wav_npy")
PRED_CSV = PRED_DIR.parent / "all_testset_reward_scores_gpu3.csv"
GT_CSV = PRED_DIR.parent / "all_gt_reward_scores_gpu3.csv"




def load_models():
    """Load reward backbone + head for inference."""
    guidance = SpeechMeshTransformer(
        vertex_size=5023 * 3,
        patch_size=5023 * 3,
        embed_dim=512,
        num_heads=8,
        depth=10,
        img_size_audio=(64, 128),
        patch_size_audio=16,
        embed_dim_audio=512,
        num_heads_audio=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        max_length=5,
        depth_audio=10,
    ).to(DEVICE).eval()
    ckpt = torch.load("checkpoints/reward/model_loss.pth", map_location="cpu", weights_only=False)
    utils.load_state_dict(guidance, ckpt["model"])

    head = ScoreHead(
        d_audio=512,
        d_mesh=512,
        hidden=256,
        dropout=0.1,
        out_activation="sigmoid",
    ).to(DEVICE).eval()
    hckpt = torch.load("checkpoints/faceformer_hj/best_head.pt", map_location="cpu", weights_only=False)
    state_dict = hckpt.get("head", hckpt)
    head.load_state_dict(state_dict, strict=True)
    return guidance, head

#print("head.weight sum:", next(head.parameters()).sum().item())
    
    

def score_sample(mesh_np: np.ndarray, mel_np: np.ndarray, guidance, head):
    """Compute mean lip/real/value scores over 5-frame windows."""
    mesh = torch.tensor(mesh_np, dtype=torch.float32, device=DEVICE)  # [T, 5023, 3]
    mel = torch.tensor(mel_np, dtype=torch.float32, device=DEVICE)

    # Expect [clip, 1, 20, 128]; add channel dim if missing.
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    if mel.dim() != 4:
        print(f"[WARN] Unexpected mel shape {mel.shape}, skip")
        return None

    clip_num = min(mel.shape[0], mesh.shape[0] - 4)
    if clip_num < 1:
        return None

    lip_sum = real_sum = val_sum = 0.0
    with torch.no_grad():
        for i in range(clip_num):
            m_clip = mesh[i : i + 5].unsqueeze(0)  # [1, 5, 5023, 3]
            a_clip = mel[i].unsqueeze(0)           # [1, 1, 20, 128]
            v_feat, a_feat = guidance.forward_features(m_clip, a_clip)
            lip, real, val = head(v_feat, a_feat, return_value=True)
            lip_sum += lip.sum().item()
            real_sum += real.sum().item()
            val_sum += val.sum().item()

    return {
        "lip_score": lip_sum / clip_num,
        "real_score": real_sum / clip_num,
        "value": val_sum / clip_num,
        "clips": clip_num,
    }


def score_directory(mesh_dir: Path, csv_path: Path, guidance, head, desc: str):
    files = sorted(mesh_dir.glob("*.npy"))
    if not files:
        print(f"[WARN] No .npy files found in {mesh_dir}")
        return

    rows = []
    for mesh_path in tqdm(files, desc=desc):
        mel_path = MEL_DIR / mesh_path.name
        if not mel_path.exists() and "_condition_" in mesh_path.stem:
            # FaceFormer 결과처럼 파일명에 condition이 붙은 경우, 앞부분만 떼서 mel을 찾는다.
            stripped = mesh_path.stem.split("_condition_")[0] + ".npy"
            alt_mel = MEL_DIR / stripped
            if alt_mel.exists():
                print(f"[INFO] mel fallback: {mesh_path.name} -> {alt_mel.name}")
                mel_path = alt_mel
        if not mel_path.exists():
            print(f"[WARN] mel not found, skip: {mel_path}")
            continue

        mesh_np = np.load(mesh_path)
        mel_np = np.load(mel_path)
        scores = score_sample(mesh_np, mel_np, guidance, head)
        if scores is None:
            print(f"[WARN] Not enough frames, skip: {mesh_path.name}")
            continue

        row = {
            "file": mesh_path.name,
            "lip_score": scores["lip_score"],
            "real_score": scores["real_score"],
            "value": scores["value"],
            "clips": scores["clips"],
        }
        rows.append(row)
        print(
            f'{desc} {row["file"]}: lip={row["lip_score"]:.4f}, real={row["real_score"]:.4f}, '
            f'value={row["value"]:.4f}, clips={row["clips"]}'
        )

    if rows:
        os.makedirs(csv_path.parent, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "lip_score", "real_score", "value", "clips"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved scores to {csv_path}")
    else:
        print(f"No scores were computed for {mesh_dir}")


def main():
    guidance, head = load_models()
    score_directory(PRED_DIR, PRED_CSV, guidance, head, desc="Pred")
    score_directory(GT_DIR, GT_CSV, guidance, head, desc="GT  ")


if __name__ == "__main__":
    main()
