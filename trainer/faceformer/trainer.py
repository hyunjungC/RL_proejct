import numpy as np
from tqdm import tqdm
import os
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import torch
import time
import wandb
from omegaconf import OmegaConf
import pickle


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

@torch.no_grad()
def test_styleindependant(cfg, model, test_loader):
    save_path = os.path.join(cfg.save_dir, cfg.wandb_name)
    result_path = os.path.join(save_path,'styleIndependant', 'results')
    os.makedirs(result_path, exist_ok=True)
    
    ckpt_path = os.path.join(save_path, 'best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model = model.to(cfg.trainer.device)
    model.eval()
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    
    prediction = 0.0
    for i, (audio, vertice, template, one_hot_all, file_name, _) in pbar:
        audio, vertice, template, one_hot_all  = audio.to(cfg.trainer.device), vertice.to(cfg.trainer.device), template.to(cfg.trainer.device), one_hot_all.to(cfg.trainer.device)
        
        for iter in range(one_hot_all.shape[-1]):
            one_hot = one_hot_all[:, iter, :]
            if iter == 0:
                prediction = model.predict(audio, template, one_hot)
            else:
                prediction += model.predict(audio, template, one_hot)
        prediction = prediction / one_hot_all.shape[-1]
       
        prediction = prediction.squeeze()  # (seq_len, V*3)
        prediction = prediction.reshape(prediction.shape[0], -1, 3)
        np.save(os.path.join(result_path, file_name[0].split(".")[0] + ".npy"), prediction.detach().cpu().numpy())

@torch.no_grad()
def test(cfg, model, test_loader):
    save_path = os.path.join(cfg.save_dir, cfg.wandb_name)
    result_path = os.path.join(save_path, 'styledependant', 'results')
    os.makedirs(result_path, exist_ok=True)
    ckpt_path = os.path.join(save_path, 'best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    model = model.to(cfg.trainer.device)
    model.eval()
    flame_mask_path = getattr(cfg.dataset, "flame_mask_path", "vocaset/FLAME_masks.pkl")
    with open(flame_mask_path, "rb") as f:
        fb = pickle.load(f, encoding="latin")

    output = []
    for r in ["eye_region", "forehead", "nose"]:
        output.extend(fb[r])
    upper_map = list(set(output))
    vertices_gt = []
    vertices_pred = []
    train_subjects_list = [i for i in cfg.dataset.train_subjects.split(" ")]
    motion_std_difference = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (audio, vertice, template, one_hot_all, file_name, _) in pbar:
        # to gpu
        audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(
            device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()  # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())

            vertices_gt.append(vertice.reshape(-1, 5023, 3).detach().cpu().numpy())
            vertices_pred.append(prediction.reshape(-1, 5023, 3).detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze()  # (seq_len, V*3)
                np.save(
                    os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())

                vertices_gt.append(vertice[:, :prediction.shape[0]].reshape(-1, 5023, 3).detach().cpu().numpy())
                vertices_pred.append(prediction.reshape(-1, 5023, 3).detach().cpu().numpy())

                motion_pred = prediction.reshape(-1, 5023, 3).detach().cpu().numpy() - template.reshape(1, 5023,
                                                                                                        3).detach().cpu().numpy()
                motion_gt = vertice[:, :prediction.shape[0]].reshape(-1, 5023,
                                                                     3).detach().cpu().numpy() - template.reshape(1,
                                                                                                                  5023,
                                                                                                                  3).detach().cpu().numpy()

                L2_dis_upper = np.array([np.square(motion_gt[:, v, :]) for v in upper_map])
                L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
                L2_dis_upper = np.sum(L2_dis_upper, axis=2)
                L2_dis_upper = np.std(L2_dis_upper, axis=0)
                gt_motion_std = np.mean(L2_dis_upper)

                L2_dis_upper = np.array([np.square(motion_pred[:, v, :]) for v in upper_map])
                L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
                L2_dis_upper = np.sum(L2_dis_upper, axis=2)
                L2_dis_upper = np.std(L2_dis_upper, axis=0)
                pred_motion_std = np.mean(L2_dis_upper)

                motion_std_difference.append(gt_motion_std - pred_motion_std)

    vertices_gt = np.concatenate(vertices_gt)
    vertices_pred = np.concatenate(vertices_pred)
    L2_dis_mouth_max = np.array(
        [np.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map]
    )
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)
    lve = np.mean(L2_dis_mouth_max)
    print('LVE: {:.4e}, FDD: {:.4e}'.format(lve, sum(motion_std_difference) / len(motion_std_difference)))

def train(cfg, train_loader, dev_loader, model, guidance_model, head, optimizer, criterion, epoch, last_train):
    trainer_cfg = cfg.trainer
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset

    # ★ RL 가중치 (config에 없으면 기본값 사용)
    actor_weight = getattr(trainer_cfg, "actor_weight", 1e-4)
    critic_weight = getattr(trainer_cfg, "critic_weight", 1e-4)

    # guidance_model은 보통 freeze, head는 critic으로 학습
    guidance_model.eval()
    head.train()

    def rep_output(audios, motion, audio_length, return_value=False):
        # audios: (1, T, 1, 20, 128) 같은 구조 → (T, 1, 20, 128)로 쓰려고 permute
        audios = audios.permute(1, 0, 2, 3)
        clip_num = min(audio_length, motion.shape[0] - 4)
        audios = list(audios[:clip_num])

        lip_score_sum, real_score_sum = 0.0, 0.0
        value_sum = 0.0  # ★ critic용

        batch_num = clip_num // trainer_cfg.guidance_batch_size
        remain_num = clip_num % trainer_cfg.guidance_batch_size

        total_clips = batch_num * trainer_cfg.guidance_batch_size + remain_num

        for b in range(batch_num):
            motion_batch = []
            for i in range(trainer_cfg.guidance_batch_size):
                motion_batch.append(
                    motion[b * trainer_cfg.guidance_batch_size + i:
                        b * trainer_cfg.guidance_batch_size + i + 5]
                )
            motion_batch_tensor = torch.stack(motion_batch, dim=0).cuda()
            audio_batch_tensor = torch.stack(
                audios[b * trainer_cfg.guidance_batch_size:
                    b * trainer_cfg.guidance_batch_size + trainer_cfg.guidance_batch_size],
                dim=0
            ).cuda()

            # [B, 5, 15069], [B, 1, 20, 128] → feature 뽑기
            with torch.no_grad():
                vertex_feat, audio_feat = guidance_model.forward_features(
                    motion_batch_tensor, audio_batch_tensor
                )
            # backbone은 고정, head만 학습시키고 싶으면 여기서 detach
            vertex_feat = vertex_feat.detach()
            audio_feat = audio_feat.detach()

            if return_value:
                # ★ RL용: lip, real, value 모두 사용
                lip_score, real_score, value = head(vertex_feat, audio_feat, return_value=True)
                value_sum += value.sum()
            else:
                # ★ 기존 방식: lip, real만 사용
                lip_score, real_score = head(vertex_feat, audio_feat)

            lip_score_sum += lip_score.sum()
            real_score_sum += real_score.sum()

        if remain_num >= 2:
            motion_batch = []
            for i in range(remain_num):
                motion_batch.append(
                    motion[batch_num * trainer_cfg.guidance_batch_size + i:
                        batch_num * trainer_cfg.guidance_batch_size + i + 5]
                )
            motion_batch_tensor = torch.stack(motion_batch, dim=0).cuda()
            audio_batch_tensor = torch.stack(
                audios[batch_num * trainer_cfg.guidance_batch_size:
                    batch_num * trainer_cfg.guidance_batch_size + remain_num],
                dim=0
            ).cuda()

            with torch.no_grad():
                vertex_feat, audio_feat = guidance_model.forward_features(
                    motion_batch_tensor, audio_batch_tensor
                )
            vertex_feat = vertex_feat.detach()
            audio_feat = audio_feat.detach()

            if return_value:
                lip_score, real_score, value = head(vertex_feat, audio_feat, return_value=True)
                value_sum += value.sum()
            else:
                lip_score, real_score = head(vertex_feat, audio_feat)

            lip_score_sum += lip_score.sum()
            real_score_sum += real_score.sum()

        # 평균으로 정규화 (clip이 하나도 없을 경우 방어)
        if total_clips == 0:
            lip_score_mean = torch.tensor(0.0, device=motion.device)
            real_score_mean = torch.tensor(0.0, device=motion.device)
            value_mean = torch.tensor(0.0, device=motion.device)
        else:
            lip_score_mean = lip_score_sum / total_clips
            real_score_mean = real_score_sum / total_clips
            value_mean = value_sum / total_clips if return_value else None

        if return_value:
            return lip_score_mean, real_score_mean, value_mean
        else:
            return lip_score_mean, real_score_mean
        
    save_path = os.path.join(cfg.save_dir, cfg.wandb_name)
    os.makedirs(save_path, exist_ok=True)
    if last_train != 0:
        model.load_state_dict(torch.load(os.path.join(cfg.load_path, '{}_model.pth'.format(last_train)),
                                        map_location=torch.device('cpu')))
        model = model.to(cfg.trainer.device)

    run_name = f"{cfg.wandb_name}_{model_cfg.name}_{time.strftime('%m_%d_%H_%M', time.localtime())}"
    wandb.init(
        project=getattr(cfg, 'wandb_project', 'RL-3D3M'),
        entity=getattr(cfg, 'wandb_entity', None),
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=getattr(cfg, 'wandb_mode', 'online')
    )
    wandb.watch(model, log="gradients", log_freq=100, log_graph=False)
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="epoch")
    except Exception:
        pass

    iteration = 0
    train_subjects_list = [i for i in dataset_cfg.train_subjects.split(" ")]
    global_step = 0
    best_val_loss = float('inf')

    for e in range(epoch + 1):
        loss_log = []
        actor_loss_log = []
        critic_loss_log = []
        reward_log = []

        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        # RL 사용 여부 플래그 (가중치가 0이면 완전 supervised로 돌리며 reward 모델도 건너뜀)
        use_rl = (actor_weight > 0 or critic_weight > 0)

        for i, (audio, vertice, template, one_hot, file_name, rep_audio) in pbar:
            iteration += 1
            # to gpu
            audio = audio.to(cfg.trainer.device)
            vertice = vertice.to(cfg.trainer.device)
            template = template.to(cfg.trainer.device)
            one_hot = one_hot.to(cfg.trainer.device)

            # =====================================================
            # 1) Actor 호출 (RL 여부에 따라 반환 형태 분기)
            # =====================================================
            if use_rl:
                vertice_mu, vertice_sample, sup_loss, dist = model(
                    audio,
                    template,
                    vertice,
                    one_hot,
                    criterion,
                    teacher_forcing=False,
                    return_dist=True,   # ★ RL 핵심 플래그
                )
            else:
                # RL 미사용 시 deterministic forward만
                vertice_sample, sup_loss = model(
                    audio,
                    template,
                    vertice,
                    one_hot,
                    criterion,
                    teacher_forcing=False,
                    return_dist=False,
                )
                dist = None

            # =====================================================
            # 2) Reward + Value (Critic) 계산
            #    - RL 가중치가 0이면 스킵하여 연산량 절감
            # =====================================================
            if use_rl and dist is not None:
                lip_score, real_score, value = rep_output(
                    rep_audio,
                    vertice_sample.squeeze(0),
                    rep_audio.shape[1],
                    return_value=True
                )

                reward = lip_score + real_score        # scalar
                reward = reward * getattr(trainer_cfg, "reward_scale", 1.0)
                reward_det = reward.detach()
                value_det = value.detach()

                # =================================================
                # 3) Advantage, Actor loss, Critic loss
                # =================================================
                advantage = reward_det - value_det     # scalar

                # dist: Normal(μ, σ), vertice_sample은 그 샘플 a
                log_prob = dist.log_prob(vertice_sample)   # (B, T, V*3)
                log_prob_mean = log_prob.mean()            # 전체 평균

                actor_loss = - advantage * log_prob_mean
                critic_loss = (reward_det - value).pow(2)
            else:
                # RL 비활성 시 모든 RL 항을 0으로
                reward = torch.tensor(0.0, device=cfg.trainer.device)
                actor_loss = torch.tensor(0.0, device=cfg.trainer.device)
                critic_loss = torch.tensor(0.0, device=cfg.trainer.device)

            # =====================================================
            # 4) 최종 loss 합산
            # =====================================================
            total_loss = sup_loss \
                         + actor_weight * actor_loss \
                         + critic_weight * critic_loss

            total_loss.backward()
            # grad clipping + step (accumulation 고려)
            if (i + 1) % cfg.trainer.gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_log.append(sup_loss.item())
            actor_loss_log.append(actor_loss.item())
            critic_loss_log.append(critic_loss.item())
            reward_log.append(reward.item())

            pbar.set_description(
                "(Epoch {}, iter {}) SUP:{:.6f} ACT:{:.6f} CRT:{:.6f} REW:{:.6f}".format(
                    e, iteration,
                    sup_loss.item(),
                    actor_loss.item(),
                    critic_loss.item(),
                    reward.item()
                )
            )

            # per-step logging to W&B
            try:
                grad_norm = 0.0
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().data.norm(2))
                if len(grads) > 0:
                    import torch as _torch
                    grad_norm = float(_torch.norm(_torch.stack(grads), 2).item())
            except Exception:
                grad_norm = None

            global_step += 1
            wandb.log({
                "train/sup_loss": float(sup_loss.item()),
                "train/actor_loss": float(actor_loss.item()),
                "train/critic_loss": float(critic_loss.item()),
                "train/reward": float(reward.item()),
                "train/grad_norm": grad_norm if grad_norm is not None else 0.0,
                "train/lr": float(optimizer.param_groups[0]['lr']),
                "global_step": int(global_step),
                "epoch": int(e),
            })

        # epoch aggregation
        if len(loss_log) > 0:
            wandb.log({
                "train/sup_loss_epoch_mean": float(np.mean(loss_log)),
                "train/actor_loss_epoch_mean": float(np.mean(actor_loss_log)),
                "train/critic_loss_epoch_mean": float(np.mean(critic_loss_log)),
                "train/reward_epoch_mean": float(np.mean(reward_log)),
                "epoch": int(e),
            })

        # ==============================
        # validation 부분은 기존 코드 거의 그대로
        # (rep_output(..., return_value=False)로 호출)
        # ==============================
        valid_loss_log = []
        vertices_gt = []
        vertices_pred = []
        val_lip_score_log = []
        val_real_score_log = []
        model.eval()
        with torch.no_grad():
            for audio, vertice, template, one_hot_all, file_name, rep_audio in dev_loader:
                audio = audio.to(cfg.trainer.device)
                vertice = vertice.to(cfg.trainer.device)
                template = template.to(cfg.trainer.device)
                one_hot_all = one_hot_all.to(cfg.trainer.device)

                train_subject = "_".join(file_name[0].split("_")[:-1])
                if train_subject in train_subjects_list:
                    condition_subject = train_subject
                    iter_idx = train_subjects_list.index(condition_subject)
                    one_hot = one_hot_all[:, iter_idx, :]
                    vertice_out, loss = model(audio, template, vertice, one_hot, criterion)
                    lip_score, real_score = rep_output(
                        rep_audio, vertice_out.squeeze(0), rep_audio.shape[1], return_value=False
                    )
                    valid_loss_log.append(loss.item())
                    val_lip_score_log.append(lip_score.item())
                    val_real_score_log.append(real_score.item())
                else:
                    for iter_idx in range(one_hot_all.shape[-1]):
                        condition_subject = train_subjects_list[iter_idx]
                        one_hot = one_hot_all[:, iter_idx, :]
                        vertice_out, loss = model(audio, template, vertice, one_hot, criterion)
                        lip_score, real_score = rep_output(
                            rep_audio, vertice_out.squeeze(0), rep_audio.shape[1], return_value=False
                        )
                        valid_loss_log.append(loss.item())
                        val_lip_score_log.append(lip_score.item())
                        val_real_score_log.append(real_score.item())

                vertices_gt.append(vertice.reshape(-1, 5023, 3).detach().cpu().numpy())
                vertices_pred.append(vertice_out.reshape(-1, 5023, 3).detach().cpu().numpy())

            wandb.log({
                "val/loss_last": float(loss.item()),
                "epoch": int(e),
            })

        current_loss = np.mean(valid_loss_log)
        vertices_gt = np.concatenate(vertices_gt)
        vertices_pred = np.concatenate(vertices_pred)
        L2_dis_mouth_max = np.array(
            [np.square(vertices_gt[:, v, :] - vertices_pred[:, v, :]) for v in mouth_map]
        )
        L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
        L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
        L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)
        lve = np.mean(L2_dis_mouth_max)

        wandb.log({
            "val/loss_epoch_mean": float(current_loss),
            "val/lve": float(lve),
            "epoch": int(e),
        })

        # save only the best model
        if lve < best_val_loss:
            best_val_loss = lve
            ckpt_path = os.path.join(save_path, 'best.pt')
            torch.save(model.state_dict(), ckpt_path)
            # reward head도 별도로 저장해 추후 평가에 사용 가능하도록 함
            head_ckpt_path = os.path.join(save_path, 'best_head.pt')
            torch.save(head.state_dict(), head_ckpt_path)

        print("epoch: {}, current loss:{:.7f}".format(e + 1, current_loss))

    wandb.finish()
    return model
