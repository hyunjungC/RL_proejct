
import numpy as np
from tqdm import tqdm
import os
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import torch
import torch.nn as nn
import time
import importlib
import wandb
from models.reward.models.modeling import SpeechMeshTransformer
from src import utils
from models.reward.models.head_v2 import ScoreHead
from functools import partial
from omegaconf import OmegaConf, DictConfig
import hydra


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # ----------------------------------------------------------------------
    # 0. Hydra로 모델 / 데이터로더 / 트레이너 모듈 로딩
    # ----------------------------------------------------------------------
    model_name = cfg.model.name
    loader_name = cfg.dataset.loader_name

    model_module = importlib.import_module(f'models.{model_name}.model')
    ModelClass = getattr(model_module, 'Model')

    trainer_module = importlib.import_module(f'trainer.{model_name}.trainer')
    dataset_loader_module = importlib.import_module(f'dataset.{loader_name}')
    get_dataloaders = getattr(dataset_loader_module, 'get_dataloaders')

    device = cfg.trainer.device

    # ----------------------------------------------------------------------
    # 1. Actor: FaceFormer (baseline 모델)
    # ----------------------------------------------------------------------
    model = ModelClass(cfg, cfg.dataset.name, device)
    print("model parameters: ", count_parameters(model))

    assert torch.cuda.is_available()
    model = model.to(device)

    # ----------------------------------------------------------------------
    # 2. Reward Backbone: SpeechMeshTransformer (Perceptual 3D Talking Head)
    #    - freeze 해서 feature extractor / 평가자 역할만 수행
    # ----------------------------------------------------------------------
    guidance_model = SpeechMeshTransformer(
        vertex_size=5023 * 3,
        patch_size=5023 * 3,
        embed_dim=512,
        num_heads=8,
        depth=10,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        embed_dim_audio=512,
        num_heads_audio=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        depth_audio=10,
    )

    guidance_model_path = OmegaConf.select(cfg, "reward.guidance_model_path")
    if guidance_model_path is None:
        guidance_model_path = cfg.trainer.guidance_model_path

    print("Load guidance model ckpt from %s" % guidance_model_path)
    checkpoint = torch.load(guidance_model_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint['model']
    utils.load_state_dict(guidance_model, checkpoint_model)

    # backbone은 고정 (freeze)
    for name, param in guidance_model.named_parameters():
        param.requires_grad = False

    guidance_model = guidance_model.to(device)
    guidance_model.eval()

    # ----------------------------------------------------------------------
    # 3. ScoreHead: Reward(+Critic Value) Head
    #    - lip_score / real_score / value(V(s))를 출력력
    # ----------------------------------------------------------------------
    head = ScoreHead(
        d_audio=512,
        d_mesh=512,
        hidden=256,
        dropout=0.1,
        out_activation='sigmoid'
    ).to(device)

    guidance_head_model_path = OmegaConf.select(cfg, "reward.guidance_head_model_path")
    if guidance_head_model_path is None:
        guidance_head_model_path = cfg.trainer.guidance_head_model_path

    print("Load guidance head ckpt from %s" % guidance_head_model_path)
    checkpoint = torch.load(guidance_head_model_path, map_location='cpu', weights_only=False)

    head_checkpoint = checkpoint['head']
    head.load_state_dict(head_checkpoint, strict=False)
    head.train()

    # ----------------------------------------------------------------------
    # 4. 데이터 로더 준비
    # ----------------------------------------------------------------------
    dataset = get_dataloaders(cfg.dataset)
    train_loader = dataset["train"]
    valid_loader = dataset["valid"]
    test_loader = dataset["test"]

    # ----------------------------------------------------------------------
    # 5. Loss & Optimizer 설정
    #    - Actor(model) + Head(head)를 함께 업데이트
    #    - Reward Backbone(guidance_model)은 고정
    # ----------------------------------------------------------------------
    if cfg.train:
        # 기존 코드 기준으로 MSELoss 사용
        criterion = nn.MSELoss(reduction='none')

        # Actor + Head 파라미터를 분리해 lr을 다르게 줄 수 있게 구성
        actor_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        head_params = list(filter(lambda p: p.requires_grad, head.parameters()))
        optimizer = torch.optim.Adam(
            [
                {"params": actor_params, "lr": cfg.trainer.lr},
                {"params": head_params, "lr": cfg.trainer.head_lr},
            ]
        )

        # RL loss weight 기본값 자동 세팅 (config에 없으면)
        actor_w = OmegaConf.select(cfg, "trainer.actor_weight")
        critic_w = OmegaConf.select(cfg, "trainer.critic_weight")
        if actor_w is None:
            cfg.trainer.actor_weight = 1e-4
        if critic_w is None:
            cfg.trainer.critic_weight = 1e-4

        trainer_module.train(
            cfg,
            train_loader,
            valid_loader,
            model,
            guidance_model,
            head,
            optimizer,
            criterion,
            epoch=cfg.trainer.max_epoch,
            last_train=cfg.trainer.last_train,
        )

    # ----------------------------------------------------------------------
    # 6. 테스트 (기존 구조 그대로 사용 가능)
    # ----------------------------------------------------------------------
    if cfg.test:
        # style dependant
        trainer_module.test(cfg, model, test_loader)

        # style independant
        #trainer_module.test_styleindependant(cfg, model, test_loader)
        # 필요하면 try/except로 두 종류 테스트를 나눌 수도 있음

if __name__ == "__main__":
    main()
