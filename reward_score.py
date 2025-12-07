import csv
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.reward.models.modeling import SpeechMeshTransformer
from models.reward.models.head_v2 import ScoreHead
from src import utils


"""
Reward 모델을 이용해 예측 메쉬 / GT 메쉬에 대해
lip / real / value score를 계산하는 스크립트의 데모용 스켈레톤입니다.

- SpeechMeshTransformer / ScoreHead의 실제 아키텍처와 하이퍼파라미터
- 학습된 체크포인트 로드 방식
- 윈도우 길이, 정규화 방식 등 세부 구현

은 논문 보호를 위해 공개하지 않습니다.
"""

DEVICE = "cuda:1"
PRED_DIR = Path("checkpoints/faceformer_hj/styledependant/results")
GT_DIR = Path("vocaset/vertices_npy")
MEL_DIR = Path("vocaset/wav_npy")
PRED_CSV = PRED_DIR.parent / "all_testset_reward_scores.csv"
GT_CSV = PRED_DIR.parent / "all_gt_reward_scores.csv"


def load_models():
    """
    Reward backbone(SpeechMeshTransformer)과 ScoreHead를 로드하는 함수의 스켈레톤입니다.

    실제 구현에서는 대략 다음과 같은 일을 수행합니다:
      - SpeechMeshTransformer(...)를 적절한 설정으로 생성
      - 학습된 reward backbone 체크포인트를 로드
      - ScoreHead(...)를 생성하고 head 가중치를 로드
      - 평가 모드(eval)로 전환 후 GPU/CPU 디바이스에 올림

    이 데모 레포에서는 모델 구조와 체크포인트 경로를 공개하지 않습니다.
    """
    raise NotImplementedError(
        "Private implementation: reward 모델 로드 로직은 비공개입니다."
    )


def score_sample(mesh_np: np.ndarray, mel_np: np.ndarray, guidance, head):
    """
    단일 샘플(메쉬 시퀀스 + 멜 스펙트로그램)에 대해
    lip / real / value score의 평균을 계산하는 함수의 스켈레톤입니다.

    실제 구현에서는:
      - mesh_np (T, V, 3)와 mel_np를 torch.Tensor로 변환
      - 고정 길이(예: 5프레임) 윈도우 단위로 잘라 guidance.forward_features(...) 호출
      - head(...)를 통해 clip 단위 lip / real / value score를 얻은 뒤,
        전체 클립에 대해 평균을 내어 하나의 score로 요약합니다.
    """
    raise NotImplementedError(
        "Private implementation: score_sample 세부 계산 로직은 비공개입니다."
    )


def score_directory(mesh_dir: Path, csv_path: Path, guidance, head, desc: str):
    """
    지정된 디렉터리(mesh_dir)에 있는 .npy 메쉬 파일들에 대해
    reward score를 일괄 계산하고 CSV로 저장하는 함수의 스켈레톤입니다.

    실제 구현에서는:
      - mesh_dir 아래의 *.npy 파일을 순회
      - 각 메쉬 파일 이름에 대응하는 멜 스펙트로그램(.npy)을 MEL_DIR에서 로드
      - score_sample(...)을 호출해 lip / real / value / 사용된 클립 수를 얻음
      - 결과를 rows에 모아서 csv_path에 CSV 형식으로 저장합니다.
    """
    raise NotImplementedError(
        "Private implementation: score_directory 일괄 처리 로직은 비공개입니다."
    )


def main():
    """
    Reward score 계산 파이프라인의 엔트리 포인트 스켈레톤입니다.

    실제 구현에서는:
      1) load_models()로 guidance, head를 로드하고
      2) 예측 결과 디렉터리(PRED_DIR)에 대해 score_directory(...)를 호출
      3) GT 메쉬 디렉터리(GT_DIR)에 대해서도 동일하게 score_directory(...)를 호출하여
         두 CSV(PRED_CSV, GT_CSV)를 생성합니다.
    """
    raise NotImplementedError(
        "Private implementation: main 파이프라인은 데모 레포에서 제공하지 않습니다."
    )


if __name__ == "__main__":
    main()
