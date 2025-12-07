import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Speech-to-3D face animation Actor 모델의 데모용 스켈레톤입니다.
    실제 아키텍처와 학습/추론 로직은 비공개입니다.
    """

    def __init__(self, cfg, dataset_name: str, device: str):
        """
        Args:
            cfg: Hydra 설정 객체
            dataset_name: 사용 데이터셋 이름 (예: "vocaset", "BIWI")
            device: "cuda" 또는 "cpu"
        """
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset_name
        self.device = device

        # 실제 네트워크 구성(wav2vec, transformer, style embedding 등)은 공개하지 않습니다.
        raise NotImplementedError(
            "Private implementation: 실제 FaceFormer 모델 구조는 공개 레포에 포함되지 않습니다."
        )

    def forward(
        self,
        audio,
        template,
        vertice,
        one_hot,
        criterion,
        teacher_forcing: bool = True,
        return_dist: bool = False,
    ):
        """
        학습/추론 공통 인터페이스 스켈레톤입니다.

        실제 구현에서는 audio, template, vertice, one_hot을 이용해
        메쉬 시퀀스를 예측하고 supervised loss 및 (옵션) 분포 정보를 반환합니다.
        """
        raise NotImplementedError(
            "Private implementation: forward 로직은 논문 보호를 위해 비공개입니다."
        )

    def predict(self, audio, template, one_hot):
        """
        순수 inference용 인터페이스 스켈레톤입니다.

        실제 구현에서는 입력 음성(audio)과 template, one_hot 정보를 이용해
        예측 메쉬 시퀀스를 반환합니다.
        """
        raise NotImplementedError(
            "Private implementation: predict 로직은 논문 보호를 위해 비공개입니다."
        )

