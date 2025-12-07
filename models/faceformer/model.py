import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Speech-to-3D face animation Actor 모델의 데모용 스켈레톤입니다.
    실제 아키텍처, 차원, 학습/추론 로직은 공개하지 않습니다.
    """

    def __init__(self, cfg, dataset_name: str, device: str):
        """
        Args:
            cfg: Hydra 구성 객체 (cfg.model, cfg.dataset 등)
            dataset_name: 사용 데이터셋 이름 (예: "vocaset", "BIWI")
            device: 모델이 올라갈 디바이스 문자열 (예: "cuda", "cpu")
        """
        super().__init__()
        # 실제 네트워크 구성, wav2vec 연동, style 임베딩 등은 비공개입니다.
        raise NotImplementedError(
            "Private implementation: 실제 FaceFormer 아키텍처는 공개 레포에 포함되지 않습니다."
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
        학습/추론 공통 인터페이스 스켈레톤.

        Args:
            audio: 음성 입력 텐서 (raw wav 또는 전처리된 형태)
            template: 템플릿 메쉬 (batch, V*3)
            vertice: GT 메쉬 시퀀스 (batch, T, V*3)
            one_hot: 화자/스타일 one-hot 벡터
            criterion: 감독 학습용 손실 함수
            teacher_forcing: teacher forcing 사용 여부
            return_dist: RL용 분포/샘플 반환 여부 (실제 구현은 비공개)

        Returns:
            실제 구현에서는 예측 메쉬 및 손실 등을 반환하지만,
            데모 스켈레톤에서는 구현을 제공하지 않습니다.
        """
        raise NotImplementedError(
            "Private implementation: forward 로직은 논문 보호를 위해 비공개입니다."
        )

    def predict(self, audio, template, one_hot):
        """
        순수 추론(inference)용 인터페이스 스켈레톤.

        Args:
            audio: 음성 입력 텐서
            template: 템플릿 메쉬
            one_hot: 화자/스타일 one-hot 벡터

        Returns:
            실제 구현에서는 예측 메쉬 시퀀스를 반환하지만,
            데모 스켈레톤에서는 구현을 제공하지 않습니다.
        """
        raise NotImplementedError(
            "Private implementation: predict 로직은 논문 보호를 위해 비공개입니다."
        )
