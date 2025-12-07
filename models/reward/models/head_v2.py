from typing import Optional
import torch
import torch.nn as nn


class ScoreHead(nn.Module):
    """
    Skeleton ScoreHead for public/demo version.

    - 오디오 임베딩 + 메쉬 임베딩을 입력으로 받아
      lip-sync / realism / value score를 낸다는 "역할"만 정의합니다.
    - 실제 네트워크 구조 및 계산 방식은 연구 보호를 위해 제거되었습니다.

    기본 사용 (기존 코드와 호환 시그니처 유지):

        lip, real = head(vertex_feat, audio_feat)
        lip, real, value = head(vertex_feat, audio_feat, return_value=True)

    이 공개 버전에서는 forward 호출 시 NotImplementedError를 발생시킵니다.
    """

    def __init__(
        self,
        d_audio: int,
        d_mesh: int,
        hidden: int = 64,
        dropout: float = 0.3,
        out_activation: Optional[str] = "sigmoid",
    ):
        super().__init__()
        # NOTE:
        #   - 공개 버전에서는 실제 파라미터/레이어를 포함하지 않습니다.
        #   - 필요한 경우, private 레포에서만 실제 구현을 채워 넣으세요.
        self.d_audio = d_audio
        self.d_mesh = d_mesh
        self.hidden = hidden
        self.dropout = dropout
        self.out_activation = out_activation

    def forward(
        self,
        vertex_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        return_value: bool = False,
    ):
        """
        vertex_feat: (B, C_mesh)
        audio_feat:  (B, C_audio)

        return_value=False:
            lip_score, real_score
        return_value=True:
            lip_score, real_score, value

        공개 skeleton 버전에서는 실제 계산을 수행하지 않습니다.
        """
        raise NotImplementedError(
            "ScoreHead is a skeleton in the public/demo repo. "
            "Implementation is removed for research protection."
        )

