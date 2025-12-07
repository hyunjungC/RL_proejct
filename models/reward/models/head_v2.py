# from this import d
# from typing import Optional
# import torch
# import torch.nn as nn


# class MLPBlock(nn.Module):
#     def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
#         super().__init__()
#         self.fc = nn.Linear(in_features, out_features)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.dropout(self.relu(self.fc(x)))


# class ScoreHead(nn.Module):
#     """오디오/메시 임베딩 concat → 2층 MLP → 스칼라 score"""
#     def __init__(self, d_audio: int, d_mesh: int,
#                  hidden: int = 64, dropout: float = 0.3,
#                  out_activation: Optional[str] = 'sigmoid'):
#         super().__init__()

#         self.mlp = nn.Sequential(
#             MLPBlock(d_audio + d_mesh, hidden, dropout=dropout),
#         )
#         self.extract_score = nn.Sequential(
#             MLPBlock(hidden, 1, dropout=dropout)
#         )
#         self.out_activation = out_activation

#     def forward(self, vertex_feat, audio_feat) -> torch.Tensor:

#         x = torch.cat([vertex_feat, audio_feat], dim=-1)
#         x = self.mlp(x)

#         lip_score = self.extract_score(x)
#         rea_score = self.extract_score(x)

#         if self.out_activation == "sigmoid":
#             lip_score = torch.sigmoid(lip_score)
#             rea_score = torch.sigmoid(rea_score)
#         elif self.out_activation == "tanh":
#             lip_score = torch.tanh(lip_score)
#             rea_score = torch.tanh(rea_score)
#         # keep graph & batch dim (B,)
#         return lip_score.view(-1), rea_score.view(-1)


from typing import Optional
import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.fc(x)))


class ScoreHead(nn.Module):
    """
    오디오 임베딩 + 메시 임베딩을 concat해서
    - lip-sync score
    - realism score
    - (추가) value V(s): 상태의 가치
    를 뽑는 헤드.

    기본 사용 (기존 코드와 호환):
        lip, real = head(vertex_feat, audio_feat)

    RL에서 Critic까지 쓰고 싶을 때:
        lip, real, value = head(vertex_feat, audio_feat, return_value=True)
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

        in_dim = d_audio + d_mesh

        # 공통 trunk
        self.trunk = nn.Sequential(
            MLPBlock(in_dim, hidden, dropout=dropout),
        )

        # lip-sync 용 head
        self.lip_head = nn.Sequential(
            MLPBlock(hidden, 1, dropout=dropout)
        )

        # realism 용 head
        self.real_head = nn.Sequential(
            MLPBlock(hidden, 1, dropout=dropout)
        )

        # (추가) value V(s) 용 head
        # → Critic에서 reward의 기대값을 예측
        self.value_head = nn.Sequential(
            MLPBlock(hidden, 1, dropout=dropout)
        )

        self.out_activation = out_activation

    def forward(
        self,
        vertex_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        return_value: bool = False,  # ★ 기본 False → 기존 코드와 동일하게 2개만 반환
    ):
        """
        vertex_feat: (B, C_mesh)
        audio_feat:  (B, C_audio)

        return_value=False:
            lip_score, real_score
        return_value=True:
            lip_score, real_score, value
        """
        # (B, C_mesh + C_audio)
        x = torch.cat([vertex_feat, audio_feat], dim=-1)

        # 공통 인코딩
        h = self.trunk(x)

        # 각각 head에서 스칼라 score 추출
        lip_score = self.lip_head(h)   # (B, 1)
        real_score = self.real_head(h) # (B, 1)
        value = self.value_head(h)     # (B, 1)  ← V(s)

        # activation은 reward score에만 적용 (value는 그대로 두는게 일반적)
        if self.out_activation == "sigmoid":
            lip_score = torch.sigmoid(lip_score)
            real_score = torch.sigmoid(real_score)
        elif self.out_activation == "tanh":
            lip_score = torch.tanh(lip_score)
            real_score = torch.tanh(real_score)

        # batch 차원만 남기도록 (B,)
        lip_score = lip_score.view(-1)
        real_score = real_score.view(-1)
        value = value.view(-1)

        if return_value:
            return lip_score, real_score, value
        else:
            # ⚠ 기존 코드와 호환: 2개만 리턴
            return lip_score, real_score
