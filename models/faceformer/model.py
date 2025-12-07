import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from models.wav2vec import Wav2Vec2Model

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, cfg, dataset_name: str, device: str):
        super(Model, self).__init__()
        """
        audio:    (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice:  (batch_size, seq_len, V*3)
        """
        model_cfg = cfg.model
        dataset_cfg = cfg.dataset
        self.dataset = dataset_name

        # ======================
        # 1) Audio encoder (wav2vec2)
        # ======================
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, model_cfg.feature_dim)

        # ======================
        # 2) Motion encoder/decoder
        # ======================
        self.vertice_dim = model_cfg.vertice_dim                     # V*3
        self.vertice_map = nn.Linear(self.vertice_dim, model_cfg.feature_dim)
        self.vertice_map_r = nn.Linear(model_cfg.feature_dim, self.vertice_dim)

        # ======================
        # 3) Positional & temporal bias
        # ======================
        self.PPE = PeriodicPositionalEncoding(model_cfg.feature_dim,
                                              period=model_cfg.period)
        self.biased_mask = init_biased_mask(
            n_head=4, max_seq_len=600, period=model_cfg.period
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_cfg.feature_dim,
            nhead=4,
            dim_feedforward=2 * model_cfg.feature_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=1
        )

        # ======================
        # 4) Style embedding
        # ======================
        self.obj_vector = nn.Linear(
            len(dataset_cfg.train_subjects.split()),
            model_cfg.feature_dim,
            bias=False
        )

        self.device = device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

        # ======================
        # 5) (추가) RL용 stochastic head
        # ======================
        # cfg.model.use_stochastic = True 로 켜고/끄기 가능
        self.use_stochastic = getattr(model_cfg, "use_stochastic", False)
        if self.use_stochastic:
            # vertice_mu (V*3)에 대해 log_sigma 예측하는 head
            self.sigma_head = nn.Linear(self.vertice_dim, self.vertice_dim)
            # 처음에는 거의 deterministic하게 시작하도록 초기화
            nn.init.constant_(self.sigma_head.weight, 0.0)
            nn.init.constant_(self.sigma_head.bias, -1.0)  # sigma ≈ exp(-1) ~ 0.37



    def forward(
        self,
        audio,
        template,
        vertice,
        one_hot,
        criterion,
        teacher_forcing: bool = True,
        return_dist: bool = False,   # ★ 추가: RL에서 분포/log_prob까지 쓸 때 True로
    ):
        """
        return_dist=False (기본값):
            -> 예전과 동일하게 (vertice_out, loss) 만 반환
        return_dist=True:
            -> (vertice_mu, vertice_sample, loss, dist) 반환
               (RL Actor에서 사용)
        """

        # ======================
        # 1) 입력 준비
        # ======================
        template = template.unsqueeze(1)  # (B, 1, V*3)
        obj_embedding = self.obj_vector(one_hot)  # (B, feature_dim)
        frame_num = vertice.shape[1]

        # ======================
        # 2) 오디오 인코딩
        # ======================
        hidden_states = self.audio_encoder(
            audio, self.dataset, frame_num=frame_num
        ).last_hidden_state  # (B, T_audio, 768)

        if self.dataset == "BIWI":
            # BIWI에서 오디오 길이가 짧을 때 mesh 길이 조정
            if hidden_states.shape[1] < frame_num * 2:
                vertice = vertice[:, :hidden_states.shape[1] // 2]
                frame_num = hidden_states.shape[1] // 2

        hidden_states = self.audio_feature_map(hidden_states)  # (B, T_audio, feature_dim)

        # ======================
        # 3) 디코딩 (teacher forcing vs autoregressive)
        # ======================
        if teacher_forcing:
            # ---- teacher forcing branch ----
            vertice_emb = obj_embedding.unsqueeze(1)  # (B,1,feature_dim)
            style_emb = vertice_emb

            # GT를 한 스텝씩 밀어서 입력으로 사용
            vertice_input = torch.cat((template, vertice[:, :-1]), dim=1)  # (B, T, V*3)
            vertice_input = vertice_input - template         # template 기준 displacement
            vertice_input = self.vertice_map(vertice_input)  # (B, T, feature_dim)
            vertice_input = vertice_input + style_emb        # style add
            vertice_input = self.PPE(vertice_input)

            tgt_mask = self.biased_mask[
                :, :vertice_input.shape[1], :vertice_input.shape[1]
            ].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(
                self.device, self.dataset,
                vertice_input.shape[1], hidden_states.shape[1]
            )
            vertice_out = self.transformer_decoder(
                vertice_input, hidden_states,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
            vertice_out = self.vertice_map_r(vertice_out)  # (B, T, V*3)

        else:
            # ---- autoregressive branch ----
            for i in range(frame_num):
                if i == 0:
                    vertice_emb = obj_embedding.unsqueeze(1)  # (B,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)

                tgt_mask = self.biased_mask[
                    :, :vertice_input.shape[1], :vertice_input.shape[1]
                ].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(
                    self.device, self.dataset,
                    vertice_input.shape[1], hidden_states.shape[1]
                )
                vertice_out = self.transformer_decoder(
                    vertice_input, hidden_states,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask
                )
                vertice_out = self.vertice_map_r(vertice_out)
                new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), dim=1)

        # ======================
        # 4) template 더해서 absolute mesh로 만들기
        # ======================
        vertice_out = vertice_out + template           # (B, T, V*3)
        vertice_mu = vertice_out                       # 해석상 "mean"으로 사용

        # ======================
        # 5) (옵션) stochastic head로 action sampling
        # ======================
        if self.use_stochastic:
            # vertice_mu: (B, T, V*3) → log_sigma: 같은 shape
            log_sigma = self.sigma_head(vertice_mu)    # (B, T, V*3)
            # 수치 폭주 방지
            log_sigma = torch.clamp(log_sigma, min=-5.0, max=2.0)
            sigma = torch.exp(log_sigma)               # (B, T, V*3) > 0

            dist = torch.distributions.Normal(vertice_mu, sigma)
            # rsample로 reparameterization (RL에서 gradient 잘 흘러가게)
            vertice_sample = dist.rsample()
        else:
            # 기존 동작 유지용 (deterministic)
            dist = None
            vertice_sample = vertice_mu

        # ======================
        # 6) supervised loss (예전과 동일)
        # ======================
        loss = criterion(vertice_mu, vertice)   # (B, T, V*3)
        loss = torch.mean(loss)

        # ======================
        # 7) 반환 형태
        # ======================
        if return_dist:
            # RL에서 actor_loss, log_prob까지 쓰고 싶을 때
            return vertice_mu, vertice_sample, loss, dist
        else:
            # 기존 trainer와 호환용 (예전과 동일한 형태)
            return vertice_mu, loss


    def predict(self, audio, template, one_hot):
        """
        inference용: 그대로 deterministic mesh 생성 (기존과 동일)
        """
        template = template.unsqueeze(1)  # (B,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1] // 2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i == 0:
                vertice_emb = obj_embedding.unsqueeze(1)  # (B,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[
                :, :vertice_input.shape[1], :vertice_input.shape[1]
            ].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(
                self.device, self.dataset,
                vertice_input.shape[1], hidden_states.shape[1]
            )
            vertice_out = self.transformer_decoder(
                vertice_input, hidden_states,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), dim=1)

        vertice_out = vertice_out + template
        return vertice_out
