```markdown
# RL-project
```

#  🎮 FaceFormer 강화학습 프로젝트 

본 프로젝트의 최종 목표는 입력 음성에 맞추어 자연스럽게 움직이는 3D 얼굴 메쉬/영상을 생성하는 것입니다.
단순히 vertex 위치 오차(MSE)를 줄이는 것만으로는 사람이 느끼는 자연스러움과 일치하지 않기 때문에,
본 연구는 **지각적 품질(perceptual quality)**을 최우선으로 둔 학습 파이프라인을 구축합니다.

##### ✔ 정량적 오차 최소화(MSE, L2) 자체가 목적이 아니라,

##### ✔ 사람이 보았을 때 '자연스럽다'고 느끼는 입모양, 얼굴 움직임, 타이밍, 표현력

을 생성하는 데 초점을 둡니다.
이를 위해 Actor 모델(FaceFormer)의 출력에 대해
Reward 모델을 활용하여 lip realism·motion naturalness 기반의 강화학습 신호를 부여함으로써,
정확도 중심의 supervised learning을 넘어 더 사람다운(face-like) 움직임을 학습하도록 설계했습니다.


```bash
✅
```

## ✅파일 역할 (핵심 시그니처만 노출)
```bash
- `main.py` : Hydra 엔트리. config 읽고 데이터로더/모델/리워드/트레이너 객체를 엮어 train/test 실행.
- `dataset/dataloader_style.py` : VOCASET 스타일 데이터 로더 인터페이스.
- `models/faceformer/model.py` : FaceFormer 모델 인터페이스(오디오 → 메쉬).
- `models/wav2vec.py` : Wav2Vec2 오디오 인코더 래퍼 인터페이스.
- `models/reward/models/modeling.py` : SpeechMeshTransformer 백본 인터페이스.
- `models/reward/models/head_v2.py` : Reward/critic 헤드 인터페이스.
- `trainer/faceformer/trainer.py` : 학습/검증/테스트 함수 시그니처만 남긴 트레이너.
- `src/utils.py` : 공용 유틸(로깅 등).
```


## ✅실제 파이프라인 흐름 ( `/workspace/RL-VOCASET_my_copy_check_3/main.py` 기준 )
1) **Hydra 설정 로드**  
   - `configs/config.yaml` 기본값: model=faceformer, dataset=style, trainer=faceformer
2) **데이터 로더 준비**  
   - `dataset/dataloader_style.py` 로 wav → mel, vertices, template 로드
3) **Actor 모델 빌드**  
   - `models/faceformer/model.py` (오디오 인코더 `models/wav2vec.py` 포함)
4) **Reward 백본/헤드 로드**  
   - `models/reward/models/modeling.py` (SpeechMeshTransformer)  
   - `models/reward/models/head_v2.py` (lip/real/value 헤드)
5) **학습 루프**  
   - `trainer/faceformer/trainer.py`에서 sup loss + RL(actor/critic) 조합 학습
6) **테스트**  
   - 같은 트레이너에서 style-dependent 테스트 수행
7) **저장**  
   - `checkpoints/{wandb_name}/best.pt` 등에 모델/헤드 저장




##  ✅Main 파이프라인 요약 (데이터 차원 포함)

## `python main.py` 실행 시 전체 흐름:

### 1) 🔧 Config 로드 (Hydra)
 - configs/config.yaml 불러오기
 - defaults:
     model: faceformer
     dataset: style
     trainer: faceformer
 - cfg.train / cfg.test 플래그에 따라 학습·평가 실행


### 2) 데이터 로딩 — dataset/dataloader_style.py
 -------------------------------------------------
- WAV 로드 → Wav2Vec2Processor 입력값 생성
audio: (T_audio, )

- Mel 특징(rep_audio_mel) — Reward 모델용
rep_audio_mel: (T_clip, 1, 20, 128)

- Template mesh
template: (5023, 3) → flatten → (15069,)

- GT vertex 시퀀스 (2프레임 샘플링 적용)
vertice: (seq_len, 15069)

- Subject one-hot
one_hot_train:     (num_speakers,)
one_hot_val_test:  (num_speakers_all, num_speakers)

- DataLoader 배치 (batch=1)
batch_audio:   (1, T_audio)
batch_vertice: (1, seq_len, 15069)
batch_template:(1, 15069)
batch_onehot:  (1, num_speakers)
batch_rep_mel: rep_audio_mel 그대로


### 3) Actor Model (FaceFormer) — models/faceformer/model.py
 -----------------------------------------------------------
- Audio Encoder
wav2vec2 → (1, T_audio', 768) → Linear → (1, T_audio', 64)

- Transformer Decoder 입력:
 - template displacement
 - style embedding(one_hot)
 - PPE, temporal bias 등 포함

- 출력:
vertice_mu:     (1, seq_len, 15069)     # mean
vertice_sample: (1, seq_len, 15069)     # stochastic mode일 때
dist: Normal(μ,σ)                       # log_prob 계산용

- Supervised Loss:
sup_loss = MSE(pred, GT)


### 4) Reward Backbone (고정) — SpeechMeshTransformer
 ---------------------------------------------------
- 입력:
mesh_clip: (B, 5, 15069)
mel_clip:  (B, 1, 20, 128)

- 출력 임베딩:
vertex_feat: (B, 512)
audio_feat:  (B, 512)

- ckpt 로드 후 freeze, eval 모드.


### 5) Score Head — head_v2.py
 --------------------------------
- 입력:
concat_feat: (B, 512 + 512)

- 출력:
lip_score:  (B,)    # sigmoid
real_score: (B,)    # sigmoid
value:      (B,)    # critic V(s)

 head ckpt는 학습 대상 (requires_grad=True)


### 6) 학습 루프 — trainer/faceformer/trainer.py::train
 ------------------------------------------------------

 (1) Actor forward
vertice_mu, vertice_sample, sup_loss

 (2) Reward 계산
 rep_audio_mel → 5프레임 단위 슬라이딩 → backbone → head
lip, real, value = mean over clips

reward    = (lip + real) * reward_scale
advantage = reward - value

actor_loss  = -advantage * mean(log_prob(sample))     # stochastic인 경우
critic_loss = (reward - value) ** 2

total_loss = sup_loss \
           + actor_weight  * actor_loss \
           + critic_weight * critic_loss

 Optimizer: Actor + Head 업데이트


### 7) Validation
 ------------------------------------------------------
 - Actor deterministic forward
 - LVE(mouth vertex error) 계산
 - best 성능 시 ckpt 저장:
     checkpoints/<wandb_name>/best.pt
     checkpoints/<wandb_name>/best_head.pt


### 8) Test — trainer/faceformer/trainer.py::test
 ------------------------------------------------------
 - best ckpt 로드
 - 모든 subject one-hot 조건별로 예측 mesh npy 저장:
   checkpoints/<wandb_name>/styledependant/results/*.npy
 - LVE / FDD 계산 출력
 - style-independent 모드: 모든 one-hot 평균값 사용



## 포함되지 않는 것
- 실제 학습/추론 구현 상세, RL/Reward 내부 로직
- 체크포인트(.pt), 데이터(wav, mel npy, vertices npy, templates, masks)

## 실제 데이터/체크포인트를 쓸 경우 필요한 경로 (참고용)
 오디오: `vocaset/wav/`  
- 멜 스펙트럼: `vocaset/wav_npy/`  
- 메쉬 GT: `vocaset/vertices_npy/`  
- 템플릿/마스크: `vocaset/templates.pkl`, `vocaset/FLAME_masks.pkl`  
- 리워드 백본/헤드 ckpt: `checkpoints/reward/model_loss.pth`, `checkpoints/reward/v4_best.pt`  
데모 리포에는 위 파일이 포함되지 않습니다.






