```markdown
# RL-project
```
```
> 이 저장소는 **기술 공개 범위를 최소화한 데모용 구조(skeleton)**입니다.  
> 실제 모델, 학습 코드, 체크포인트, 데이터셋은 **포함되어 있지 않습니다.**  
> 본 문서는 **전체 파이프라인 이해**를 위한 요약이며,  
> 세부 구조·차원·알고리즘은 연구 보호를 위해 공개하지 않습니다.
```


#  🎮 Speech-to-3D Face Animation


본 프로젝트의 최종 목표는 입력 음성에 맞추어 자연스럽게 움직이는 3D 얼굴 메쉬/영상을 생성하는 것입니다.
단순히 vertex 위치 오차(MSE)를 줄이는 것만으로는 사람이 느끼는 자연스러움과 일치하지 않기 때문에,
본 연구는 **지각적 품질(perceptual quality)**을 최우선으로 둔 학습 파이프라인을 구축합니다.

##### ✔ 정량적 오차 최소화(MSE, L2) 자체가 목적이 아니라,

##### ✔ 사람이 보았을 때 '자연스럽다'고 느끼는 입모양, 얼굴 움직임, 타이밍, 표현력

을 생성하는 데 초점을 둡니다.
이를 위해 Actor 모델(FaceFormer)의 출력에 대해
Reward 모델을 활용하여 lip realism·motion naturalness 기반의 강화학습 신호를 부여함으로써,
정확도 중심의 supervised learning을 넘어 더 사람다운(face-like) 움직임을 학습하도록 설계했습니다.

<img width="1345" height="479" alt="image" src="https://github.com/user-attachments/assets/eb7943ce-6e47-4447-a66b-6fff0682a755" />





##  ✅Main 파이프라인 흐름 및 요약 (`/workspace/RL-VOCASET/main.py` 기준)

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
##  1. Main Workflow Summary (`python main.py`)
모델을 실행하면 아래와 같은 순서로 파이프라인이 진행됩니다.

Config 로드 → 데이터 준비 → Actor 모델 추론 → Reward 평가 → RL 학습 → Validation/Test

각 과정의 목적만 설명하며 **구체적인 구현, 수식, 차원 정보는 포함하지 않습니다.**

---

# 2. 🔧 Config & Environment Loading
- Hydra 기반 설정 로딩
- 모델 종류, 데이터셋 타입, 학습/평가 옵션을 포함한 high-level configuration
- 실제 training parameter, architecture detail은 비공개

---

# 3. 🎧 Data Loader (Audio & Mesh Preparation)
입력으로 사용되는 데이터는 다음과 같은 형태로 구성됩니다.

### 포함 요소
- **음성 신호**  
  - waveform → 음성 인코더가 처리할 수 있는 embedding으로 변환
- **멜 스펙트럼 특징**  
  - Reward 모델에서 품질 평가에 참고되는 보조 오디오 표현
- **Template Mesh**  
  - 한 얼굴 template에 대해 displacement를 예측하는 방식
- **Ground Truth Mesh Sequence**  
  - time-aligned mesh 시퀀스
- **Subject Embedding**  
  - 화자 조건 부여(원핫 or 임베딩)

### 비공개 요소
- 데이터 차원 및 내부 전처리 로직
- wav2vec/mesh loader 등의 구체 구현

---

# 4. 🧑‍🏫 Actor Model (Face Animation Generator)
Actor 모델은 다음 기능을 수행합니다.

### 모델 역할
- 음성 인코더가 추출한 speech embedding과  
  template mesh · subject 정보 등을 결합하여  
  **타임라인에 따른 3D 얼굴 메쉬 시퀀스를 생성**합니다.
- deterministic / stochastic 두 가지 모드로 실행 가능

### 비공개(연구 보호) 처리
- 인코더 구조, 차원, attention/bias 구조, normalization 방식
- mesh representation 차원
- distribution 기반 sampling 공식
- loss function의 구체적인 조합 및 계산 과정

---

# 5. ⭐ Reward Model (Freeze된 품질 평가 네트워크)
학습 시 actor가 생성한 mesh를 평가하기 위해 **별도의 품질 측정 모델**을 사용합니다.

### 기능
- 오디오–메쉬 간 **lip-sync**, **realism**, **temporal consistency** 등을 측정  
- freeze 상태로 사용되며, actor 업데이트를 위한 reward를 제공

### 비공개 처리
- backbone/score-head architecture  
- 입력 clip 형식, 차원  
- score 계산 방식  

---

# 6. 🧠 Reinforcement Learning Loop (High-Level Description Only)
본 프로젝트는 supervised loss와 reinforcement signal을 함께 사용합니다.

### 전체 흐름
1) Actor가 예측 메쉬 생성  
2) Reward 모델이 품질 점수 계산  
3) Critic이 예측 안정성을 돕는 방향으로 평가  
4) Actor는 높은 품질 방향으로 업데이트  
5) Critic은 reward를 더 잘 예측하도록 업데이트  
6) Supervised 학습과 RL 신호가 함께 최종 loss 구성

### 비공개 처리
- advantage 계산식  
- actor/critic loss 공식  
- weight/scale 값  
- log_prob 기반 정책 업데이트 방식  

---

# 7. 🧪 Validation
- deterministic 모드로 mesh를 생성하여 품질 지표를 기록  
- LVE·FDD 등 일부 지표는 내부 reference에 의해 계산되나 공개 범위 최소화

---

# 8. 🧫 Test / Inference (Demo Version)
- best checkpoint가 주어질 경우 전체 subject 조건에서 메쉬 생성  
- 결과는 .npy 또는 .obj 등 다양한 포맷으로 저장 가능  
- public repo에는 checkpoint가 포함되지 않으며, demo용 dummy predictor만 제공됨

---

# 🚫 포함되지 않는 항목 (Important)
본 공개 버전에는 아래 파일/구현이 절대 포함되지 않습니다.

- 실제 학습된 모델 weight (*.pt, *.pth)
- wav2vec 기반 음성 인코더 구현
- mesh reconstruction/decoder 내부 구조
- reward backbone 및 score head 상세 구조
- reinforcement learning 공식, weighting, optimizer 세부 로직
- 데이터셋(wav, mel, vertices, template, mask)
- mask index, facial region mapping 등 research-critical 정보
- 논문 구현과 직접 연결되는 차원/수식/알고리즘

---

## 📁 Repository Structure (Demo Skeleton)
```
RL-VOCASET/
├── pycache/
├── checkpoints/
├── configs/
│ ├── dataset/
│ ├── model/
│ ├── trainer/
│ └── config.yaml
├── dataset/
├── models/
│ ├── faceformer/
│ └── reward/
├── src/
├── trainer/
├── utils/
├── vocaset/
├── .gitignore
├── main.py 
├── README.md
├── render.py
└── reward_score.py
```


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




###📌 Notes for Reviewers / Collaborators

* 본 문서는 연구 파이프라인 설명용 문서이며,
실제 모델을 재현할 수 있는 정보는 포함하지 않습니다.

* 요청 시 full version은 private 저장소 및 오프라인 환경에서만 공유됩니다.

* 본 repo는 학습/평가 실행이 불가능하며, 구조 이해와 데모 목적만 수행합니다.

```bash
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

```




