```markdown
# RL-project
```

#  🎮 Speech-to-3D Face Animation 

---
## ✅ 0. Scope & Audience
> 이 저장소는 **기술 공개 범위를 최소화한 데모용 구조(skeleton)**입니다.  
> 실제 모델, 학습 코드, 체크포인트, 데이터셋은 **포함되어 있지 않습니다.**  
> 본 문서는 **전체 파이프라인 이해**를 위한 요약이며,  
> 세부 구조·차원·알고리즘은 연구 보호를 위해 공개하지 않습니다.



## ✅1. Overview ([🔗 : 발표자료](https://www.canva.com/design/DAG6h43SCnU/Bg0qUI8bYz8JZkyo-mPMCg/edit?utm_content=DAG6h43SCnU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) )


<img width="1345" height="479" alt="image" src="https://github.com/user-attachments/assets/eb7943ce-6e47-4447-a66b-6fff0682a755" />




## ✅2. Project Goals


본 프로젝트의 최종 목표는 입력 음성에 맞추어 자연스럽게 움직이는 3D 얼굴 메쉬/영상을 생성하는 것입니다.
단순히 vertex 위치 오차(MSE)를 줄이는 것만으로는 사람이 느끼는 자연스러움과 일치하지 않기 때문에,
본 연구는 **지각적 품질(perceptual quality)**을 최우선으로 둔 학습 파이프라인을 구축합니다.

##### ✔ 정량적 오차 최소화(MSE, L2) 자체가 목적이 아니라,

##### ✔ 사람이 보았을 때 '자연스럽다'고 느끼는 입모양, 얼굴 움직임, 타이밍, 표현력


을 생성하는 데 초점을 둡니다.
이를 위해 Actor 모델(FaceFormer)의 출력에 대해
Reward 모델을 활용하여 lip realism·motion naturalness 기반의 강화학습 신호를 부여함으로써,
정확도 중심의 supervised learning을 넘어 더 사람다운(face-like) 움직임을 학습하도록 설계했습니다.

## ✅ 3. Dataset & Data Structure (High-Level)

이 프로젝트는 speech-to-3D mesh 작업을 위해 다음과 같은 데이터 구조를 사용합니다:

   - Raw audio (.wav): `vocaset/wav/`
   - Mel-spectrogram (.npy): `vocaset/wav_npy/`
   - Ground-truth vertex sequences: `vocaset/vertices_npy/`
   - Template mesh: `vocaset/templates.pkl`
   - (Optional, test) FLAME face region masks: `vocaset/FLAME_masks.pkl`
   - Reward model checkpoints are loaded from `checkpoints/reward/`
   - 모든 출력 및 학습 체크포인트는 `checkpoints/<wandb_name>/` 아래 저장됩니다.



## ✅ 4. Main 파이프라인 흐름 및 요약 (`/workspace/RL-VOCASET/main.py` 기준)

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
       
   ### 1) Main Workflow Summary (`python main.py`)
   모델을 실행하면 아래와 같은 순서로 파이프라인이 진행됩니다.
   
   #### ✔ Config 로드 → 데이터 준비 → Actor 모델 추론 → Reward 평가 → RL 학습 → Validation/Test
   
   각 과정의 목적만 설명하며 **구체적인 구현, 수식, 차원 정보는 포함하지 않습니다.**
   
   ---
   
   ### 2) 🔧 Config & Environment Loading
   - Hydra 기반 설정 로딩
   - 모델 종류, 데이터셋 타입, 학습/평가 옵션을 포함한 high-level configuration
   - 실제 training parameter, architecture detail은 비공개
   
   ---
   
   ### 3) 🎧 Data Loader (Audio & Mesh Preparation)
   입력으로 사용되는 데이터는 다음과 같은 형태로 구성됩니다.
   
   ##### 포함 요소
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
   
   ##### 비공개 요소
   - 데이터 차원 및 내부 전처리 로직
   - wav2vec/mesh loader 등의 구체 구현
   
   ---
   
   ### 4) 🧑‍🏫 Actor Model (Face Animation Generator)
   Actor 모델은 다음 기능을 수행합니다.
   
   ##### 모델 역할
   - 음성 인코더가 추출한 speech embedding과  
     template mesh · subject 정보 등을 결합하여  
     **타임라인에 따른 3D 얼굴 메쉬 시퀀스를 생성**합니다.
   
   
   ##### 비공개(연구 보호) 처리
   - 인코더 구조, 차원, attention/bias 구조, normalization 방식
   - mesh representation 차원
   - distribution 기반 sampling 공식
   - loss function의 구체적인 조합 및 계산 과정
   
   ---
   
   ### 5) ⭐ Reward Model (Freeze된 품질 평가 네트워크)
   학습 시 actor가 생성한 mesh를 평가하기 위해 **별도의 품질 측정 모델**을 사용합니다.
   
   ##### 기능
   - 오디오–메쉬 간 **lip-sync**, **realism** 등을 측정  
   - freeze 상태로 사용되며, actor 업데이트를 위한 reward를 제공
   
   ##### 비공개 처리
   - backbone/score-head architecture  
   - 입력 clip 형식, 차원  
   - score 계산 방식  
   
   ---
   
   ### 6) 🧠 Reinforcement Learning Loop (High-Level Description Only)
   본 프로젝트는 supervised loss와 reinforcement signal을 함께 사용합니다.
   
   ##### 전체 흐름
   1) Actor가 예측 메쉬 생성  
   2) Reward 모델이 품질 점수 계산  
   3) Critic이 예측 안정성을 돕는 방향으로 평가  
   4) Actor는 높은 품질 방향으로 업데이트  
   5) Critic은 reward를 더 잘 예측하도록 업데이트  
   6) Supervised 학습과 RL 신호가 함께 최종 loss 구성
   
   ##### 비공개 처리
   - advantage 계산식  
   - actor/critic loss 공식  
   - weight/scale 값  
   - log_prob 기반 정책 업데이트 방식  
   
   ---
   
   ### 7) 🧪 Validation
   - baseline model(faceformer)가 mesh를 생성하여 품질 지표를 기록  
   
   
   ---
   
   ### 8) 🧫 Test / Inference (Demo Version)
   - best checkpoint가 주어질 경우 전체 subject 조건에서 메쉬 생성  
   - 결과는 .npy 로 저장 가능  
   - public repo에는 checkpoint가 포함되지 않음음
   
   ---
   
   ### 🚫 포함되지 않는 항목 (Important)
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

## 📁 5 .Repository Structure (Demo Skeleton)
```
RL-VOCASET/
├── checkpoints/              # (비어 있음) 실제 모델 가중치는 포함되지 않음
├── configs/                  # Hydra 기반 설정 파일 구조
│   ├── dataset/              # 데이터 로딩 관련 옵션 
│   ├── model/                # 모델 설정 placeholder
│   ├── trainer/              # 학습/평가 설정 placeholder
│   └── config.yaml           # 기본 엔트리 설정
├── dataset/                  # VOCASET 스타일 데이터 로더 인터페이스
├── models/                   # 모델 인터페이스 구조
│   ├── faceformer/           # Actor(생성 모델) 인터페이스(오디오 → 메쉬).
│   └── reward/               # Reward 평가 모델 인터페이스
├── src/                     
├── trainer/                  # 학습/검증/평가 흐름 skeleton
├── utils/                    
├── vocaset/                  # (비어 있음) 실제 데이터 포함되지 않음
├── .gitignore
├── main.py                   # Hydra 엔트리 포인트
├── README.md
├── render.py                 # 결과 렌더링/시각화용 placeholder
└── reward_score.py           # Reward 계산 

```






### 📌 Notes for Reviewers / Collaborators

* 본 문서는 연구 파이프라인 설명용 문서이며,
실제 모델을 재현할 수 있는 정보는 포함하지 않습니다.

* 요청 시 full version은 private 저장소 및 오프라인 환경에서만 공유됩니다.

* 본 repo는 학습/평가 실행이 불가능하며, 구조 이해와 데모 목적만 수행합니다.












