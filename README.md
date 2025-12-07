# 🚀 RL-VOCASET 

이 저장소는 RL-VOCASET의 전체 코드, Docker 환경, Checkpoint 다운로드 링크,  
그리고 학습 및 추론 방법을 포함하고 있습니다.  

> ⚠️ **주의:** 이 저장소는 직접 수정하지 마세요!  
> 작업 시 반드시 **Fork** 후 개인 저장소에서 진행하세요.
---

# 💪 TODO List  
- [X] style(codetalker, faceformer), no-style(selftalk) 이해 및 Metric
- [ ] Add Codetalker
- [ ] Checkpoints Huggingface upload
- [ ] Datasets Huggingface Upload
- [ ] Reinforcement Learning Optimization (PPO)
- [ ] Reinforcement Learning Optimization (GRPO)
- [ ] Reward Model Dataset build Code
- [ ] Reward Model Training Code
- [ ] Reward Model 개선


## 📘 1. GitHub 사용법

### 🔹 저장소 Fork 및 Clone
1. 상단의 **Fork** 버튼을 클릭하여 개인 계정으로 복제합니다.
2. Fork된 저장소에서 아래 명령어로 클론합니다.
   ```bash
   git clone https://github.com/<your-username>/RL-VOCA.git

## 📘 2. Environment Setting
   ```bash
   docker pull esh0504/project:RL-VOCASET
   docker run -it --gpus all -v RL-VOCASET:/workspace -v /data/vocaset:/data/vocaset esh0504/project:RL-VOCA
   ```
## 📘 3. Reward Model Checkpoints
- encoder: [link](https://drive.google.com/file/d/10bYZp4-O23HFdriY7AfF3iYn8LH5vOfn/view?usp=drive_link)
- head: [link](https://drive.google.com/file/d/1V4yeorO4buESqAnwnzow9dddRrKyLXA6/view?usp=drive_link)

## 📘 4. Train (you can setting your config file (in configs/config.yaml).
python main.py

