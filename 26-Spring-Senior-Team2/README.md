# 🚀 BicDRL-Reproduction: Deep Reinforcement Learning for Imbalanced Medical Image Classification

![BicDRL Architecture Overview](plan/Gemini_Generated_Image_td82lutd82lutd82.png)

## 📌 Project Overview
본 프로젝트는 심층 강화학습(DRL)을 활용하여 의료 이미지 데이터의 극심한 클래스 불균형(Class Imbalance) 문제를 해결하는 **BicDRL 프레임워크**를 PyTorch와 MONAI 기반으로 구현하고 재현하는 것을 목표로 합니다. 

에이전트(Agent)가 소수 클래스(희귀 병변)를 정확히 분류했을 때 더 높은 보상을 주는 **적응형 보상 메커니즘(Adaptive Reward Mechanism)** 을 적용하며, 주로 **NIH Chest X-ray 14** 데이터셋을 활용하여 진행됩니다.

---

## 🏗️ System Pipeline & Architecture

### Phase 1: Data Preparation & Simplification
NIH 데이터셋의 다중 라벨(Multi-label) 특성을 강화학습 에이전트의 단일 행동(Action) 공간에 맞게 단순화합니다.
- **단일 병변 필터링**: 단일 질병(Single-finding) 혹은 정상(No Finding) 케이스로 필터링하여 이진/다중 클래스 평가 환경 구성.
- **적응형 클래스 가중치 계산**: 각 클래스의 전체 샘플 수($c_k$)를 카운트해 보상 함수의 가중치 $\gamma_k = 1 / \ln(c_k + \epsilon)$ 를 사전 연산합니다.

### Phase 2: Medical Image Preprocessing
MONAI를 활용하여 원시 의료 이미지(Raw Data)를 에이전트가 관찰할 상태(State) 텐서로 변환합니다.
- `LoadImaged`, `EnsureChannelFirstd`, 네트워크 입력 크기에 맞춘 `Resized` (224x224), `ScaleIntensityd`, 최종 `EnsureTyped` (PyTorch Tensor).

### Phase 3: DRL Agent Architecture (CNN-based DDQN)
의료 이미지의 공간적 특징(Spatial Features)을 추출하기 위해 Q-Network 앞단에 CNN 백본을 결합한 에이전트를 설계합니다.
- **특징 추출기 (Feature Extractor)**: `torchvision.models.resnet50` 을 백본으로 채택하며, 기존 분류(FC) 레이어를 행동 공간(Action Space) 크기에 맞춘 **Q-Value 출력 레이어**로 교체했습니다.

### Phase 4: Environment & Reward Mechanism
가상 의료 환경 내에서 에이전트의 예측 분류($a_t$)와 실제 정답 라벨($b_t$)을 지속적으로 평가하는 적응형 보상(Reward)을 제공합니다.
- 정답 시 ($a_t = b_t$): $r_t = \gamma_k$ (소수 병변일수록 **더 높은 긍정적 보상**)
- 오답 시 ($a_t \neq b_t$): $r_t = -\gamma_k$ (소수 병변일수록 **더 큰 손실 페널티**)

### Phase 5: Training Loop
에이전트 네트워크 훈련 모듈 구성 요소 (DDQN 알고리즘)
- **탐험과 활용**: Epsilon($\epsilon$)-greedy 전략 체계를 기반으로 한 행동 결정.
- **Replay Buffer**: 에피소드 진행 데이터(State, Action, Reward, Next State)를 보관하며, 추후 Prioritized Experience Replay(PER) 로 적용될 수 있는 구조.
- **Overestimation 방지**: `MainNet` 최적화 및 `TargetNet` 에 대한 주기적 가중치 업데이트 분리.

---

## 📂 Directory Structure

본 코드베이스는 모듈화된 프로젝트 구조를 유지합니다.

```text
BicDRL-Reproduction/
│
├── data/                       # 데이터셋 및 메타데이터 폴더 (Gitignore 처리됨)
│   ├── raw/                    # 원본 의료 이미지 자료 (e.g., NIH Chest X-ray 14)
│   └── Data_Entry_2017.csv     # 정답 라벨이 포함된 원시 메타데이터 CSV 파일
│
├── configs/                    # 실험용 하이퍼파라미터 및 설정 파일 보관 폴더
│   └── config.yaml             # 학습률, 배치 크기, epsilon-decay 등 제반 설정값 통합
│
├── src/                        # 핵심 DRL 프레임워크 소스 코드 모듈
│   ├── __init__.py
│   ├── dataset.py              # 데이터 필터링 통계 분석 및 MONAI DataLoader 반환
│   ├── transforms.py           # 영상 데이터 증강 및 MONAI 전처리 파이프라인
│   ├── networks.py             # CNN 특징 추출기(ResNet50 등) 기반의 통합 네트워크 아키텍처
│   ├── agent.py                # DDQNAgent의 행동 선택 로직 및 파라미터 업데이트 
│   ├── environment.py          # 의료 진단용 MDP 환경 설계 (적응형 보상 함수 반환기)
│   ├── replay_buffer.py        # 이전 예측 기록을 보관 및 반환하는 Experience 재생 버퍼
│   └── utils.py                # 재현성 보장을 위한 Random Seed 제어 프로세스 등
│
├── plan/                       # 기획안 문서 및 에셋 파일 보관소 (Gitignore 처리됨)
│
├── train.py                    # 강화학습 파이프라인의 에이전트 학습 메인 실행 스크립트
├── evaluate.py                 # F1-score, G-mean, Accuracy 등 학습된 모델 모델의 테스트 스크립트
├── requirements.txt            # 실행에 필요한 기초 파이썬 패키지 (PyTorch, MONAI, yaml 등)
└── README.md                   # 프로젝트 개요, 파이프라인 및 실행 가이드 (현재 문서)
```

---

## 🛠️ 시작하기 (Getting Started)

### 1. 패키지 의존성 환경 구성
Python 3.8 이상 구동을 표방하며 다음 명령어로 관련 라이브러리(파이토치, 몬아이 등)를 초기화합니다:
```bash
pip install -r requirements.txt
```

### 2. 데이터셋 구성
- NIH X-ray 데이터셋의 전체 `.png` 이미지 파일들을 `data/raw/` 경로에 위치시킵니다.
- 메타데이터 파일 `Data_Entry_2017.csv`를 최상위 `data/` 경로에 배치시킵니다.

### 3. 학습 및 모델 평가 스크립트 실행
- 엑스레이 이미지를 대상으로 하는 DDQN 에이전트를 가상 환경에서 학습:
```bash
python train.py
```
*(로깅 및 TD 에러 출력 기능이 진행되며, 우수 학습 파일은 `checkpoints/best_model.pth`로 생성됩니다.)*

- 테스트 데이터 셋에 대해 검증 평가 결과 조회:
```bash
python evaluate.py
```
