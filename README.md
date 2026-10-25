# RareZoom-RL

**Cost-Aware Reinforcement Learning for Adaptive High-Resolution Region Exploration in Imbalanced Chest X-ray Classification**

클래스 불균형 흉부 X-ray에서, 제한된 추가 연산 예산 안에서 강화학습 에이전트가 고해상도로 다시 볼 영역을 고르는 연구 설계입니다. 기존 BicDRL의 ResNet50과 Double DQN을 재사용하고, 행동의 의미를 질환 클래스에서 영역 선택으로 바꿉니다.

상태: **메인 가설은 비용 인지 적응형 해상도이고, 확인 실험(시드 44)에서 기각됐습니다.** 모든 영상을 224로 먼저 보고 학습된 가치가 높은 영상만 512로 다시 보는 설계입니다. ROI 영역 선택 DDQN도 이전 실험에서 음성 결과였습니다. 권장 출발 설정은 [`configs/rarezoom.yaml`](configs/rarezoom.yaml)에 있습니다.

## 결론

NIH ChestX-ray14에서 고해상도를 더 쓰는 세 접근 모두, 통제된 비교에서 성능 향상을 보이지 못했습니다.

| 접근 | 공식 테스트 결과 |
| --- | --- |
| ROI 영역 선택 DDQN | 같은 확대 횟수의 무작위와 같음(0.2275 대 0.2274), 확대 없음(0.2343)보다 낮음 |
| 전역 입력 512 | 배치를 32로 맞추면 Macro-AUPRC 차이 +0.001–0.007, 두 시드 모두 95% 구간이 0을 포함. 연산은 5.2배 |
| 적응형 해상도(224 후 일부 512) | 확인 시드에서 검증이 재판독 0%를 고름. 테스트 곡선도 무작위와 비슷. 탐색 시드의 이득은 앙상블 효과 |

1. 처음 관찰한 "512가 Macro-AUPRC +0.028 높다"의 대부분은 배치 크기 차이(256 대 32)였습니다.
2. 고해상도 재판독의 작은 이득은 다른 시드의 224 모델 하나를 섞어도 같게 나옵니다.
3. 라벨을 아는 오라클은 큰 여유분을 보이지만, 라벨 없이 224 출력으로 고르는 정책은 그 여유분을 찾지 못했습니다.

## 연구 질문

> 제한된 추가 연산 예산에서 강화학습이 희귀·국소 병변에 유리한 영역을 선택하여, 같은 예산의 비교 방법보다 높은 다중 라벨 진단 성능을 낼 수 있는가?

BicDRL의 질문은 “희귀 질환에 더 큰 보상을 주면 분류 성능이 좋아지는가?”였습니다. RareZoom-RL은 “희귀 질환을 놓치지 않기 위해 영상의 어느 부분을, 몇 번, 얼마나 자세히 볼 것인가?”로 질문을 옮깁니다.

영역 확대에 강화학습을 쓰는 것 자체는 새롭지 않습니다. PatchDrop(CVPR 2020)은 선택적 고해상도 분석을, Ypsilantis와 Montana(2017)는 흉부 X-ray의 순차적 시각 주의를 제안했습니다. MedicalPatchNet은 패치 분류로 위치 해석을 다룹니다. 이 설계의 차별점은 **클래스별 희귀성, 미탐지 위험, 추가 분석 비용**을 한 순차 정책의 보상에 함께 넣는 데 있습니다.

## 동작

NIH ChestX-ray14 원본은 1024×1024입니다. 확대 영역은 반드시 이 원본에서 잘라 냅니다. 이미 224×224로 줄인 영상을 다시 키우면 고해상도 분석이 아닙니다.

1. **저해상도 1차 분류.** 1024×1024를 224×224로 줄여 Global ResNet50에 넣습니다. 14개 질환은 한 영상에 함께 있을 수 있으므로 Softmax 대신 Sigmoid를 씁니다.
2. **영역 선택.** 원본을 512×512, stride 256인 3×3 겹침 격자로 나눕니다. 원점 \((r, c)\)는 \(\{0, 256, 512\} \times \{0, 256, 512\}\)입니다. 중앙 칸을 고른다는 것은 병변이 중앙에 있다는 뜻이 아닙니다. 행동 공간은 10개입니다.

\[
A = \{\mathrm{ROI}_1, \ldots, \mathrm{ROI}_9, \mathrm{STOP}\}
\]

같은 ROI는 다시 고르지 못하고, 확대는 최대 3회입니다.

3. **선택 영역만 고해상도 분석.** 선택한 512×512를 224×224로 줄여 Local ResNet50에 통과시킵니다. 전역 입력과 픽셀 수는 같지만, 해당 영역의 원본 픽셀을 더 많이 보존합니다.
4. **특징 결합 후 진단 갱신.**

\[
h_t = \mathrm{GRU}(h_{t-1}, f_{\mathrm{ROI}_t}), \qquad
p_t = \sigma\left(W[f_g; h_t] + b\right)
\]

\(f_g \in \mathbb{R}^{2048}\)는 전역 특징, \(h_t\)는 지금까지 확대한 영역의 누적 특징, \(p_t \in (0,1)^{14}\)는 질환 확률입니다. DDQN은 갱신된 상태만 보고 다음 ROI 또는 STOP을 고릅니다. 아직 확대하지 않은 ROI의 고해상도 특징은 정책 입력에 넣지 않습니다. 9개 영역을 미리 인코딩하면 연산 절감 주장이 성립하지 않습니다.

초기 보고는 **Nodule, Mass, Pneumothorax**에 집중합니다. 이 세 질환이 NIH에서 가장 희귀해서가 아닙니다. 작은 결절, 국소 종괴, 흉막선처럼 영역 확대의 이득을 검증하기 쉬운 병변이기 때문입니다. 헤드는 14개 라벨을 계속 예측합니다.

## 환경

한 에피소드는 X-ray 한 장입니다. STOP을 고르거나 ROI를 3개 분석하면 끝납니다.

| MDP 요소 | 설계 |
| --- | --- |
| 상태 \(s_t\) | \(f_g\), 현재 확률 \(p_{t-1}\), GRU 상태 \(h_{t-1}\), 9차원 방문 이력, 남은 확대 횟수 |
| 행동 \(a_t\) | ROI 9개 중 하나, 또는 STOP |
| 전이 | 선택 ROI만 Local Encoder 통과 → GRU 갱신 → \(p_t\) 갱신 |
| 보상 \(r_t\) | 클래스 가중 분류 손실의 감소량에서 분석 비용을 뺌 |
| 종료 | STOP, 또는 확대 3회 |

확대 행동의 보상은 다음과 같습니다.

\[
r_t = \mathcal{L}_{t-1} - \mathcal{L}_t - \lambda C_t
\]

\(\mathcal{L}\)는 클래스 가중 binary cross-entropy입니다. 가중치는 훈련 집합의 질환별 양성·음성 수로 계산하고 `class_weight_clip`으로 상한과 하한을 둡니다. \(C_t\)는 확대 1회의 비용이고, 출발값은 1입니다. \(\lambda\)는 `compute_penalty`이며 출발값은 0.02입니다. 손실 크기와 비용의 단위가 맞도록 1차 분류기의 손실 분포를 본 뒤 다시 정합니다.

STOP의 보상은 0입니다. 할인율 \(\gamma = 1\)이면 에피소드 총보상은 다음으로 정리됩니다.

\[
R = \mathcal{L}_0 - \mathcal{L}_T - \lambda \sum_{t=1}^{T} C_t
\]

학습 환경만 정답 라벨로 \(\mathcal{L}\)을 계산합니다. 추론 시 DDQN에는 라벨을 넣지 않습니다. 초기 정책 학습에서는 CNN, GRU, 분류 헤드를 고정합니다.

Q-value는 질환 클래스의 가치가 아니라, 다음에 분석할 영역 또는 종료의 가치입니다. 갱신은 Double DQN입니다. Main network가 다음 행동을 고르고 Target network가 그 가치를 평가합니다. 이미 방문한 ROI와, 예산이 0일 때의 확대 행동은 마스크로 제거합니다.

## 네트워크

| 모듈 | 역할 |
| --- | --- |
| Global Encoder | ImageNet 사전학습 ResNet50. 224×224 전역 영상 → 2048차원 \(f_g\) |
| Local Encoder | 공유하거나 따로 둔 ResNet50. 선택된 512 crop을 224×224로 줄인 입력 → \(f_{\mathrm{ROI}}\) |
| GRU Fusion | 지역 특징을 순서대로 누적. 은닉 차원 출발값은 512 |
| Classification Head | \([f_g; h_t]\) → 14차원 Sigmoid |
| DDQN Q-Network | 상태 벡터 → 행동 10개의 Q-value. 영상 CNN이 아닙니다 |

Global Encoder의 첫 층은 3채널 ImageNet 가중치를 유지합니다. 단일채널 X-ray는 채널 축으로 3번 반복해 넣습니다.

## 학습 순서

CNN과 DDQN을 한 번에 학습하지 않습니다.

| 단계 | 학습 | 고정 |
| --- | --- | --- |
| Stage 1 | 전역 ResNet50 14-라벨 분류 | 없음 |
| Stage 2 | 임의 또는 균형 샘플링한 1–3개 ROI로 Local Encoder, GRU, Head 학습 | Global Encoder |
| Stage 3 | 비용 인식 DDQN | 모든 분류 네트워크 |
| Stage 4 | 선택적 미세조정 | 실험별로 명시 |

Stage 2에서 여러 ROI 조합으로 분류기를 먼저 안정시켜야, Stage 3의 보상이 정책이 아닌 분류기 변동을 따라가지 않습니다. Stage 3은 epsilon-greedy, replay buffer, target network, action mask를 사용합니다.

## 데이터와 평가

NIH ChestX-ray14는 112,120장, 30,805명, 14개 질환 라벨입니다. 880장에 984개 bounding box가 있습니다. 분할은 환자 단위입니다. 공식 `test_list`는 최종 평가 전까지 열지 않습니다. 공식 train/val 목록 안에서만 validation 환자를 분리합니다.

라벨만으로 정책과 분류기를 학습할 수 있습니다. Bounding box는 에이전트가 병변과 겹치는 ROI를 골랐는지 보는 **ROI Hit Rate**에만 씁니다. 위치 주석을 학습에 쓰는 실험은, 그 주석의 환자 분할이 평가 환자와 겹치지 않을 때만 수행합니다.

비교는 같은 ROI 개수 또는 같은 연산 예산에서 합니다. 예산이 다르면 성능 차이가 정책 때문인지 추가 CNN 연산 때문인지 구분할 수 없습니다.

| ID | 방법 | 역할 |
| --- | --- | --- |
| B1 | ResNet50, 입력 224 | 저해상도 전역 분류 |
| B2 | ResNet50, 입력 512 | 전역 고해상도의 상한에 가까운 참고 |
| B3 | ResNet50 + Focal Loss | 클래스 불균형 보정 |
| B4 | Random ROI, 동일 확대 횟수 | 무작위 확대 |
| B5 | Grad-CAM Top-K ROI | 주의지도 기반 확대 |
| B6 | Patch fusion, 고정 K | 고정 개수 영역 분석 |
| Ours | RareZoom-RL DDQN | 비용 인식 적응 선택 |

지표는 Macro-AUPRC, 초점 세 질환의 고정 특이도 Tail Recall, 평균 확대 횟수, boxed 영상에서의 ROI Hit Rate입니다. AUROC, 클래스별 F1, GPU 시간, MACs/FLOPs를 함께 기록합니다. 비교는 환자 단위 bootstrap 신뢰구간과 여러 시드로 합니다.

## 가설

### 메인 가설: 비용 인지 적응형 해상도

> 고해상도로 다시 볼 가치는 영상마다 다르고, 224 관찰 결과로 그 가치를 예측할 수 있다. 따라서 모든 영상을 224로 먼저 판독하고, "클래스 가중 분류 손실의 감소"를 보상으로 학습한 가치가 높은 일부 영상만 512로 다시 판독하면, 512 전체 판독보다 적은 연산으로 512 이상의 Macro-AUPRC를 내고, 같은 비율의 무작위 재판독보다 높다.

행동은 영상마다 {STOP, ZOOM(512 재판독)} 두 개입니다. ZOOM의 보상은 \(\mathcal{L}_{224} - \mathcal{L}_{512} - \lambda C\)이고, 비용 \(C\)는 512 한 번의 FLOPs입니다. 학습된 가치 \(Q(s,\mathrm{ZOOM}) - Q(s,\mathrm{STOP})\)의 순위로 재판독할 영상을 고르고, \(\lambda\)는 재판독 비율과 일대일로 대응합니다. 상태 \(s\)는 보정된 224 확률의 logit과 클래스별 엔트로피입니다.

결정이 한 번이라 Q의 목표값은 즉시 보상과 같고, 학습은 Q-네트워크 회귀가 됩니다. Double DQN의 target network가 실제로 필요해지는 것은 224→384→512나 512 이후 ROI 확대처럼 결정이 여러 단계로 이어질 때입니다.

### 왜 학습된 가치(DDQN 계열)인가

가설의 각 조건이 DDQN의 구성 요소 하나씩에 대응합니다.

| 문제의 성질 | 그래서 필요한 것 | DDQN의 해당 요소 |
| --- | --- | --- |
| 더 볼지, 어디를 볼지, 언제 멈출지는 앞선 관찰에 따라 달라진다 | 멈춤을 포함한 순차 의사결정 | MDP와 \(Q(s, a)\). \(\gamma = 1\)이면 총보상이 \(\mathcal{L}_0 - \mathcal{L}_T - \lambda \sum C_t\)이므로, Q를 최대화하는 것이 곧 비용 대비 진단 개선을 최대화하는 것이다 |
| 정답 라벨은 학습 때만 있고, 추론 때는 추가 분석의 이득을 계산할 수 없다 | 상태만 보고 이득을 예측하는 함수 | 가치함수 학습. 불확실성 순위나 Grad-CAM은 이득을 예측하지 않고 대리 지표를 쓴다 |
| 행동이 ROI 9개와 STOP, 또는 해상도 단계처럼 작은 이산 집합이고, 방문한 행동은 금지된다 | 이산 행동과 마스크 | Q-learning과 action mask |
| CNN 한 번 통과가 비싸서 전이를 버리기 아깝다 | 표본 효율 | off-policy replay buffer |
| 희귀 질환의 보상은 드물고 잡음이 크다. 최댓값 연산의 과대추정은 불필요한 확대로 이어진다 | 과대추정 억제 | Double DQN. Main network가 고르고 Target network가 평가한다 |
| 연산을 줄여야 한다 | 멈춤의 기준 | 비용 항 \(\lambda C_t\). \(\lambda = 0\) ablation으로 확인한다 |

### 반증 가능한 예측

| ID | 예측 | 역할 |
| --- | --- | --- |
| P1 | 영상마다 512 재판독의 이득이 다르다. 라벨을 아는 오라클 순위의 재판독이 224와 512보다 크게 높다 | 전제. 거짓이면 어떤 정책도 이길 수 없다 |
| P2 | 검증으로 고른 비율에서 적응형 판독이 512 전체보다 연산이 적고, Macro-AUPRC가 512 이상이며, 같은 비율의 무작위 재판독보다 높다 | 핵심 |
| P3 | 같은 영상을 512 대신 다른 시드의 224 모델로 재판독한 것보다 높다 | 이득이 해상도 덕분인지, 모델을 하나 더 쓴 앙상블 덕분인지 구분 |
| P4 | 학습된 가치가 불확실성 순위(엔트로피 합)보다 높다 | 학습된 가치를 쓰는 이유 |

### 실험 절차

1. 224와 512 ResNet50을 같은 설정(배치 32, AdamW 1e-4, patience 3)으로 학습합니다. 224는 시드 42, 43, 44, 512는 시드 42, 44입니다. 에폭은 검증 Macro-AUPRC로 고릅니다.
2. 검증 환자를 해시로 반씩 나눕니다. A 절반으로 클래스별 Platt 보정과 가치 네트워크를 학습하고, B 절반으로 재판독 비율을 고릅니다.
3. 비율 규칙은 두 개입니다. 사전에 정한 규칙은 "B에서 512와 0.005 이내가 되는 가장 작은 비율"입니다. 시드 42에서 224와 512의 검증 성능이 비슷해 이 규칙이 0%를 골랐기 때문에, "B에서 Macro-AUPRC가 가장 높은 비율"을 추가했습니다. 추가 규칙은 시드 42의 테스트 곡선을 본 뒤 정했으므로, 시드 42는 탐색 결과로만 쓰고 테스트를 보지 않은 시드 44 쌍을 확인 실험으로 씁니다.
4. 공식 테스트는 고른 비율에서 한 번만 채점합니다. 차이는 환자 단위 1,000번 복원추출의 95% 구간으로 봅니다. 0.5 정확도는 쓰지 않습니다.

코드는 `scripts/rarezoom/cascade.py`이고, 결과는 `checkpoints/rarezoom/v4/cascade_report.json`에 저장됩니다.

### 결과

**확인 실험(시드 44)에서 P2, P3, P4가 성립하지 않았습니다.** 검증 B 절반에서는 두 규칙 모두 재판독 비율 0%(224만 사용)를 골랐습니다. 테스트 곡선에서도 학습된 가치는 같은 비율의 무작위 재판독과 비슷하고, 512 전체보다 낮습니다.

| 시드 44, 공식 테스트 Macro-AUPRC | 10% | 30% | 50% | 70% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 불확실성 순위 | 0.2502 | 0.2541 | 0.2549 | 0.2595 | 0.2597 |
| 학습된 가치 | 0.2490 | 0.2529 | 0.2563 | 0.2568 | 0.2597 |
| 무작위 | 0.2502 | 0.2523 | 0.2538 | 0.2558 | 0.2597 |
| 오라클(라벨 사용) | 0.3069 | 0.3343 | 0.3412 | 0.3313 | 0.2597 |

비율 0%는 224 단독(0.2526)입니다. 연산량은 224가 이미지당 8.2 GFLOPs, 512가 42.7 GFLOPs이고, 비율 \(f\)의 적응형 판독은 512 대비 \(0.19 + f\)입니다.

탐색 결과(시드 42)에서는 일부 재판독이 224와 512보다 모두 높았습니다. 불확실성 순위 10%가 0.2623으로, 224보다 +0.0029(95% 구간 0.0010–0.0049), 같은 비율의 무작위보다 +0.0036(0.0013–0.0057) 높았습니다. 하지만 같은 영상을 다른 시드의 224 모델로 재판독한 경우도 0.2620이었습니다(차이 +0.0003, −0.0015–0.0022). 학습된 가치(비율 90%)도 두 번째 224 모델 대조군과 차이가 −0.0010(−0.0081–0.0056)입니다. 따라서 시드 42의 이득은 해상도가 아니라 모델 두 개를 섞은 앙상블 효과입니다.

P1은 형식상 성립합니다. 라벨을 아는 오라클은 30–50% 재판독에서 +0.08–0.10을 얻습니다. 다만 오라클은 정답으로 영상별 손실을 비교해 모델을 고르므로, 이 여유분에는 라벨 정보가 섞여 있습니다. 라벨 없이 224 출력만으로는 그 여유분을 찾지 못했습니다.

| 예측 | 판정 |
| --- | --- |
| P1 영상별 재판독 이득의 차이 | 오라클 기준 성립. 라벨 정보가 섞인 상한 |
| P2 적은 연산으로 512 이상, 무작위보다 높음 | 기각. 검증이 0%를 골랐고, 테스트에서도 무작위와 비슷함 |
| P3 두 번째 224 모델보다 높음 | 기각. 시드 42의 이득은 앙상블 효과 |
| P4 학습된 가치가 불확실성 순위보다 높음 | 기각. 시드 44에서 더 낮거나 비슷함 |

### 이전 실험

**ROI 영역 선택에서는 재분석 이득이 작아서 DDQN이 무작위와 같았습니다.** 원본 1024에서 자른 512 ROI를 3×3 격자에서 고르는 설정입니다. 검증 Macro-AUPRC는 다음과 같습니다(`checkpoints/rarezoom/selector_val.json`, `checkpoints/rarezoom/v2/report.json`).

| 방법 | 검증 Macro-AUPRC | 확대 없음 대비 |
| --- | ---: | ---: |
| 확대 없음 | 0.2060 | — |
| 라벨 없는 규칙 중 최고 (공간 주의 Top-3) | 0.2076 | +0.0016 |
| 라벨을 쓰는 오라클 Top-3 | 0.2209 | +0.0149 |

라벨을 아는 오라클조차 +0.015에 그쳐, 정책이 배울 신호가 작았습니다. 첫 DDQN은 평균 0.05회만 확대해 확대 없음과 같았습니다. 비용을 보정한 v2 DDQN은 평균 2.0회를 확대했지만 0.2013으로, 같은 횟수의 무작위(0.2010)와 같고 확대 없음(0.2059)보다 낮았습니다.

**전역 해상도 224 대 512 비교는 배치 크기가 섞여 있었습니다.** 224를 배치 256, 512를 배치 32로 학습한 단일 시드 결과에서는, 공식 테스트에서 512가 224보다 아래만큼 높았습니다.

| 지표 | 차이 | 95% 구간 |
| --- | ---: | --- |
| Macro-AUPRC | +0.028 | 0.021–0.037 |
| Nodule | +0.060 | 0.042–0.078 |
| Mass | +0.044 | 0.026–0.062 |
| Pneumothorax | +0.032 | 0.012–0.053 |

숫자는 `checkpoints/rarezoom/v3/bootstrap_ci.json`에 있습니다. 224도 배치 32로 맞추면 시드 42의 테스트 Macro-AUPRC가 0.2321에서 0.2594로 올라 512(0.2607)와 거의 같아집니다. 배치를 맞춘 공식 테스트의 512 − 224 차이는 다음과 같습니다(`checkpoints/rarezoom/v4/resolution_seeds_report.json`).

| 지표 | 시드 42 | 시드 44 |
| --- | --- | --- |
| Macro-AUPRC | +0.0014 (−0.009–0.012) | +0.0070 (−0.002–0.015) |
| Nodule | +0.048 (0.029–0.066) | +0.001 (−0.015–0.019) |
| Mass | +0.030 (0.014–0.048) | +0.010 (−0.011–0.030) |
| Pneumothorax | +0.033 (0.010–0.054) | +0.019 (−0.001–0.042) |

Macro-AUPRC 차이는 두 시드 모두 구간이 0을 포함합니다. 초점 질환의 이득은 시드 42에서만 유의합니다. 그래서 "512가 낫다"는 가설로 쓰지 않습니다. 배치 64 추론 시간은 이미지당 224가 0.14 ms, 384가 0.46 ms, 512가 0.84 ms입니다.

## 저장소 구성

코드는 연구 트랙별로 나뉩니다. 명령은 저장소 루트에서 실행합니다.

```
configs/          bicdrl.yaml, multiclass.yaml, rarezoom.yaml, segmentation.yaml
src/bicdrl/       이진 분류 DDQN
src/classification/   15-way 분류, BBox fusion, 영역 선택 프로토타입
src/segmentation/     결절 segmentation
scripts/bicdrl/
scripts/classification/
scripts/segmentation/
tests/test_segmentation.py
data/             NIH 라벨 CSV, 공식 split, 다운로드 스크립트. 1024 원본(data/raw/)은 git 제외
```

| 코드 | 하는 일 |
| --- | --- |
| `src/bicdrl/`, `configs/bicdrl.yaml`, `scripts/bicdrl/test_data_pipeline.py` | BicDRL. 행동이 질환 클래스이고, 정답이면 \(+\gamma_k\), 오답이면 \(-\gamma_k\)입니다. 에피소드는 한 스텝입니다. |
| `src/classification/`, `scripts/classification/train_multiclass.py` | 15-way 주 헤드와 14-라벨 보조 헤드. Stage 1의 전역 분류기와는 출력 정의가 다릅니다. |
| `scripts/classification/train_ddqn.py`, `evaluate_ddqn.py`, `src/classification/region_ddqn.py` | 영역 DDQN 프로토타입. 224 텐서에서 비중첩 3×3을 모두 미리 추론하고, 로짓을 최댓값으로 합친 뒤, 종료 시점의 정답 여부로 보상합니다. RareZoom-RL 결과가 아닙니다. |
| `scripts/classification/train_fusion.py`, `src/classification/fusion.py` | 전역 특징과 BBox 마스크 ROI를 이어 붙입니다. 512 격자 crop과 GRU가 아닙니다. |
| `SEGMENTATION.md`, `scripts/segmentation/train_seg.py` | 별도 결절 segmentation 트랙입니다. |

RareZoom-RL 구현은 위 트랙을 덮어쓰지 않고 `src/rarezoom/`과 `scripts/rarezoom/`에 둡니다.

| 파일 | 역할 |
| --- | --- |
| `src/rarezoom/data.py` | 14개 다중 라벨, 환자 단위 검증 분할(시드 42), 공식 test 분리, 누수 검사, `pos_weight` |
| `src/rarezoom/roi.py` | 1024 원본 로드, 3×3 512 ROI 좌표, 전역 view와 원본 crop 텐서 |
| `src/rarezoom/model.py` | 전역 ResNet50, GRU 융합 모델, 클래스 가중 BCE, Q-network |
| `src/rarezoom/rollout.py` | ROI 정책 실행(STOP, 무작위, 탐욕, DDQN)과 같은 예산 비교 |
| `src/rarezoom/metrics.py` | AUPRC, AUROC, Macro 요약 |
| `scripts/rarezoom/train_classifier.py` | 전역 분류기 학습. `--image-size`, `--batch-size`, `--seed`(데이터 분할은 고정), `--patience` |
| `scripts/rarezoom/run_resolution_seeds.sh` | 224와 512를 배치 32, 여러 시드로 학습 |
| `scripts/rarezoom/resolution_study.py` | 검증으로 고른 체크포인트의 val/test 점수, 시드 평균과 범위, 환자 단위 bootstrap, FLOPs와 추론 시간 |
| `scripts/rarezoom/cascade.py` | 메인 실험. 224 우선 판독 후 일부만 512 재판독. 보정, 가치 학습, 비율 선택, 대조군 |
| `scripts/rarezoom/train_fusion.py` | ROI 실험 Stage 2. ROI 1–3개로 GRU 융합 학습 |
| `scripts/rarezoom/train.py` | ROI 실험 Stage 3. 비용 인식 DDQN |
| `scripts/rarezoom/select_baselines.py`, `evaluate_v2.py` | ROI 실험의 라벨 없는 선택 규칙, 오라클, DDQN 비교 |

`scripts/classification/prepare_multiclass_manifest.py`는 `images-224`를 읽습니다. RareZoom-RL의 확대 주장을 지키려면 학습과 평가에 1024 원본 경로가 있어야 합니다.

## 재현

원본 이미지는 저장소에 없습니다. `data/download_nih_images.py`로 NIH 1024 원본을 `data/raw/images/`에 받습니다. 체크포인트와 결과 JSON은 `checkpoints/`에 저장되고 git에서 제외됩니다.

```bash
pip install -r requirements.txt
python data/download_nih_images.py
bash scripts/rarezoom/run_resolution_seeds.sh   # 224·512 학습, resolution_study, cascade
```

`run_resolution_seeds.sh`는 512 시드 42로 `checkpoints/rarezoom/v3/global512`를 재사용합니다. 없으면 다음으로 먼저 학습합니다.

```bash
python scripts/rarezoom/train_classifier.py --image-size 512 --batch-size 32 --patience 3 \
  --seed 42 --save-dir checkpoints/rarezoom/v4/s512_seed42
```

RTX PRO 6000 한 장 기준으로 224 학습은 에폭당 약 2–3분, 512는 약 4분입니다. 1024 PNG 디코딩이 병목이라 여러 학습을 GPU 하나에서 함께 돌립니다.

## 한계

- 512 시드가 2개입니다. 224는 3개입니다.
- 외부 데이터(CheXpert, MIMIC-CXR)로 검증하지 않았습니다. 두 데이터 모두 사용 승인이 필요합니다.
- 적응형 해상도의 두 번째 비율 규칙은 시드 42의 테스트 곡선을 본 뒤 정했습니다. 그래서 시드 42는 탐색 결과로만 쓰고 시드 44로 확인했습니다.
- 오라클 상한은 라벨로 영상별 모델을 고르므로, 라벨 없는 정책이 도달할 수 있는 값보다 큽니다.
- 384 입력은 FLOPs와 추론 시간만 쟀고 학습하지 않았습니다.

## 참고

- Wang et al., ChestX-ray8/ChestX-ray14, CVPR 2017. NIH Clinical Center release.
- PatchDrop, CVPR 2020. 강화학습 기반 선택적 고해상도 분석.
- Ypsilantis and Montana, 2017. 흉부 X-ray의 순차적 시각 주의.
- MedicalPatchNet, Scientific Reports. 패치 기반 흉부 X-ray 분류와 위치 평가.
- CXR-LT. 장꼬리 확장 후보. 첫 실험의 비교가 끝난 뒤의 범위입니다.
