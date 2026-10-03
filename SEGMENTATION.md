# 결절 segmentation baseline

기존 DDQN 분류 코드는 유지하고 별도 supervised segmentation 파이프라인을 추가했습니다.
픽셀 마스크 없는 이미지 라벨만으로는 이 baseline을 학습할 수 없습니다.
박스/pseudo-label 생성과 전문가 라벨 작성은 자동으로 수행하지 않습니다.

## 설치

프로젝트 Python 환경에서 `pip install -r requirements-segmentation.txt`를 실행하세요.
Python 3.10 이상을 권장합니다. GPU용 PyTorch 설치는 사용 환경에 맞게 선택하세요.

## 데이터 준비

```
python prepare_seg_data.py --csv data/Data_Entry_2017.csv --images data/raw --output data/seg_manifest.csv
```

누락 파일은 `data/seg_manifest.audit.json`에 모두 기록합니다. 같은 환자는 같은 split에 들어갑니다.
해시 분할의 비율은 근사값입니다. 작은 결절 표본에서는 각 split의 양성 환자 수를 확인하고,
필요하면 환자 단위로 split을 조정한 뒤 학습 전에 고정하세요. 공식 테스트 분할을 사용한다면
manifest의 split을 공식 분할에 맞춰 수정하세요. 이 생성기는 공식 분할을 자동 적용하지 않습니다.

생성 직후 모든 행은 `unannotated,pending`입니다. 검토 후 다음처럼 수정하세요.

- 양성: `annotation_type=mask`, `review_status=approved`, `mask_path`에 실제 결절 마스크 경로.
- 검토된 음성: `annotation_type=negative`, `review_status=approved`, mask_path는 비워도 됩니다.
- 위치 라벨 없음: `unannotated,pending` 유지. 결절 라벨 0만으로 negative 승인하지 마세요.
- 불확실: `review_status=uncertain`. 학습/평가에서 제외합니다.
- 의사 라벨: train에만 `pseudo,approved`로 등록하고 config에서 allow_pseudo를 켭니다.

경로는 절대 경로 또는 manifest 파일 기준 상대 경로입니다.
마스크는 원본 영상과 동일 크기의 단일 채널 PNG이며 0/1 또는 0/255만 허용합니다.
양성의 모든 보이는 결절을 검토하세요. 박스를 채운 사각형은 실제 경계 정답이 아닙니다.
validation/test에는 검토된 실제 마스크만 사용합니다.

## 검증과 학습

```
python train_seg.py --validate-only
python train_seg.py
python train_seg.py --resume checkpoints/segmentation/latest_model.pth
```

config: `configs/segmentation.yaml`. 512x512 종횡비 유지 resize/padding, 공동 좌우반전,
MONAI 2D U-Net, Dice+BCE loss를 사용합니다. 작은 결절이 resize 후 사라지면 학습을 중단합니다.
해상도를 올리거나 패치 학습으로 확장해야 합니다. pseudo 포함 여부는 명시적으로 설정합니다.
최적 모델은 원본 좌표로 복원한 validation 양성 영상 Dice로 선택합니다.
음성 오탐도 history.jsonl에 기록하지만 복합 모델 선택 지표는 아직 구현하지 않았습니다.
검증 threshold 기본값은 0.5입니다. 변경하려면 validation 실험으로 결정하고 최종 학습 전에 고정하세요.
기존 실험을 덮어쓰지 않도록 비어 있지 않은 저장 폴더에서는 신규 학습을 거부합니다.
새 실험은 save_dir를 변경하세요. 재시작은 같은 config/manifest에만 허용합니다.
checkpoint는 모델/optimizer/epoch/config/threshold/manifest hash/RNG를 저장합니다.
원본 이미지 및 마스크의 내용을 수정하면 새로운 실험으로 실행하세요.

## 평가

```
python evaluate_seg.py --checkpoint checkpoints/segmentation/best_model.pth --output outputs/seg_test
```

test 또는 `--split val`에 등록된 approved 실제 마스크/음성만 평가합니다.
양성 이미지 Dice/IoU와 음성 이미지 오탐 비율/음성 픽셀 오탐 비율을 분리합니다.
빈 정답의 Dice를 1로 더해 전체 평균을 부풀리지 않습니다.
원본 크기의 확률맵(.npy), 예측 마스크, 원본/정답/예측 3패널 overlay,
metrics.json 및 per_image.json을 저장합니다. 결절 단위 FROC는 아직 구현하지 않았습니다.
테스트 성능을 보고 threshold/모델을 수정하지 마세요.

## 테스트

```
python -m unittest discover -s tests -v
```

환자 누수, unannotated/negative 구분, test pseudo 차단, 누락 감사, Dice 빈 정답 정책,
공동 변환 정렬, 잘못된 마스크 및 forward/backward를 검사합니다.
PyTorch/MONAI가 없으면 해당 의존성을 요구하는 테스트는 skip합니다.

다음 작업은 실제 영상 및 라벨 확보, 전문가 검토, 소량 표본 과적합 검증입니다.
실제 데이터에서의 성능이나 임상 유효성을 검증한 구현은 아닙니다.
