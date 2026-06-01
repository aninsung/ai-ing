import os
import math
import random
import numpy as np
import pandas as pd
from monai.data import Dataset, DataLoader
from .transforms import get_transforms

def extract_label(finding):
    # Nodule 이진 분류 ('Nodule' 포함 병변 -> 1, 그 외 모든 경우(정상 및 타질병) -> 0)
    if pd.isna(finding):
        return 0
    elif 'Nodule' in str(finding):
        return 1
    return 0

def load_and_filter_data(csv_path, image_dir, epsilon=1e-5):
    """
    데이터셋 필터링 및 적응형 가중치(gamma_k) 사전 계산 (Phase 1)
    클래스 불균형 완화를 위해 소수 클래스(1)를 오버샘플링하여 다수 클래스(0)와 동일한 비율로 맞춥니다.
    """
    data_dicts = []
    class_counts = {0: 0, 1: 0} # Binary example
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Nodule이나 No Finding 단일 병변만 남기도록 필터링
    df['label'] = df['Finding Labels'].apply(extract_label)
    
    # NIH 데이터셋은 images_001 ~ images_012 폴더에 나뉘어 저장되어 있을 수 있습니다.
    # 모든 하위 폴더를 스캔하여 파일명 -> 전체 경로 맵을 생성합니다.
    image_path_map = {}
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith('.png'):
                image_path_map[file] = os.path.join(root, file)

    for _, row in df.iterrows():
        lbl = row['label']
        if lbl == -1:
            continue # 설정한 타겟 질환이 아닌 경우 스킵
        
        img_filename = row['Image Index']
        
        # 실제 파일이 존재하는지 맵에서 확인
        if img_filename not in image_path_map:
            continue
        
        img_path = image_path_map[img_filename]
        
        data_dicts.append({"image": img_path, "label": lbl})
        class_counts[lbl] += 1
            
    # 클래스 불균형 완화: 소수 클래스(1)를 다수 클래스(0) 크기만큼 무작위 복제(오버샘플링)
    max_count = max(class_counts.values())
    if class_counts[1] < max_count and class_counts[1] > 0:
        minority_samples = [d for d in data_dicts if d['label'] == 1]
        extra_needed = max_count - class_counts[1]
        for _ in range(extra_needed):
            sample = random.choice(minority_samples)
            data_dicts.append(sample.copy())
        class_counts[1] = max_count

    # Calculate adaptive weights (gamma_k = 1 / ln(c_k + epsilon))
    class_weights = {}
    for cls_idx, count in class_counts.items():
        if count > 0:
            class_weights[cls_idx] = 1.0 / math.log(count + epsilon)
        else:
            class_weights[cls_idx] = 1.0
            
    print(f"Data Loaded: {len(data_dicts)} files (after oversampling). Class counts: {class_counts}. Weights: {class_weights}")
    return data_dicts, class_weights

def get_dataloader(data_dicts, batch_size=32, image_size=224, phase='train'):
    transforms = get_transforms(image_size, phase=phase)
    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(phase=='train'), num_workers=0)
    return loader
