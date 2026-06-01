# 데이터셋 전처리 코드 (5:5 비율로 300개 샘플링 - 결절/비결절)

import pandas as pd

df = pd.read_csv("data/Data_Entry_2017.csv")

# 1. 'Nodule'이 포함되지 않은 무작위 데이터 필터링 및 150개 샘플링
# pd.isna(df['Finding Labels'])를 포함하거나, 'Nodule'이 포함되지 않은 경우
df_non_nodule = df[~df['Finding Labels'].str.contains('Nodule', na=False)]
df_non_nodule_sample = df_non_nodule.sample(n=150, random_state=42)

# 2. 'Nodule' 포함 데이터 필터링 및 150개 샘플링
df_nodule = df[df['Finding Labels'].str.contains('Nodule', na=False)]
df_nodule_sample = df_nodule.sample(n=150, random_state=42)

# 3. 데이터 병합 및 셔플링
df_balanced = pd.concat([df_non_nodule_sample, df_nodule_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 저장
df_balanced.to_csv("data/sampled_300_nodule_balanced.csv", index=False)
print("Balanced dataset with 300 samples (Non-Nodule vs Nodule) created successfully.")