# 'Nodule' 비율 유지하면서 데이터를 7:3으로 분할하는 코드

import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading original dataset...")
df = pd.read_csv("data/Data_Entry_2017.csv")

# 계층적 샘플링(Stratified split)을 위해 Nodule 여부 컬럼 임시 생성
df['has_nodule'] = df['Finding Labels'].str.contains('Nodule', na=False).astype(int)

print("Splitting dataset 7:3 with stratification...")
# 7:3 비율로 분할 (Nodule 비율 유지)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['has_nodule'])

# 임시 컬럼 제거 후 저장
train_df.drop(columns=['has_nodule']).to_csv("data/train_data.csv", index=False)
test_df.drop(columns=['has_nodule']).to_csv("data/test_data.csv", index=False)

print(f"Data successfully split.")
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
