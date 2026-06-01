import torch
import numpy as np
from src.utils import get_config
from src.dataset import load_and_filter_data, get_dataloader
from src.networks import DDQNNetwork
from sklearn.metrics import classification_report, accuracy_score, f1_score

def evaluate():
    config = get_config("configs/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = config['agent']['num_classes']
    
    # 1. 테스트 데이터 준비 (실제 테스트용 분리 데이터 사용)
    data_dicts, _ = load_and_filter_data(config['data']['test_csv_path'], config['data']['image_dir'])
    test_loader = get_dataloader(data_dicts, batch_size=config['agent']['batch_size'], phase='test')
    
    # 2. 모델 로드
    import os
    import glob
    model = DDQNNetwork(num_classes, config['agent']['backbone']).to(device)
    
    checkpoint_dirs = glob.glob("checkpoints/*/")
    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoints found in 'checkpoints/' directory.")
    latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
    best_model_path = os.path.join(latest_dir, "best_model.pth")
    
    print(f"Loading model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # 3. 평가 메트릭 수집
    all_preds = []
    all_targets = []
    
    print("평가 시작...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device).float()
            labels = batch['label'].to(device)
            
            # Q-network 출력 값 중 가장 큰 값을 행동(예측)으로 선택
            q_values = model(images)
            actions = q_values.argmax(dim=1)
            
            all_preds.extend(actions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # 평가 지표 (F1-score 등) 출력
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    print("-" * 30)
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(all_targets, all_preds))
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
