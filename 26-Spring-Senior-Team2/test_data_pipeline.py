import os
import torch
from src.utils import get_config, _set_seed
from src.dataset import load_and_filter_data, get_dataloader

def test_data_pipeline():
    print("🛠️ [Phase 1] Data Pipeline Validator 🛠️\n")
    _set_seed(42)
    config = get_config("configs/config.yaml")
    
    csv_path = config['data']['csv_path']
    image_dir = config['data']['image_dir']
    
    print(f"1. Checking Paths:")
    print(f"   - CSV Path: {csv_path} (Exists: {os.path.exists(csv_path)})")
    print(f"   - Image Dir: {image_dir} (Exists: {os.path.exists(image_dir)})\n")
    
    print(f"2. Loading and Filtering Data...")
    data_dicts, class_weights = load_and_filter_data(csv_path, image_dir)
    print(f"   -> Found {len(data_dicts)} valid samples.")
    print(f"   -> Class Weights (gamma_k): {class_weights}\n")
    
    if len(data_dicts) == 0:
        print("❌ Error: No data found. Please check your dataset configuration.")
        return

    print(f"3. Building DataLoader and applying MONAI Transforms...")
    # 테스트를 위해 배치 사이즈를 4로 설정
    loader = get_dataloader(data_dicts, batch_size=4, phase='train')
    
    print(f"4. Fetching a single batch...")
    try:
        batch = next(iter(loader))
        images = batch['image']
        labels = batch['label']
        
        print("\n✅ Batch successfully loaded!")
        print(f"   - Images Shape: {images.shape} (Expected: [Batch, 1, 224, 224])")
        print(f"   - Images dtype: {images.dtype} (Expected: torch.float32)")
        print(f"   - Images Min/Max: {images.min().item():.3f} ~ {images.max().item():.3f} (Expected: ~0.0 to ~1.0)")
        print(f"   - Labels Shape: {labels.shape}")
        print(f"   - Labels Data: {labels.tolist()}")
        
    except Exception as e:
        print(f"\n❌ Exception occurred during dataloading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_pipeline()
