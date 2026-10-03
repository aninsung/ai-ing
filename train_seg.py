"""Train supervised 2D nodule segmentation from reviewed masks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
import yaml
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from src.seg_manifest import load_manifest, supervised_rows
from src.seg_dataset import SegDataset
from src.seg_network import build_model
from src.seg_engine import seed_everything, evaluate_model

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/segmentation.yaml'))
    parser.add_argument('--resume', help='Resume latest checkpoint with identical config/manifest')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    data, cfg = config['data'], config['train']
    size = int(data['image_size'])
    if size < 64 or size % 16:
        raise ValueError('image_size must be >=64 and divisible by 16')
    threshold = float(cfg['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('threshold must lie strictly between 0 and 1')
    if int(cfg['epochs']) < 1 or int(cfg['batch_size']) < 1:
        raise ValueError('epochs and batch_size must be positive')
    seed_everything(cfg['seed'])
    manifest = Path(data['manifest'])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = load_manifest(manifest)
    train_rows = supervised_rows(rows, 'train', data.get('allow_pseudo', False))
    val_rows = supervised_rows(rows, 'val')
    train_set = SegDataset(train_rows, size, augment=True)
    val_set = SegDataset(val_rows, size)
    train_set.validate_all()
    val_set.validate_all()
    if not any(row['annotation_type'] == 'mask' for row in train_rows):
        raise ValueError('Train requires at least one reviewed positive ground-truth mask')
    if not any(row['annotation_type'] == 'mask' for row in val_rows):
        raise ValueError('Validation requires positive ground-truth masks for model selection')
    print(f'Validated {len(train_set)} train / {len(val_set)} validation images')
    if args.validate_only:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg.get('num_workers', 0))
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg.get('num_workers', 0))
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=False)
    save_dir = Path(cfg['save_dir'])
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    start, best = 0, -1.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['manifest_sha256'] != manifest_hash:
            raise ValueError('Resume requires the same config and manifest; start a new run for changes')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start, best = checkpoint['epoch'] + 1, checkpoint['best_dice']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        if 'python_rng' in checkpoint:
            import random
            random.setstate(checkpoint['python_rng'])
    else:
        if save_dir.exists() and any(save_dir.iterdir()):
            raise ValueError('Save directory is non-empty. Resume explicitly or choose a new save_dir')
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    # Freeze all splits for reproducibility; evaluation rejects modified manifests.
    with (save_dir / 'manifest_snapshot.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for epoch in range(start, cfg['epochs']):
        model.train()
        loss_sum, samples = 0., 0
        for batch in train_loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            # A tiny lesion must not silently become an all-background training target.
            for i, index in enumerate(batch['index'].tolist()):
                if train_rows[index]['annotation_type'] != 'negative' and not masks[i].any():
                    raise ValueError('Positive mask disappeared during resize; increase image_size/use patches')
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), masks)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite segmentation loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            samples += len(images)
        metrics, _ = evaluate_model(model, val_loader, device, threshold)
        score = metrics['positive_dice']
        if score is None:
            raise ValueError('Validation has no positive masks')
        improved = score > best
        best = max(best, score)
        record = dict(epoch=epoch+1, train_loss=loss_sum/samples, **metrics)
        print(json.dumps(record))
        with (save_dir / 'history.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')
        import random
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'best_dice': best, 'config': config,
                      'manifest_sha256': manifest_hash, 'threshold': threshold,
                      'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        torch.save(checkpoint, save_dir / 'latest_model.pth')
        if improved:
            torch.save(checkpoint, save_dir / 'best_model.pth')


if __name__ == '__main__':
    main()
