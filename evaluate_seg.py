"""Evaluate a specific checkpoint on reviewed, held-out masks."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.seg_manifest import load_manifest, supervised_rows
from src.seg_dataset import SegDataset
from src.seg_network import build_model
from src.seg_engine import evaluate_model

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New directory for metrics and original-size predictions')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to prevent mixing evaluation runs')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    manifest = Path(config['data']['manifest'])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint['manifest_sha256']:
        raise ValueError('Manifest changed after training; use the original frozen manifest')
    rows = supervised_rows(load_manifest(manifest), args.split)
    dataset = SegDataset(rows, config['data']['image_size'])
    dataset.validate_all()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(checkpoint['model'])
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)
    metrics, scores = evaluate_model(model, loader, device, checkpoint['threshold'], output)
    metrics.update(split=args.split, threshold=checkpoint['threshold'],
                   checkpoint=str(Path(args.checkpoint).resolve()),
                   epoch=checkpoint['epoch']+1, overlay_panels='original | ground truth (green) | prediction (red)')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (output / 'per_image.json').write_text(json.dumps(scores, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
