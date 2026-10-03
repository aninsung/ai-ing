"""Train the UNet++ lesion segmenter from an approved segmentation manifest."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.seg_manifest import load_manifest, supervised_rows
from src.seg_dataset import SegDataset
from src.seg_network import build_model


def dice_loss(logits, target, eps=1e-6):
    prob = logits.sigmoid()
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--image-size', type=int, default=448)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', default='checkpoints/segmentation_unetpp.pth')
    args = p.parse_args()
    rows = load_manifest(args.manifest)
    train_rows = supervised_rows(rows, 'train', allow_pseudo=True)
    val_rows = supervised_rows(rows, 'val')
    train = DataLoader(SegDataset(train_rows, args.image_size, augment=True), args.batch_size, shuffle=True)
    val = DataLoader(SegDataset(val_rows, args.image_size), args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for batch in train:
            x, y = batch['image'].to(device), batch['mask'].to(device)
            logits = model(x)
            loss = .7 * dice_loss(logits, y) + .3 * torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval(); score_sum = n = 0
        with torch.no_grad():
            for batch in val:
                pred, target = model(batch['image'].to(device)).sigmoid() > .5, batch['mask'].to(device) > .5
                inter = (pred & target).sum((1,2,3)).float(); denom = pred.sum((1,2,3)) + target.sum((1,2,3))
                score_sum += ((2*inter+1e-6)/(denom+1e-6)).sum().item(); n += len(pred)
        dice = score_sum / max(n, 1)
        print({'epoch': epoch + 1, 'val_dice': dice}, flush=True)
        if dice > best:
            best = dice
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'val_dice': dice}, args.output)


if __name__ == '__main__':
    main()
