"""Train the RareZoom stage-1 global 14-label ResNet50.

Official NIH test patients are never loaded into this loop. Validation is a
patient-disjoint 15% bucket of the official train_val list.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rarezoom.data import DISEASES, ChestDataset, load_records, positive_weights
from src.rarezoom.model import GlobalClassifier


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def average_precision(target, score):
    positive = int(target.sum())
    if positive == 0:
        return None
    order = np.argsort(-score)
    hits = np.cumsum(target[order])
    precision = hits / np.arange(1, len(target) + 1)
    return float(precision[target[order] == 1].mean())


def auroc(target, score):
    positive = int(target.sum())
    negative = len(target) - positive
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    sum_ranks = ranks[target == 1].sum()
    return float((sum_ranks - positive * (positive + 1) / 2) / (positive * negative))


def macro(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(labels.numpy())
    score = np.concatenate(scores)
    target = np.concatenate(targets)
    ap = [average_precision(target[:, i], score[:, i]) for i in range(target.shape[1])]
    auc = [auroc(target[:, i], score[:, i]) for i in range(target.shape[1])]
    return {
        "macro_auprc": macro(ap),
        "macro_auroc": macro(auc),
        "auprc": {name: value for name, value in zip(DISEASES, ap)},
        "auroc": {name: value for name, value in zip(DISEASES, auc)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rarezoom.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop each epoch after this many updates")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Training seed; the data split always uses the config seed")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    split_seed = int(config["train"]["seed"])
    seed = split_seed if args.seed is None else args.seed
    epochs = args.epochs or int(config["train"]["stage1_epochs"])
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, summary = load_records(ROOT / "data", seed=split_seed)
    weights, positive, negative = positive_weights(records, config["reward"]["class_weight_clip"])
    print(json.dumps({"split": summary, "train_positive": dict(zip(DISEASES, positive.tolist()))}, default=str), flush=True)

    train_rows = [record for record in records if record["split"] == "train"]
    val_rows = [record for record in records if record["split"] == "val"]
    size = args.image_size or int(config["dataset"]["global_size"])
    train_loader = DataLoader(
        ChestDataset(train_rows, size, augment=True), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        ChestDataset(val_rows, size, augment=False), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    model = GlobalClassifier(config["dataset"]["num_classes"], pretrained=config["model"]["pretrained"]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    save_dir = ROOT / (args.save_dir or str(Path(config["train"]["save_dir"]) / "stage1"))
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.jsonl"
    best, stalled = -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = seen = 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            seen += len(images)
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": total / seen}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "macro_auprc": metrics["macro_auprc"],
            "macro_auroc": metrics["macro_auroc"],
            "focus_auprc": {name: metrics["auprc"][name] for name in config["dataset"]["focus_classes"]},
        }
        print(json.dumps(record), flush=True)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "pos_weight": weights.tolist(),
            "diseases": DISEASES,
            "metrics": record,
            "image_size": size,
            "batch_size": args.batch_size,
            "seed": seed,
            "split_seed": split_seed,
        }
        torch.save(payload, save_dir / "last_model.pth")
        if metrics["macro_auprc"] is not None and metrics["macro_auprc"] > best:
            best = metrics["macro_auprc"]
            stalled = 0
            torch.save(payload, save_dir / "best_model.pth")
        else:
            stalled += 1
            if args.patience and stalled >= args.patience:
                print(json.dumps({"early_stop": epoch, "best_macro_auprc": best}), flush=True)
                break
    print(json.dumps({"best_macro_auprc": best, "checkpoint": str(save_dir / "best_model.pth"), "image_size": size}), flush=True)


if __name__ == "__main__":
    main()
