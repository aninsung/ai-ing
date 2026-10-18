"""Train a linear primary head on frozen pretrained features and report validation accuracy.

The decision offset is chosen on validation only. The test split is not read.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.multiclass_dataset import CLASSES, NIHMultiTaskDataset, read_manifest
from src.multiclass_model import MultiTaskResNet50


def single_label(rows):
    return [row for row in rows if row["primary_class_id"]]


def encode(model, rows, root, batch_size, device):
    loader = DataLoader(NIHMultiTaskDataset(rows, root, image_size=224), batch_size=batch_size, shuffle=False)
    features, labels = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            features.append(model.encode(batch["image"].to(device)).cpu())
            labels.append(batch["primary"])
    return torch.cat(features), torch.cat(labels)


def accuracy(logits, labels, log_prior, tau):
    adjusted = logits + tau * log_prior
    return float((adjusted.argmax(1) == labels).float().mean())


def main():
    root = Path("data/archive")
    manifest = "data/nih_multiclass_manifest.csv"
    train_rows = single_label(read_manifest(manifest, "train"))
    val_rows = single_label(read_manifest(manifest, "val"))
    random.Random(42).shuffle(train_rows)
    train_rows = train_rows[:8000]
    device = torch.device("cpu")
    torch.set_num_threads(4)
    model = MultiTaskResNet50(15, 14, pretrained=True).to(device)
    print(json.dumps({"train_sample": len(train_rows), "val_images": len(val_rows)}), flush=True)
    train_x, train_y = encode(model, train_rows, root, 32, device)
    val_x, val_y = encode(model, val_rows, root, 32, device)
    counts = torch.bincount(train_y, minlength=15).float()
    log_prior = (counts / counts.sum()).clamp_min(1e-8).log()
    head = nn.Linear(train_x.shape[1], 15)
    with torch.no_grad():
        head.bias.copy_(log_prior)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    for epoch in range(25):
        head.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            logits = head(train_x[batch])
            loss = nn.functional.cross_entropy(logits, train_y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            head.eval()
            with torch.no_grad():
                train_logits = head(train_x)
                val_logits = head(val_x)
            scores = {tau: accuracy(val_logits, val_y, log_prior, tau) for tau in (0.0, 0.5, 1.0)}
            tau = max(scores, key=scores.get)
            print(json.dumps({
                "epoch": epoch + 1,
                "train_loss": total / len(train_x),
                "train_accuracy": accuracy(train_logits, train_y, log_prior, 0.0),
                "val_accuracy": scores[tau],
                "val_accuracy_by_tau": scores,
                "tau": tau,
                "majority_accuracy": float((val_y == 0).float().mean()),
            }), flush=True)
    majority = float((val_y == 0).float().mean())
    print(json.dumps({"done": True, "classes": CLASSES, "majority_val_accuracy": majority}), flush=True)


if __name__ == "__main__":
    main()
