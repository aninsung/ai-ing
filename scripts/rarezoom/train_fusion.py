"""Stage 2: train the local encoder, GRU, and fusion head. The global encoder stays fixed."""
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

from src.rarezoom.data import DISEASES, SequenceDataset, collate_sequences, load_records, positive_weights
from src.rarezoom.metrics import summarize
from src.rarezoom.model import RareZoomNet, focus_bce, per_sample_bce


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefix_loss(outputs, lengths, labels, pos_weight):
    terms = []
    for step, logits in enumerate(outputs):
        active = lengths >= step
        if not active.any():
            continue
        loss = per_sample_bce(logits, labels, pos_weight)
        terms.append(loss[active].mean())
    return torch.stack(terms).mean()


def fusion_objective(model, images, crops, lengths, labels, pos_weight, focus_index):
    outputs, hidden, global_features, _ = model.forward_sequence(images, crops, lengths)
    base = prefix_loss(outputs, lengths, labels, pos_weight)
    residual = per_sample_bce(
        model.residual_logits(global_features, hidden), labels, pos_weight,
    ).mean()
    before = focus_bce(outputs[0], labels, pos_weight, focus_index)
    after = focus_bce(outputs[-1], labels, pos_weight, focus_index)
    hinge = torch.relu(after - before).mean()
    return base + residual + hinge


@torch.no_grad()
def evaluate(model, loader, device, focus):
    model.eval()
    zero_scores, full_scores, targets = [], [], []
    for images, crops, lengths, labels in loader:
        images = images.to(device, non_blocking=True)
        crops = crops.to(device, non_blocking=True)
        lengths = lengths.to(device)
        outputs, _, _, _ = model.forward_sequence(images, crops, lengths)
        zero_scores.append(torch.sigmoid(outputs[0]).float().cpu().numpy())
        full_scores.append(torch.sigmoid(outputs[-1]).float().cpu().numpy())
        targets.append(labels.numpy())
    target = np.concatenate(targets)
    zero = summarize(target, np.concatenate(zero_scores), DISEASES)
    full = summarize(target, np.concatenate(full_scores), DISEASES)
    return {
        "zero_macro_auprc": zero["macro_auprc"],
        "fusion_macro_auprc": full["macro_auprc"],
        "fusion_macro_auroc": full["macro_auroc"],
        "focus_auprc": {name: full["auprc"][name] for name in focus},
        "selection_score": full["macro_auprc"],
        "fusion_gain": full["macro_auprc"] - zero["macro_auprc"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rarezoom.yaml")
    parser.add_argument("--stage1", default="checkpoints/rarezoom/stage1/best_model.pth")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--patience", type=int, default=21)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(config["train"]["seed"])
    epochs = args.epochs or int(config["train"]["stage2_epochs"])
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, summary = load_records(ROOT / "data", seed=seed)
    weights, _, _ = positive_weights(records, config["reward"]["class_weight_clip"])
    train_rows = [record for record in records if record["split"] == "train"]
    val_rows = [record for record in records if record["split"] == "val"]
    max_zooms = int(config["roi"]["max_zooms"])
    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_sequences,
        pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    train_loader = DataLoader(
        SequenceDataset(train_rows, augment=True, max_zooms=max_zooms), shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        SequenceDataset(val_rows, augment=False, fixed=True, max_zooms=max_zooms), shuffle=False, **loader_kwargs,
    )
    model = RareZoomNet(config["dataset"]["num_classes"], config["model"]["gru_hidden"]).to(device)
    stage1 = torch.load(ROOT / args.stage1, map_location="cpu", weights_only=False)
    model.load_stage1(stage1)
    model.to(device)
    pos_weight = torch.tensor(weights, device=device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4, weight_decay=1e-4,
    )
    focus_index = torch.tensor(
        [DISEASES.index(name) for name in config["dataset"]["focus_classes"]], device=device,
    )
    save_dir = ROOT / config["train"]["save_dir"] / args.tag / "stage2"
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.jsonl"
    best, stalled = -1.0, 0
    print(json.dumps({"stage": 2, "split": summary["images"], "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = seen = 0
        for step, (images, crops, lengths, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            crops = crops.to(device, non_blocking=True)
            lengths = lengths.to(device)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = fusion_objective(
                    model, images, crops, lengths, labels, pos_weight, focus_index,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            seen += len(images)
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": total / seen}), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        metrics = evaluate(model, val_loader, device, config["dataset"]["focus_classes"])
        record = {"epoch": epoch, "train_loss": total / max(seen, 1), **metrics}
        print(json.dumps(record), flush=True)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        payload = {"model": model.state_dict(), "epoch": epoch, "pos_weight": weights.tolist(), "metrics": record}
        torch.save(payload, save_dir / "last_model.pth")
        if metrics["fusion_macro_auprc"] > best:
            best = metrics["selection_score"]
            stalled = 0
            torch.save(payload, save_dir / "best_model.pth")
        else:
            stalled += 1
            if stalled >= args.patience:
                print(json.dumps({"early_stop": epoch, "best_selection_score": best}), flush=True)
                break
    print(json.dumps({"best_selection_score": best, "checkpoint": str(save_dir / "best_model.pth")}), flush=True)


if __name__ == "__main__":
    main()
