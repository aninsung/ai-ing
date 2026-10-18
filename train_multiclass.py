"""Train the NIH 15-class primary head with a 14-label auxiliary head."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.multiclass_dataset import read_manifest, NIHMultiTaskDataset, CLASSES, DISEASES
from src.multiclass_model import MultiTaskResNet50
from src.multiclass_losses import auxiliary_positive_weight, multitask_loss
from src.multiclass_metrics import macro_binary_auroc


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); correct = total = 0; predictions = []; targets = []; disease_probs = []; disease_targets = []
    for batch in loader:
        primary = batch["primary"].to(device)
        primary_logits, auxiliary_logits = model(batch["image"].to(device))
        disease_probs.append(auxiliary_logits.sigmoid().cpu().numpy())
        disease_targets.append(batch["auxiliary"].numpy())
        mask = primary >= 0
        if mask.any():
            pred = primary_logits[mask].argmax(1); correct += (pred == primary[mask]).sum().item(); total += mask.sum().item()
            predictions.extend(pred.cpu().tolist()); targets.extend(primary[mask].cpu().tolist())
    balanced = None
    if targets:
        recalls = []
        for cls in range(len(CLASSES)):
            actual = [i for i, value in enumerate(targets) if value == cls]
            if actual:
                recalls.append(sum(predictions[i] == cls for i in actual) / len(actual))
        balanced = float(np.mean(recalls)) if recalls else None
    macro, per_label = macro_binary_auroc(np.concatenate(disease_targets), np.concatenate(disease_probs))
    return {
        "accuracy": correct / total if total else None,
        "balanced_accuracy": balanced,
        "primary_images": total,
        "macro_auroc": macro,
        "label_auroc": {name: score for name, score in zip(DISEASES, per_label)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiclass.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epochs for a smoke or staged run")
    parser.add_argument("--resume", default=None, help="Resume from a saved checkpoint")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_all(config["train"]["seed"])
    rows = {split: read_manifest(config["data"]["manifest"], split) for split in ("train", "val", "test")}
    root = config["data"]["image_root"]
    image_size = int(config["data"].get("image_size", 224))
    datasets = {split: NIHMultiTaskDataset(rows[split], root, augment=split == "train", image_size=image_size)
                for split in ("train", "val", "test")}
    loaders = {split: DataLoader(dataset, batch_size=config["train"]["batch_size"],
                                 shuffle=split == "train", num_workers=0,
                                 pin_memory=torch.cuda.is_available())
               for split, dataset in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet50(config["data"]["num_classes"], config["data"]["multilabel_classes"], config["model"]["pretrained"]).to(device)
    counts = torch.zeros(config["data"]["num_classes"], device=device)
    for row in rows["train"]:
        if row["primary_class_id"]: counts[int(row["primary_class_id"])] += 1
    if (counts == 0).any(): raise ValueError(f"No train primary examples for classes: {[CLASSES[i] for i,v in enumerate(counts) if v == 0]}")
    weights = (1.0 / counts.clamp_min(1).log1p()); weights = weights / weights.mean()
    lo, hi = config["train"]["class_weight_clip"]; weights = weights.clamp(lo, hi)
    pos_weight = auxiliary_positive_weight(rows["train"], DISEASES).to(device)
    auxiliary_loss_name = config["train"].get("auxiliary_loss", "asymmetric_bce")
    head_lr = float(config["train"].get("head_learning_rate", config["train"]["learning_rate"]))
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["train"]["learning_rate"]},
        {"params": list(model.primary_head.parameters()) + list(model.auxiliary_head.parameters()), "lr": head_lr},
    ])
    use_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    save_dir = Path(config["train"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    epochs = args.epochs if args.epochs is not None else config["train"]["epochs"]
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best = float(checkpoint.get("best_score", -1.0))
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    for epoch in range(start_epoch, epochs):
        model.train(); loss_sum = primary_sum = auxiliary_sum = samples = 0
        step = 0
        total_steps = len(loaders["train"])
        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True); primary = batch["primary"].to(device, non_blocking=True); auxiliary = batch["auxiliary"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                p_logits, a_logits = model(images)
                loss, primary_loss, auxiliary_loss = multitask_loss(
                    p_logits, a_logits, primary, auxiliary, weights, config["model"]["auxiliary_loss_weight"],
                    auxiliary_loss_name, pos_weight)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            n = len(images); samples += n; loss_sum += loss.item()*n; primary_sum += primary_loss.item()*n; auxiliary_sum += auxiliary_loss.item()*n
            step += 1
            if step == 1 or step % 200 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "steps": total_steps, "loss": loss_sum / samples}), flush=True)
        metrics = validate(model, loaders["val"], device); metrics.update(epoch=epoch+1, loss=loss_sum/samples, primary_loss=primary_sum/samples, auxiliary_loss=auxiliary_sum/samples)
        print(json.dumps(metrics))
        score = metrics["macro_auroc"] if metrics["macro_auroc"] is not None else -1.0
        improved = score > best
        best = max(best, score)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best, "config": config, "classes": CLASSES, "class_weights": weights.cpu(), "selection_metric": "macro_auroc"}
        torch.save(checkpoint, save_dir / "latest_model.pth")
        if improved:
            torch.save(checkpoint, save_dir / "best_model.pth")
        (save_dir / "history.jsonl").open("a", encoding="utf-8").write(json.dumps(metrics) + "\n")
    best_path = save_dir / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
        test_metrics = validate(model, loaders["test"], device)
        test_metrics["split"] = "test"
        print(json.dumps({"test_macro_auroc": test_metrics["macro_auroc"], "label_auroc": test_metrics["label_auroc"]}))
        (save_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
