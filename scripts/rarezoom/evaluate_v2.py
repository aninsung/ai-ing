"""Validation comparison and one official-test report. Test scores are not used to choose a model."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rarezoom.data import DISEASES, ChestDataset, load_records
from src.rarezoom.metrics import summarize
from src.rarezoom.model import GlobalClassifier, QNetwork, RareZoomNet
from src.rarezoom.roi import BOXES, RAW_SIZE
from src.rarezoom.rollout import run_matched, run_policy


def load_model(path, device, classes, hidden):
    model = RareZoomNet(classes, hidden).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()
    return model


def load_policy(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = QNetwork(checkpoint["state_dim"], 10).to(device)
    policy.load_state_dict(checkpoint["q_network"])
    policy.eval()
    return policy, checkpoint.get("penalty")


def pack(rollout, focus):
    metrics = summarize(rollout["label"], rollout["score"], DISEASES)
    report = {
        "macro_auprc": metrics["macro_auprc"],
        "macro_auroc": metrics["macro_auroc"],
        "focus_auprc": {name: metrics["auprc"][name] for name in focus},
    }
    if "zooms" in rollout:
        report["avg_zooms"] = float(rollout["zooms"].mean())
    return report


def hit_rate(rollout):
    boxes = {}
    with (ROOT / "data" / "BBox_List_2017.csv").open(encoding="utf-8", newline="") as file:
        next(file)
        for row in csv.reader(file):
            if len(row) >= 6:
                boxes.setdefault(row[0], []).append(tuple(float(value) for value in row[2:6]))
    located = {
        name: (choice, size)
        for name, choice, size in zip(rollout["names"], rollout["selected"], rollout["sizes"])
    }
    hits = total = 0
    for name, rectangles in boxes.items():
        if name not in located:
            continue
        choice, (width, height) = located[name]
        for left, top, box_w, box_h in rectangles:
            total += 1
            center_x = (left + box_w / 2) * RAW_SIZE / width
            center_y = (top + box_h / 2) * RAW_SIZE / height
            for roi in choice:
                roi_top, roi_left, roi_bottom, roi_right = BOXES[roi]
                if roi_left <= center_x < roi_right and roi_top <= center_y < roi_bottom:
                    hits += 1
                    break
    return {"boxes": total, "hit_rate": hits / total if total else None}


def eval_global(path, rows, size, device, focus):
    model = GlobalClassifier(14, pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()
    loader = DataLoader(ChestDataset(rows, size, False), batch_size=32, shuffle=False, num_workers=8)
    scores, labels = [], []
    with torch.no_grad():
        for images, target in loader:
            scores.append(torch.sigmoid(model(images.to(device, non_blocking=True))).float().cpu().numpy())
            labels.append(target.numpy())
    metrics = summarize(np.concatenate(labels), np.concatenate(scores), DISEASES)
    return {
        "macro_auprc": metrics["macro_auprc"],
        "macro_auroc": metrics["macro_auroc"],
        "focus_auprc": {name: metrics["auprc"][name] for name in focus},
        "image_size": size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v3")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/rarezoom.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focus = config["dataset"]["focus_classes"]
    hidden = int(config["model"]["gru_hidden"])
    records, _ = load_records(ROOT / "data", seed=int(config["train"]["seed"]))
    splits = {name: [record for record in records if record["split"] == name] for name in ("val", "test")}
    root = ROOT / config["train"]["save_dir"] / args.tag
    model = load_model(root / "stage2" / "best_model.pth", device, config["dataset"]["num_classes"], hidden)
    cost_policy, cost_penalty = load_policy(root / "stage3" / "ddqn.pth", device)
    free_policy, _ = load_policy(root / "stage3" / "ddqn_free.pth", device)
    report = {"selection": "validation_only", "cost_penalty": cost_penalty, "test_boxes_not_used_for_selection": True}
    for split, rows in splits.items():
        stop = run_policy(model, rows, "stop", device)
        greedy = run_policy(model, rows, "greedy", device, q_net=cost_policy)
        matched = run_matched(model, rows, greedy["zooms"].tolist(), device)
        free = run_policy(model, rows, "greedy", device, q_net=free_policy)
        entry = {
            "stop": pack(stop, focus),
            "ddqn": pack(greedy, focus),
            "matched_random": pack(matched, focus),
            "ddqn_no_cost": pack(free, focus),
        }
        entry["matched_random"]["avg_zooms"] = entry["ddqn"]["avg_zooms"]
        if split == "test":
            entry["ddqn"]["roi_hit"] = hit_rate(greedy)
            entry["ddqn_no_cost"]["roi_hit"] = hit_rate(free)
        global512 = root / "global512" / "best_model.pth"
        if global512.is_file():
            entry["global512"] = eval_global(global512, rows, 512, device, focus)
        report[split] = entry
        print(json.dumps({split: entry}), flush=True)
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(root / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
