"""Compare non-RL region selectors on the original fusion model.

The greedy selector uses validation labels, so it is an upper bound, not a deployable policy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rarezoom.data import DISEASES, load_records
from src.rarezoom.metrics import summarize
from src.rarezoom.model import GlobalClassifier, _encoder
from src.rarezoom.roi import BOXES, crop_tensor, load_raw, view_tensor

FOCUS = ["Nodule", "Mass", "Pneumothorax"]
FOCUS_INDEX = [DISEASES.index(name) for name in FOCUS]


class LegacyFusion(nn.Module):
    """Original stage-2 network: concatenation head, 224 local crops."""

    def __init__(self):
        super().__init__()
        self.global_encoder = _encoder()
        self.local_encoder = _encoder()
        self.gru = nn.GRU(2048, 512, batch_first=True)
        self.head = nn.Linear(2048 + 512, 14)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _encode(self, encoder, images):
        pixels = images.repeat(1, 3, 1, 1)
        pixels = (pixels - self.mean) / self.std
        return encoder(pixels)

    def forward_order(self, global_features, local_features, order):
        hidden = global_features.new_zeros(1, global_features.shape[0], 512)
        logits = self.head(torch.cat([global_features, hidden.squeeze(0)], dim=1))
        outputs = [logits]
        for step in range(order.shape[1]):
            chosen = local_features.gather(1, order[:, step].view(-1, 1, 1).expand(-1, 1, 2048)).squeeze(1)
            _, hidden = self.gru(chosen.unsqueeze(1), hidden)
            outputs.append(self.head(torch.cat([global_features, hidden.squeeze(0)], dim=1)))
        return outputs


def load_fusion(path, device):
    model = LegacyFusion().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    weights = torch.tensor(checkpoint["pos_weight"], device=device)
    return model, weights


def box_scores(heatmap):
    """heatmap: (B, 1024, 1024) -> (B, 9) mean inside each ROI."""
    scores = []
    for top, left, bottom, right in BOXES:
        scores.append(heatmap[:, top:bottom, left:right].mean(dim=(1, 2)))
    return torch.stack(scores, dim=1)


def grad_cam(stage1, images):
    features = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        features["map"] = output

    handle = stage1.network.layer4.register_forward_hook(hook)
    logits = stage1(images)
    score = logits[:, FOCUS_INDEX].sum()
    stage1.zero_grad(set_to_none=True)
    score.backward()
    handle.remove()
    feature_map = features["map"]
    weights = feature_map.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=1))
    cam = F.interpolate(cam.unsqueeze(1), size=1024, mode="bilinear", align_corners=False).squeeze(1)
    return cam.detach()


def per_sample_loss(logits, target, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return loss.mean(dim=1)


def pack(target, score, focus=FOCUS):
    metrics = summarize(target, score, DISEASES)
    return {
        "macro_auprc": metrics["macro_auprc"],
        "macro_auroc": metrics["macro_auroc"],
        "focus_auprc": {name: metrics["auprc"][name] for name in focus},
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, _ = load_records(ROOT / "data", seed=42)
    val_rows = [record for record in records if record["split"] == "val"]
    train_rows = [record for record in records if record["split"] == "train"]
    fusion, pos_weight = load_fusion(ROOT / "checkpoints/rarezoom/stage2/best_model.pth", device)
    stage1 = GlobalClassifier(14, pretrained=False).to(device)
    stage1.load_state_dict(torch.load(
        ROOT / "checkpoints/rarezoom/stage1/best_model.pth", map_location=device, weights_only=False,
    )["model"])
    stage1.eval()

    def layer4_energy(views):
        pixels = (views.repeat(1, 3, 1, 1) - fusion.mean) / fusion.std
        x = fusion.global_encoder.conv1(pixels)
        x = fusion.global_encoder.relu(fusion.global_encoder.bn1(x))
        x = fusion.global_encoder.maxpool(x)
        x = fusion.global_encoder.layer4(
            fusion.global_encoder.layer3(
                fusion.global_encoder.layer2(fusion.global_encoder.layer1(x))
            )
        )
        energy = torch.relu(x).mean(dim=1, keepdim=True)
        return F.interpolate(energy, size=1024, mode="bilinear", align_corners=False).squeeze(1)

    def embed(rows):
        global_features, locals_, cam_scores, act_scores, labels = [], [], [], [], []
        for start in range(0, len(rows), 16):
            chunk = rows[start:start + 16]
            raws = [load_raw(record["path"])[0] for record in chunk]
            views = torch.stack([view_tensor(image, 224) for image in raws]).to(device)
            crops = torch.stack([
                torch.stack([crop_tensor(image, roi, 224) for roi in range(9)]) for image in raws
            ]).to(device)
            with torch.no_grad():
                fg = fusion._encode(fusion.global_encoder, views)
                local = fusion._encode(
                    fusion.local_encoder, crops.reshape(-1, 1, 224, 224),
                ).view(len(chunk), 9, 2048)
                energy = layer4_energy(views)
            cam = grad_cam(stage1, views)
            global_features.append(fg.cpu())
            locals_.append(local.cpu())
            cam_scores.append(box_scores(cam).detach().cpu())
            act_scores.append(box_scores(energy).cpu())
            labels.append(torch.tensor(np.stack([record["label"] for record in chunk])))
            if (start // 16) % 20 == 0:
                print(json.dumps({"embedded": start + len(chunk), "of": len(rows)}), flush=True)
        return (torch.cat(global_features), torch.cat(locals_), torch.cat(cam_scores),
                torch.cat(act_scores), torch.cat(labels))

    print(json.dumps({"split": "val", "images": len(val_rows)}), flush=True)
    fg, local, cam, energy, labels = embed(val_rows)
    fg, local = fg.to(device), local.to(device)
    target = labels.numpy()

    def logits_for(order):
        with torch.no_grad():
            outputs = fusion.forward_order(fg, local, order.to(device))
        return torch.sigmoid(outputs[-1]).cpu().numpy(), outputs

    stop_order = torch.zeros(len(val_rows), 0, dtype=torch.long)
    # forward_order with empty order
    with torch.no_grad():
        hidden = fg.new_zeros(1, fg.shape[0], 512)
        stop_logits = fusion.head(torch.cat([fg, hidden.squeeze(0)], dim=1))
    report = {"stop": pack(target, torch.sigmoid(stop_logits).cpu().numpy())}

    def apply_ranking(scores, name):
        for k in (1, 3):
            order = torch.argsort(scores, dim=1, descending=True)[:, :k]
            probability, _ = logits_for(order)
            entry = pack(target, probability)
            entry["k"] = k
            report[f"{name}_top{k}"] = entry
            print(json.dumps({f"{name}_top{k}": entry}), flush=True)

    apply_ranking(cam, "gradcam")
    apply_ranking(energy, "spatial")

    # Oracle and supervised top-k use single-step loss reduction.
    reductions = []
    base_loss = per_sample_loss(stop_logits, labels.to(device), pos_weight)
    for roi in range(9):
        order = torch.full((len(val_rows), 1), roi, dtype=torch.long)
        _, outputs = logits_for(order)
        step_loss = per_sample_loss(outputs[-1], labels.to(device), pos_weight)
        reductions.append((base_loss - step_loss).cpu())
    reductions = torch.stack(reductions, dim=1)
    apply_ranking(reductions, "oracle")

    # Train a linear scorer on a training subset, labels = oracle best ROI.
    subset = train_rows[:3000]
    print(json.dumps({"train_scorer_on": len(subset)}), flush=True)
    train_fg, train_local, _, train_energy, train_labels = embed(subset)
    train_fg, train_local = train_fg.to(device), train_local.to(device)
    with torch.no_grad():
        hidden = train_fg.new_zeros(1, train_fg.shape[0], 512)
        train_stop = fusion.head(torch.cat([train_fg, hidden.squeeze(0)], dim=1))
        train_base = per_sample_loss(train_stop, train_labels.to(device), pos_weight)
        train_gain = []
        for roi in range(9):
            order = torch.full((len(subset), 1), roi, dtype=torch.long)
            step_outputs = fusion.forward_order(train_fg, train_local, order.to(device))
            train_gain.append((train_base - per_sample_loss(step_outputs[-1], train_labels.to(device), pos_weight)).cpu())
    best_roi = torch.stack(train_gain, dim=1).argmax(dim=1)
    scorer = nn.Linear(9, 9).to(device)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=1e-2)
    features = train_energy.to(device)
    goal = best_roi.to(device)
    for _ in range(200):
        loss = F.cross_entropy(scorer(features), goal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        learned = scorer(energy.to(device)).cpu()
    apply_ranking(learned, "learned_topk")
    with torch.no_grad():
        report["scorer_train_accuracy"] = float((scorer(features).argmax(1) == goal).float().mean())
    out = ROOT / "checkpoints/rarezoom/selector_val.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
