"""Cost-aware adaptive resolution: read every image at 224, re-read a fraction at 512.

Validation patients are split in half. Half A fits per-class Platt calibration
and the learned escalation value; half B picks the escalation fraction. The
official test list is scored once at that fraction.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from scripts.rarezoom.resolution_study import KEYS, bootstrap_metrics, ci, predict
from src.rarezoom.data import DISEASES, load_records, positive_weights
from src.rarezoom.metrics import summarize
from src.rarezoom.model import GlobalClassifier

FRACTIONS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(z, y, steps=50):
    """Per-class logistic calibration sigmoid(a*z + b), fitted by Newton's method."""
    params = np.zeros((z.shape[1], 2))
    params[:, 0] = 1.0
    for c in range(z.shape[1]):
        a, b = 1.0, 0.0
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(a * z[:, c] + b)))
            g = p - y[:, c]
            w = p * (1 - p) + 1e-9
            grad = np.array([(g * z[:, c]).sum(), g.sum()])
            hess = np.array([[(w * z[:, c] ** 2).sum(), (w * z[:, c]).sum()],
                             [(w * z[:, c]).sum(), w.sum()]]) + 1e-6 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            a, b = a - step[0], b - step[1]
            if np.abs(step).max() < 1e-8:
                break
        params[c] = a, b
    return params


def calibrate(p, params):
    return 1 / (1 + np.exp(-(logit(p) * params[:, 0] + params[:, 1])))


def weighted_bce(p, y, pos_weight):
    p = np.clip(p, EPS, 1 - EPS)
    return -(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p)).sum(axis=1)


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def state(p224):
    return np.concatenate([logit(p224), entropy(p224)], axis=1).astype(np.float32)


def train_value(x, gain, seed, epochs=300):
    """One-step Q(s, ZOOM) - Q(s, STOP): the expected loss reduction from the 512 read."""
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(gain.astype(np.float32))
    for _ in range(epochs):
        for index in torch.randperm(len(xt)).split(256):
            loss = torch.nn.functional.smooth_l1_loss(net(xt[index]).squeeze(1), yt[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()

    def score(z):
        with torch.no_grad():
            return net(torch.from_numpy((z - mean) / std)).squeeze(1).numpy()
    return score


def mix(p224, p512, priority, fraction):
    count = int(round(fraction * len(p224)))
    out = p224.copy()
    if count:
        chosen = np.argsort(-priority, kind="stable")[:count]
        out[chosen] = p512[chosen]
    return out


def macro(y, p):
    return summarize(y, p, DISEASES)["macro_auprc"]


def gflops(size):
    model = GlobalClassifier(len(DISEASES), pretrained=False).eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 1, size, size))
    return counter.get_total_flops() / 1e9


def load_scores(base, run, size, rows, device):
    path = base / "predictions" / f"{run}.npz"
    if path.exists():
        saved = np.load(path)
        return {split: saved[split] for split in rows}
    checkpoint = torch.load(base / f"s{run}" / "best_model.pth", map_location="cpu", weights_only=False)
    model = GlobalClassifier(len(DISEASES), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    scores = {split: predict(model, rows[split], size, device, 16) for split in rows}
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **scores)
    return scores


def run_seed(seed, alt_seed, base, rows, y, patients, half, pos_weight, cost, margin, reps, device):
    low = load_scores(base, f"224_seed{seed}", 224, rows, device)
    high = load_scores(base, f"512_seed{seed}", 512, rows, device)
    alt = load_scores(base, f"224_seed{alt_seed}", 224, rows, device)
    a, b = half == 0, half == 1
    cal224 = fit_platt(logit(low["val"][a]), y["val"][a])
    cal512 = fit_platt(logit(high["val"][a]), y["val"][a])
    calalt = fit_platt(logit(alt["val"][a]), y["val"][a])
    p224 = {s: calibrate(low[s], cal224) for s in rows}
    p512 = {s: calibrate(high[s], cal512) for s in rows}
    palt = {s: calibrate(alt[s], calalt) for s in rows}
    gain = {s: weighted_bce(p224[s], y[s], pos_weight) - weighted_bce(p512[s], y[s], pos_weight) for s in rows}
    value = train_value(state(p224["val"][a]), gain["val"][a], seed)
    rng = np.random.default_rng(seed)

    def priorities(split, mask=slice(None)):
        n = len(p224[split][mask])
        return {
            "uncertainty": entropy(p224[split][mask]).sum(axis=1),
            "learned": value(state(p224[split][mask])),
            "random": rng.random(n),
            "oracle": gain[split][mask],
        }

    def curve(split, mask=slice(None)):
        pri = priorities(split, mask)
        ys, lo, hi = y[split][mask], p224[split][mask], p512[split][mask]
        out = {}
        for name, order in pri.items():
            out[name] = [macro(ys, mix(lo, hi, order, f)) for f in FRACTIONS]
        out["random"] = np.mean([[macro(ys, mix(lo, hi, rng.random(len(lo)), f)) for f in FRACTIONS]
                                 for _ in range(10)], axis=0).tolist()
        return out, macro(ys, lo), macro(ys, hi)

    val_curve, val224, val512 = curve("val", b)
    chosen = {}
    for policy in ("uncertainty", "learned"):
        ok = [f for f, v in zip(FRACTIONS, val_curve[policy]) if v >= val512 - margin]
        chosen[f"{policy}_margin"] = float(ok[0]) if ok else 1.0
        chosen[f"{policy}_max"] = float(FRACTIONS[int(np.argmax(val_curve[policy]))])

    test_curve, test224, test512 = curve("test")
    pri = priorities("test")
    scores = {"224": p224["test"], "512": p512["test"]}
    for name, f in chosen.items():
        scores[name] = mix(p224["test"], p512["test"], pri[name.rsplit("_", 1)[0]], f)
        scores[f"{name}_alt224"] = mix(p224["test"], palt["test"], pri[name.rsplit("_", 1)[0]], f)
        for k in range(5):
            scores[f"random_{name}_{k}"] = mix(p224["test"], p512["test"], rng.random(len(p224["test"])), f)
    boot = bootstrap_metrics(y["test"], scores, patients["test"], reps, np.random.default_rng(2026 + seed))
    observed = {run: dict(zip(KEYS, [summarize(y["test"], s, DISEASES)["macro_auprc"]]
                              + [summarize(y["test"], s, DISEASES)["auprc"][k] for k in KEYS[1:]]))
                for run, s in scores.items() if not run.startswith("random_")}

    tests = {}
    for name in chosen:
        random_boot = np.mean([boot[f"random_{name}_{k}"] for k in range(5)], axis=0)
        random_point = np.mean([[macro(y["test"], scores[f"random_{name}_{k}"])] for k in range(5)])
        f = chosen[name]
        tests[name] = {
            "fraction": f,
            "relative_compute_vs_512": (cost[224] + f * cost[512]) / cost[512],
            "gflops_per_image": cost[224] + f * cost[512],
            "test_macro_auprc": observed[name]["macro_auprc"],
            "random_same_fraction_macro_auprc": float(random_point),
            "minus_512": {key: ci(boot[name][:, i] - boot["512"][:, i]) | {
                "point": observed[name][key] - observed["512"][key]} for i, key in enumerate(KEYS)},
            "minus_224": {key: ci(boot[name][:, i] - boot["224"][:, i]) | {
                "point": observed[name][key] - observed["224"][key]} for i, key in enumerate(KEYS)},
            "minus_random": {"macro_auprc": ci(boot[name][:, 0] - random_boot[:, 0]) | {
                "point": observed[name]["macro_auprc"] - float(random_point)}},
            "same_images_second_224_model": observed[f"{name}_alt224"],
            "minus_second_224_model": {key: ci(boot[name][:, i] - boot[f"{name}_alt224"][:, i]) | {
                "point": observed[name][key] - observed[f"{name}_alt224"][key]} for i, key in enumerate(KEYS)},
        }
    return {
        "val_half_b": {"224": val224, "512": val512, "curve": val_curve},
        "chosen_fraction": chosen,
        "test": {"224": test224, "512": test512, "curve": test_curve, "observed": observed, "operating_point": tests},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/rarezoom/v4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 44])
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--alt-seed", type=int, default=43, help="Second 224 model for the ensemble control")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", default="cascade_report.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    base = ROOT / args.root
    config = yaml.safe_load((ROOT / "configs/rarezoom.yaml").read_text(encoding="utf-8"))
    records, _ = load_records(ROOT / "data", seed=42)
    pos_weight, _, _ = positive_weights(records, config["reward"]["class_weight_clip"])
    rows = {split: [r for r in records if r["split"] == split] for split in ("val", "test")}
    y = {s: np.stack([r["label"] for r in rows[s]]).astype(np.float64) for s in rows}
    patients = {s: np.array([r["patient_id"] for r in rows[s]]) for s in rows}
    half = np.array([int(hashlib.sha256(f"cascade:{pid}".encode()).hexdigest(), 16) % 2 for pid in patients["val"]])
    cost = {224: gflops(224), 512: gflops(512)}
    report = {
        "protocol": {
            "policy": "all images read at 224; a fraction re-read at 512 replaces the 224 probabilities",
            "calibration": "per-class Platt on validation half A",
            "learned_value": "MLP on [logit, entropy] of calibrated 224 probabilities regressing the "
                             "class-weighted BCE reduction from the 512 read (one-step Q(ZOOM)-Q(STOP)), half A",
            "fraction_rules": {
                "margin": f"smallest fraction whose half-B macro-AUPRC >= 512 minus {args.margin} (pre-specified)",
                "max": "fraction with the highest half-B macro-AUPRC (fixed after seed 42; seed 44 is confirmatory)",
            },
            "test_usage": "scored once at the half-B fraction",
            "val_half_sizes": [int((half == 0).sum()), int((half == 1).sum())],
            "gflops_per_image": cost,
        },
        "seeds": {},
    }
    for seed in args.seeds:
        result = run_seed(seed, args.alt_seed, base, rows, y, patients, half, pos_weight, cost,
                          args.margin, args.bootstrap, device)
        report["seeds"][str(seed)] = result
        print(json.dumps({seed: {"chosen": result["chosen_fraction"], "op": {
            k: {kk: v[kk] for kk in ("fraction", "relative_compute_vs_512", "test_macro_auprc",
                                     "random_same_fraction_macro_auprc")}
            for k, v in result["test"]["operating_point"].items()},
            "test224": result["test"]["224"], "test512": result["test"]["512"]}}, indent=1), flush=True)
    out = base / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
