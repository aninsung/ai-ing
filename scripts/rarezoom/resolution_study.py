"""Multi-seed input-resolution study for the global ResNet50 classifier.

Every checkpoint was selected on the patient-disjoint validation split during
training. The official test list is only scored here, once per checkpoint.
Patient-level bootstrap resamples whole patients; resampled AP is computed with
per-image multiplicity weights, which matches duplicated-row AP up to ties.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rarezoom.data import DISEASES, ChestDataset, load_records
from src.rarezoom.metrics import summarize
from src.rarezoom.model import GlobalClassifier

FOCUS = ["Nodule", "Mass", "Pneumothorax"]
KEYS = ["macro_auprc"] + FOCUS


@torch.no_grad()
def predict(model, rows, size, device, workers):
    loader = DataLoader(ChestDataset(rows, size, augment=False), batch_size=128, shuffle=False,
                        num_workers=workers, pin_memory=True)
    scores = []
    for images, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images.to(device, non_blocking=True))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(scores)


def brief(metrics):
    return {
        "macro_auprc": metrics["macro_auprc"],
        "macro_auroc": metrics["macro_auroc"],
        **{name: metrics["auprc"][name] for name in FOCUS},
    }


def weighted_ap(target, score, weights):
    """AP for each bootstrap replicate. weights: (reps, n) image multiplicities."""
    order = np.argsort(-score, kind="stable")
    t = target[order].astype(np.float64)
    w = weights[:, order]
    hits = np.cumsum(w * t, axis=1)
    seen = np.cumsum(w, axis=1)
    precision = hits / np.maximum(seen, 1e-12)
    positive = (w * t).sum(axis=1)
    return (w * t * precision).sum(axis=1) / np.maximum(positive, 1e-12)


def bootstrap_metrics(target, scores, patients, reps, rng, chunk=50):
    """Return {run: (reps, len(KEYS))} with paired patient resampling across runs."""
    unique, inverse = np.unique(patients, return_inverse=True)
    out = {run: [] for run in scores}
    focus_index = [DISEASES.index(name) for name in FOCUS]
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        draws = rng.integers(0, len(unique), size=(count, len(unique)))
        patient_weight = np.stack([np.bincount(row, minlength=len(unique)) for row in draws])
        weights = patient_weight[:, inverse].astype(np.float64)
        for run, score in scores.items():
            ap = np.stack([weighted_ap(target[:, c], score[:, c], weights) for c in range(len(DISEASES))], axis=1)
            out[run].append(np.concatenate([ap.mean(axis=1, keepdims=True), ap[:, focus_index]], axis=1))
    return {run: np.concatenate(parts) for run, parts in out.items()}


def ci(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "excludes_zero": bool(low > 0 or high < 0)}


def spread(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


@torch.no_grad()
def compute_profile(sizes, device, repeats=50):
    model = GlobalClassifier(len(DISEASES), pretrained=False).to(device).eval()
    profile = {}
    for size in sizes:
        x1 = torch.randn(1, 1, size, size, device=device)
        counter = FlopCounterMode(display=False)
        with counter:
            model(x1)
        flops = counter.get_total_flops()
        timing = {}
        for batch in (1, 64):
            x = torch.randn(batch, 1, size, size, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for _ in range(10):
                    model(x)
                torch.cuda.synchronize()
                samples = []
                for _ in range(repeats):
                    begin = time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - begin)
            per_image_ms = 1000 * float(np.median(samples)) / batch
            timing[f"batch{batch}_ms_per_image"] = per_image_ms
        profile[str(size)] = {"gflops_per_image": flops / 1e9, "relative_pixels": (size / 224) ** 2, **timing}
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/rarezoom/v4")
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 512])
    parser.add_argument("--profile-sizes", type=int, nargs="+", default=[224, 384, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda")
    base = ROOT / args.root
    records, _ = load_records(ROOT / "data", seed=42)
    rows = {split: [r for r in records if r["split"] == split] for split in ("val", "test")}
    targets = {split: np.stack([r["label"] for r in rows[split]]).astype(np.float32) for split in rows}
    patients = {split: np.array([r["patient_id"] for r in rows[split]]) for split in rows}

    cache = base / "predictions"
    cache.mkdir(exist_ok=True)
    scores = {split: {} for split in rows}
    runs = {}
    for size in args.sizes:
        for seed in args.seeds:
            run = f"{size}_seed{seed}"
            path = base / f"s{size}_seed{seed}" / "best_model.pth"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            runs[run] = {"size": size, "seed": seed, "selected_epoch": checkpoint["epoch"],
                         "batch_size": checkpoint.get("batch_size", 32)}
            npz = cache / f"{run}.npz"
            if npz.exists():
                saved = np.load(npz)
                for split in rows:
                    scores[split][run] = saved[split]
            else:
                model = GlobalClassifier(len(DISEASES), pretrained=False)
                model.load_state_dict(checkpoint["model"])
                model.to(device).eval()
                for split in rows:
                    scores[split][run] = predict(model, rows[split], size, device, args.workers)
                np.savez(npz, **{split: scores[split][run] for split in rows})
                del model
                torch.cuda.empty_cache()
            for split in rows:
                runs[run][split] = brief(summarize(targets[split], scores[split][run], DISEASES))
            print(json.dumps({run: runs[run]}), flush=True)

    per_size = {}
    for size in args.sizes:
        per_size[str(size)] = {}
        for split in rows:
            group = [runs[f"{size}_seed{seed}"][split] for seed in args.seeds]
            per_size[str(size)][split] = {key: spread([g[key] for g in group])
                                          for key in ["macro_auprc", "macro_auroc"] + FOCUS}

    pairs = [(b, a) for a, b in [(224, 384), (224, 512), (384, 512)] if a in args.sizes and b in args.sizes]
    comparisons = {}
    for split in rows:
        rng = np.random.default_rng(2026)
        boot = bootstrap_metrics(targets[split], scores[split], patients[split], args.bootstrap, rng)
        for high, low in pairs:
            name = f"{high}_minus_{low}"
            entry = comparisons.setdefault(name, {}).setdefault(split, {"per_seed": {}, "seed_mean": {}})
            diffs = []
            for seed in args.seeds:
                hi, lo = f"{high}_seed{seed}", f"{low}_seed{seed}"
                diff = boot[hi] - boot[lo]
                diffs.append(diff)
                entry["per_seed"][str(seed)] = {
                    key: {"point": runs[hi][split][key] - runs[lo][split][key], **ci(diff[:, k])}
                    for k, key in enumerate(KEYS)
                }
            mean_diff = np.mean(diffs, axis=0)
            for k, key in enumerate(KEYS):
                points = [entry["per_seed"][str(seed)][key]["point"] for seed in args.seeds]
                entry["seed_mean"][key] = {
                    "point": float(np.mean(points)), "seed_min": float(np.min(points)),
                    "seed_max": float(np.max(points)), "all_seeds_positive": bool(min(points) > 0),
                    **ci(mean_diff[:, k]),
                }

    report = {
        "protocol": {
            "selection": "best validation macro-AUPRC epoch, patience 3, batch 32, AdamW lr 1e-4",
            "split_seed": 42, "training_seeds": args.seeds, "bootstrap": args.bootstrap,
            "bootstrap_unit": "patient (paired across runs)",
            "test_usage": "scored once per validation-selected checkpoint; never used for selection",
            "val_images": len(rows["val"]), "test_images": len(rows["test"]),
            "test_patients": int(len(np.unique(patients["test"]))),
        },
        "runs": runs,
        "per_size": per_size,
        "comparisons": comparisons,
        "compute": compute_profile(args.profile_sizes, device),
        "gpu": torch.cuda.get_device_name(0),
    }
    out = base / "resolution_seeds_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"per_size": per_size, "comparisons_seed_mean": {
        name: {split: value[split]["seed_mean"] for split in value} for name, value in comparisons.items()
    }, "compute": report["compute"]}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
