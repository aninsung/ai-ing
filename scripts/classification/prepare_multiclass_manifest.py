"""Build a patient-disjoint NIH manifest for multiclass + multilabel experiments.

The official NIH train/test lists are preserved. A deterministic patient-level
validation split is carved out of official train_val only. Multi-label rows are
kept for the auxiliary sigmoid head and are never assigned a fake multiclass
target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
CLASSES = ["No Finding", *DISEASES]


def bucket(patient_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()[:16]
    return int(digest, 16) / 2**64


def make_manifest(archive: Path, output: Path, seed: int = 42, val_fraction: float = 0.15):
    metadata_path = archive / "Data_Entry_2017.csv"
    image_root = archive / "images-224" / "images-224"
    train_names = set((archive / "train_val_list_NIH.txt").read_text().splitlines())
    test_names = set((archive / "test_list_NIH.txt").read_text().splitlines())
    if train_names & test_names:
        raise ValueError("Official train/test image lists overlap")
    image_paths = {path.name: path for path in image_root.glob("*.png")}
    missing = sorted((train_names | test_names) - image_paths.keys())
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} official images; first: {missing[:3]}")

    rows = []
    with metadata_path.open(encoding="utf-8-sig", newline="") as file:
        for record in csv.DictReader(file):
            name = record["Image Index"]
            labels = tuple(x for x in record["Finding Labels"].split("|") if x)
            if name not in train_names and name not in test_names:
                raise ValueError(f"Metadata image is absent from official lists: {name}")
            if any(label not in CLASSES for label in labels):
                raise ValueError(f"Unexpected label in {name}: {labels}")
            if labels == ("No Finding",):
                primary = "No Finding"
            elif len(labels) == 1:
                primary = labels[0]
            else:
                primary = ""  # multi-label rows are auxiliary-only
            official = "test" if name in test_names else "train_val"
            split = "test" if official == "test" else (
                "val" if bucket(record["Patient ID"], seed) < val_fraction else "train"
            )
            rows.append({
                "image_index": name,
                "patient_id": record["Patient ID"],
                "image_path": str(image_paths[name].relative_to(archive)),
                "official_split": official,
                "finding_labels": "|".join(labels),
                "primary_label": primary,
                "primary_class_id": "" if not primary else str(CLASSES.index(primary)),
                "is_multilabel": str(int(len(labels) > 1)),
                "split": split,
                **{f"label_{label}": str(int(label in labels)) for label in DISEASES},
            })
    if len(rows) != len(image_paths):
        raise ValueError(f"Metadata rows ({len(rows)}) and image files ({len(image_paths)}) differ")
    fields = list(rows[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    patients_by_split = {s: {r["patient_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    report = {
        "classes": CLASSES,
        "seed": seed,
        "validation_rule": f"patient SHA256 bucket < {val_fraction}",
        "rows": len(rows),
        "split_images": dict(Counter(r["split"] for r in rows)),
        "official_split_images": dict(Counter(r["official_split"] for r in rows)),
        "split_patients": {k: len(v) for k, v in patients_by_split.items()},
        "patient_intersections": {
            f"{a}_{b}": len(patients_by_split[a] & patients_by_split[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "primary_class_counts": {
            split: dict(Counter(r["primary_label"] for r in rows if r["split"] == split and r["primary_label"]))
            for split in ("train", "val", "test")
        },
        "multilabel_images": {split: sum(r["is_multilabel"] == "1" for r in rows if r["split"] == split) for split in ("train", "val", "test")},
        "image_root": str(image_root.resolve()),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/archive")
    parser.add_argument("--output", default="data/nih_multiclass_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    print(json.dumps(make_manifest(Path(args.archive), Path(args.output), args.seed, args.val_fraction), indent=2))
