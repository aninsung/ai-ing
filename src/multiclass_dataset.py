"""Dataset for the NIH multiclass head plus auxiliary multilabel head."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
CLASSES = ["No Finding", *DISEASES]


def primary_sample_weights(rows, power=1.0):
    """Inverse-frequency draw weights. Rare primary classes are sampled more often.

    Rows without a single primary label keep the median labeled-class weight so
    the auxiliary head still sees them.
    """
    if power < 0:
        raise ValueError("power must be non-negative")
    labels = []
    counts = {}
    for row in rows:
        raw = row["primary_class_id"]
        label = int(raw) if raw else -1
        labels.append(label)
        if label >= 0:
            counts[label] = counts.get(label, 0) + 1
    labeled_counts = list(counts.values())
    fallback = float(np.median(labeled_counts)) if labeled_counts else 1.0
    weights = []
    for label in labels:
        count = counts[label] if label >= 0 else fallback
        weights.append((1.0 / count) ** power)
    return weights


def read_manifest(path, split):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == split]
    if not rows:
        raise ValueError(f"Manifest has no rows for split={split}")
    patients = {row["patient_id"] for row in rows}
    if len(patients) < 2 and split != "test":
        raise ValueError(f"Split {split} has too few patients")
    return rows


class NIHMultiTaskDataset(Dataset):
    def __init__(self, rows, image_root, augment=False, image_size=224):
        self.rows = rows
        self.image_root = Path(image_root)
        self.augment = augment
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Manifest is generated on Windows but may be consumed from WSL/Linux.
        relative_path = Path(row["image_path"].replace("\\", "/"))
        path = self.image_root / relative_path
        with Image.open(path) as source:
            image = np.asarray(
                source.convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).copy()
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image[:, ::-1].copy()
            if torch.rand(()) < 0.2:
                image = np.clip(image * float(torch.empty((),).uniform_(0.9, 1.1)), 0, 255)
        image = torch.from_numpy(image / 255.0).unsqueeze(0)
        primary = int(row["primary_class_id"]) if row["primary_class_id"] else -1
        auxiliary = torch.tensor([int(row[f"label_{label}"]) for label in DISEASES], dtype=torch.float32)
        return {"image": image, "primary": torch.tensor(primary, dtype=torch.long),
                "auxiliary": auxiliary, "is_multilabel": row["is_multilabel"] == "1",
                "image_index": row["image_index"], "patient_id": row["patient_id"]}
