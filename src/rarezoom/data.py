"""NIH ChestX-ray14 loader. Official test patients stay out of train and validation."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .roi import BOXES, load_raw, view_tensor, crop_tensor

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def patient_bucket(patient_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest()[:16]
    return int(digest, 16) / 2**64


def load_records(root: Path, seed: int = 42, val_fraction: float = 0.15):
    root = Path(root)
    metadata = root / "Data_Entry_2017.csv"
    image_dir = root / "raw" / "images"
    train_names = set((root / "train_val_list.txt").read_text().split())
    test_names = set((root / "test_list.txt").read_text().split())
    if train_names & test_names:
        raise ValueError("Official train and test lists overlap")
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")

    records = []
    with metadata.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            name = row["Image Index"]
            labels = [part for part in row["Finding Labels"].split("|") if part and part != "No Finding"]
            unknown = [label for label in labels if label not in DISEASES]
            if unknown:
                raise ValueError(f"Unexpected label on {name}: {unknown}")
            if name in test_names:
                split = "test"
            elif name in train_names:
                split = "val" if patient_bucket(row["Patient ID"], seed) < val_fraction else "train"
            else:
                raise ValueError(f"{name} is in neither official list")
            target = np.zeros(len(DISEASES), dtype=np.float32)
            for label in labels:
                target[DISEASES.index(label)] = 1.0
            path = image_dir / name
            records.append({
                "path": path,
                "label": target,
                "patient_id": row["Patient ID"],
                "split": split,
                "image_index": name,
            })

    missing = [record["image_index"] for record in records if not record["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[:3]}")
    patients = {split: {record["patient_id"] for record in records if record["split"] == split}
                for split in ("train", "val", "test")}
    leaked = {f"{left}_{right}": len(patients[left] & patients[right])
              for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(leaked.values()):
        raise ValueError(f"Patient leakage: {leaked}")
    summary = {
        "images": {split: sum(record["split"] == split for record in records) for split in patients},
        "patients": {split: len(ids) for split, ids in patients.items()},
        "patient_intersections": leaked,
    }
    return records, summary


def positive_weights(records, clip):
    labels = np.stack([record["label"] for record in records if record["split"] == "train"])
    positive = labels.sum(axis=0)
    negative = len(labels) - positive
    if np.any(positive == 0):
        raise ValueError("A training disease has no positive samples")
    weights = np.clip(negative / positive, clip[0], clip[1]).astype(np.float32)
    return weights, positive.astype(int), negative.astype(int)


class ChestDataset(Dataset):
    def __init__(self, records, image_size=224, augment=False):
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as source:
            image = source.convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.augment and torch.rand(()) < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), torch.from_numpy(record["label"])


class SequenceDataset(Dataset):
    """Global view plus 1–3 raw-image ROI crops. Validation uses a fixed trio."""

    def __init__(self, records, augment=False, fixed=False, max_zooms=3):
        self.records = records
        self.augment = augment
        self.fixed = fixed
        self.max_zooms = max_zooms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, _, _ = load_raw(record["path"])
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
        if self.fixed:
            order = [4, 0, 8][:self.max_zooms]
        else:
            count = random.randint(1, self.max_zooms)
            order = random.sample(range(len(BOXES)), count)
        crops = torch.stack([crop_tensor(image, roi) for roi in order])
        label = torch.from_numpy(record["label"])
        return view_tensor(image), crops, label


def collate_sequences(batch):
    globals_ = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    lengths = torch.tensor([item[1].shape[0] for item in batch], dtype=torch.long)
    steps = int(lengths.max())
    _, channels, height, width = batch[0][1].shape
    crops = globals_.new_zeros(len(batch), steps, channels, height, width)
    for index, item in enumerate(batch):
        crops[index, :item[1].shape[0]] = item[1]
    return globals_, crops, lengths, labels
