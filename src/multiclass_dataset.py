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
    def __init__(self, rows, image_root, augment=False):
        self.rows = rows
        self.image_root = Path(image_root)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Manifest is generated on Windows but may be consumed from WSL/Linux.
        relative_path = Path(row["image_path"].replace("\\", "/"))
        path = self.image_root / relative_path
        with Image.open(path) as source:
            image = np.asarray(source.convert("L"), dtype=np.float32).copy()
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
