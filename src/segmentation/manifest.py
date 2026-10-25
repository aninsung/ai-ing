"""Manifest validation independent of torch; unannotated is never negative."""
import csv
from pathlib import Path


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        required = {'patient_id', 'image_path', 'mask_path', 'annotation_type', 'review_status', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest must contain {sorted(required)}')
        rows = list(reader)
    patients, images = {}, set()
    for row in rows:
        for key, value in row.items():
            if key is not None and value is not None:
                row[key] = value.strip()
        patient = row['patient_id']
        split = row['split']
        if not patient or split not in {'train', 'val', 'test'}:
            raise ValueError('Every row needs a patient_id and train/val/test split')
        if patient in patients and patients[patient] != split:
            raise ValueError(f'Patient leakage: {patient}')
        patients[patient] = split
        if row['annotation_type'] not in {'unannotated', 'mask', 'negative', 'pseudo'}:
            raise ValueError('Invalid annotation_type')
        if row['review_status'] not in {'pending', 'approved', 'uncertain'}:
            raise ValueError('Invalid review_status')
        if not row['image_path']:
            raise ValueError('Empty image_path')
        for key in ('image_path', 'mask_path'):
            if row[key]:
                resolved = Path(row[key])
                if not resolved.is_absolute():
                    resolved = path.parent / resolved
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                row[key] = str(resolved.resolve())
        if row['image_path'] in images:
            raise ValueError(f'Duplicate image: {row["image_path"]}')
        images.add(row['image_path'])
        if row['annotation_type'] in {'mask', 'pseudo'} and not row['mask_path']:
            raise ValueError('mask/pseudo rows require mask_path')
        if row['annotation_type'] == 'pseudo' and split != 'train':
            raise ValueError('Pseudo labels are allowed only in train')
        if row['annotation_type'] == 'unannotated' and row['mask_path']:
            raise ValueError('Set annotation_type=mask when supplying a ground-truth mask')
    return rows


def supervised_rows(rows, split, allow_pseudo=False):
    types = {'mask', 'negative'}
    if allow_pseudo and split == 'train':
        types.add('pseudo')
    result = [r for r in rows if r['split'] == split and
              r['review_status'] == 'approved' and r['annotation_type'] in types]
    if not result:
        raise ValueError(f'No approved masks/verified negatives for {split}. '
                         'Annotate and review the manifest first; image labels are insufficient.')
    return result
