"""End-to-end smoke on synthetic data: prepare → approve → validate → 1-epoch train."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_seg_data import patient_split  # noqa: E402


def write_png(path: Path, size=(128, 128), box=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('L', size, 30)
    if box:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, fill=200)
    image.save(path)


def write_mask(path: Path, size=(128, 128), box=(40, 40, 80, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    mask.save(path)


def run(cmd, cwd=ROOT):
    print('+', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def main():
    work = Path(tempfile.mkdtemp(prefix='ai-ing-smoke-'))
    try:
        images = work / 'images'
        masks = work / 'masks'
        # Patient IDs chosen so SHA split puts 1 in train and 2 in val (seed=42).
        # One train + one val for supervised seg; several train patients so the
        # weak pseudo trainer's internal re-split still has train and val rows.
        samples = []
        train_count = 0
        for patient in range(1, 800):
            split = patient_split(str(patient))
            if split == 'val' and not any(s['split'] == 'val' for s in samples):
                pass
            elif split == 'train' and train_count < 12:
                pass
            else:
                continue
            name = f'{patient:08d}_000.png'
            write_png(images / name, box=(40, 40, 80, 80))
            write_mask(masks / f'{patient:08d}_mask.png')
            samples.append({'patient': str(patient), 'name': name, 'split': split})
            if split == 'train':
                train_count += 1
            if train_count >= 12 and any(s['split'] == 'val' for s in samples):
                break
        assert train_count >= 12 and any(s['split'] == 'val' for s in samples), samples

        # --- BBox pseudo prepare ---
        bbox_csv = work / 'bbox.csv'
        with bbox_csv.open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image Index', 'Finding Label', 'Bbox [x]', 'Bbox [y]', 'Bbox [w]', 'Bbox [h]'])
            for sample in samples:
                # Coordinates in 1024 space for a box covering roughly the drawn region.
                writer.writerow([sample['name'], 'Nodule', 320, 320, 320, 320])
        pseudo_manifest = work / 'pseudo_manifest.csv'
        run([
            sys.executable, 'prepare_bbox_pseudomasks.py',
            '--bbox-csv', str(bbox_csv),
            '--images', str(images),
            '--mask-dir', str(work / 'pseudo_masks'),
            '--manifest', str(pseudo_manifest),
        ])

        # --- Formal seg manifest with approved GT masks (1 train + 1 val) ---
        seg_manifest = work / 'seg_manifest.csv'
        fields = ['patient_id', 'image_path', 'mask_path', 'image_label',
                  'annotation_type', 'review_status', 'split']
        formal = [next(s for s in samples if s['split'] == 'train'),
                  next(s for s in samples if s['split'] == 'val')]
        with seg_manifest.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for sample in formal:
                writer.writerow({
                    'patient_id': sample['patient'],
                    'image_path': str((images / sample['name']).resolve()),
                    'mask_path': str((masks / f"{int(sample['patient']):08d}_mask.png").resolve()),
                    'image_label': '1',
                    'annotation_type': 'mask',
                    'review_status': 'approved',
                    'split': sample['split'],
                })

        cfg = work / 'segmentation.yaml'
        save_dir = work / 'checkpoints'
        cfg.write_text(
            'data:\n'
            f'  manifest: {seg_manifest}\n'
            '  image_size: 64\n'
            '  allow_pseudo: false\n'
            'train:\n'
            '  seed: 42\n'
            '  batch_size: 1\n'
            '  num_workers: 0\n'
            '  epochs: 1\n'
            '  learning_rate: 0.0001\n'
            '  threshold: 0.5\n'
            f'  save_dir: {save_dir}\n',
            encoding='utf-8',
        )

        run([sys.executable, 'train_seg.py', '--config', str(cfg), '--validate-only'])
        run([sys.executable, 'train_seg.py', '--config', str(cfg)])

        # Pseudo smoke: approve the prepared train row and train 1 epoch.
        approved = work / 'pseudo_approved.csv'
        with pseudo_manifest.open(encoding='utf-8-sig', newline='') as src, \
                approved.open('w', encoding='utf-8', newline='') as dst:
            rows = list(csv.DictReader(src))
            writer = csv.DictWriter(dst, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                row['review_status'] = 'approved'
                writer.writerow(row)
        run([
            sys.executable, 'train_pseudo_segmentation.py',
            '--manifest', str(approved),
            '--epochs', '1',
            '--image-size', '64',
            '--batch-size', '1',
            '--output', str(work / 'pseudo_unetpp.pth'),
        ])

        print(json.dumps({'ok': True, 'work': str(work), 'save_dir': str(save_dir)}, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
