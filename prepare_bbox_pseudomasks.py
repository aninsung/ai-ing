"""Convert NIH BBox annotations into binary weak pseudo-masks.

The rectangles are deliberately marked as pseudo labels; they are not pixel-level
ground truth and must not be reported as true segmentation masks.

Only patient-disjoint *train* rows are written to the manifest. Pseudo labels are
rejected in val/test by ``src.seg_manifest``. Mask PNGs are still written for every
matched image so they can be reviewed or used by ROI smoke scripts that ignore the
manifest split rules.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from prepare_seg_data import patient_split


def prepare(bbox_csv, images, mask_dir, manifest, seed=42):
    image_root, mask_root = Path(images), Path(mask_dir)
    mask_root.mkdir(parents=True, exist_ok=True)
    boxes_by_name = {}
    with Path(bbox_csv).open(encoding='utf-8-sig', newline='') as file:
        for row in csv.reader(file):
            if not row or row[0] == 'Image Index':
                continue
            try:
                name, label = row[0].strip(), row[1].strip()
                x, y, w, h = map(float, row[2:6])
            except (ValueError, IndexError):
                continue
            boxes_by_name.setdefault(name, []).append((x, y, w, h, label))

    image_index = {path.name: path for path in image_root.rglob('*.png')}
    output, skipped_missing, skipped_non_train, masks_written = [], 0, 0, 0
    for name, boxes in boxes_by_name.items():
        image_path = image_index.get(name)
        if image_path is None:
            skipped_missing += 1
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        # Official coordinates refer to 1024x1024 originals; resize to local image.
        sx, sy = width / 1024.0, height / 1024.0
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for x, y, w, h, _ in boxes:
            draw.rectangle(
                (round(x * sx), round(y * sy), round((x + w) * sx), round((y + h) * sy)),
                fill=255,
            )
        out = mask_root / name
        out.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out)
        masks_written += 1
        # NIH filenames are zero-padded (e.g. 00000001_000.png); CSV Patient ID is "1".
        raw_patient = name.split('_')[0]
        patient = str(int(raw_patient)) if raw_patient.isdigit() else raw_patient
        split = patient_split(patient, seed=seed)
        if split != 'train':
            # Keep on-disk mask for review; do not emit invalid pseudo rows.
            skipped_non_train += 1
            continue
        labels = '|'.join(sorted({box[4] for box in boxes}))
        output.append({
            'patient_id': patient,
            'image_path': str(image_path.resolve()),
            'mask_path': str(out.resolve()),
            'image_label': labels,
            'annotation_type': 'pseudo',
            'review_status': 'pending',
            'split': 'train',
        })

    manifest_path = Path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['patient_id', 'image_path', 'mask_path', 'image_label',
              'annotation_type', 'review_status', 'split']
    with manifest_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return {
        'images': len(output),
        'masks_written': masks_written,
        'skipped_missing': skipped_missing,
        'skipped_non_train': skipped_non_train,
        'manifest': str(manifest_path),
        'mask_dir': str(mask_root),
        'note': 'Pseudo rows are train/pending only; approve before supervised training',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bbox-csv', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--mask-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = prepare(args.bbox_csv, args.images, args.mask_dir, args.manifest, args.seed)
    print(report)


if __name__ == '__main__':
    main()
