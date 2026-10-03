"""Audit NIH images and create an unlabelled, patient-disjoint manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FIELDS = ['patient_id', 'image_path', 'mask_path', 'image_label',
          'annotation_type', 'review_status', 'split']


def patient_split(patient_id, seed=42):
    value = int(hashlib.sha256(f'{seed}:{patient_id}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if value < .7 else 'val' if value < .85 else 'test'


def prepare(csv_path, image_dir, output, seed=42):
    index = {}
    for path in sorted(Path(image_dir).rglob('*')):
        if path.is_file() and path.suffix.lower() == '.png':
            if path.name in index:
                raise ValueError(f'Duplicate image filename: {path.name}')
            index[path.name] = str(path.resolve())
    rows, missing, seen = [], [], set()
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        required = {'Patient ID', 'Image Index', 'Finding Labels'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must contain {sorted(required)}')
        for row in reader:
            patient = row['Patient ID'].strip()
            name = row['Image Index'].strip()
            if not patient or not name:
                raise ValueError('Empty patient ID or image name')
            if name in seen:
                raise ValueError(f'Duplicate CSV image: {name}')
            seen.add(name)
            if name not in index:
                missing.append(name)
                continue
            finding = row['Finding Labels'].strip()
            label = '' if not finding else int('Nodule' in finding.split('|'))
            rows.append(dict(zip(FIELDS, [patient, index[name], '', label,
                                         'unannotated', 'pending', patient_split(patient, seed)])))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {'seed': seed, 'split_method': 'patient SHA256 buckets 70/15/15; approximate ratios',
              'images': len(rows), 'missing_images': missing, 'splits': {}}
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split]
        report['splits'][split] = {'images': len(subset),
                                  'patients': len({r['patient_id'] for r in subset}),
                                  'nodule_images': sum(r['image_label'] == 1 for r in subset)}
    output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', default='data/seg_manifest.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.csv, args.images, args.output, args.seed), indent=2))
