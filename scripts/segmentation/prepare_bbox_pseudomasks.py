"""Convert NIH BBox annotations into binary weak pseudo-masks.

The rectangles are deliberately marked as pseudo labels; they are not pixel-level
ground truth and must not be reported as true segmentation masks.
"""
import argparse, csv
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bbox-csv', required=True); p.add_argument('--images', required=True)
    p.add_argument('--mask-dir', required=True); p.add_argument('--manifest', required=True)
    args = p.parse_args(); image_root, mask_root = Path(args.images), Path(args.mask_dir); mask_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    with Path(args.bbox_csv).open(encoding='utf-8-sig', newline='') as f:
        for r in csv.reader(f):
            if not r or r[0] == 'Image Index': continue
            try: name, label = r[0].strip(), r[1].strip(); x,y,w,h = map(float, r[2:6])
            except (ValueError, IndexError): continue
            rows.setdefault(name, []).append((x,y,w,h,label))
    image_index = {p.name: p for p in image_root.rglob('*.png')}
    output = []
    for name, boxes in rows.items():
        image_path = image_index.get(name)
        if image_path is None: continue
        with Image.open(image_path) as im: width, height = im.size
        # Official coordinates refer to 1024x1024 originals; resize to local image.
        sx, sy = width / 1024.0, height / 1024.0
        mask = Image.new('L', (width, height), 0); draw = ImageDraw.Draw(mask)
        for x,y,w,h,_ in boxes:
            draw.rectangle((round(x*sx), round(y*sy), round((x+w)*sx), round((y+h)*sy)), fill=255)
        out = mask_root / name; out.parent.mkdir(parents=True, exist_ok=True); mask.save(out)
        output.append((name, str(image_path), str(out), '|'.join(sorted({b[4] for b in boxes}))))
    manifest = Path(args.manifest); manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['patient_id','image_path','mask_path','image_label','annotation_type','review_status','split'])
        for name, image, mask, label in output:
            patient = name.split('_')[0]; writer.writerow([patient,image,mask,label,'pseudo','pending','test'])
    print({'images': len(output), 'manifest': str(manifest), 'mask_dir': str(mask_root)})

if __name__ == '__main__': main()
