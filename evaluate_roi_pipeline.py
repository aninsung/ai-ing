"""Evaluate full-image versus BBox pseudo-mask ROI classification."""
import argparse,csv,json
from pathlib import Path
import torch
from PIL import Image
from src.multiclass_model import MultiTaskResNet50
from src.multiclass_dataset import DISEASES, CLASSES

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 with Path(a.manifest).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(a.checkpoint,map_location=dev); model=MultiTaskResNet50(15,14,False).to(dev); model.load_state_dict(ck['model']); model.eval(); full=roi=total=0
 with torch.no_grad():
  for r in rows:
   with Image.open(r['image_path']) as im: image=im.convert('L').resize((224,224))
   with Image.open(r['mask_path']) as mm: mask=mm.convert('L').resize((224,224))
   x=torch.from_numpy(__import__('numpy').asarray(image,dtype='float32')/255).unsqueeze(0).unsqueeze(0).to(dev); m=torch.from_numpy(__import__('numpy').asarray(mask,dtype='float32')/255).unsqueeze(0).unsqueeze(0).to(dev); y=CLASSES.index(r['image_label']) if r['image_label'] in CLASSES else -1
   if y < 0: continue
   lf,_=model(x); lr,_=model(x*m); full+=int(lf.argmax(1).item()==y); roi+=int(lr.argmax(1).item()==y); total+=1
 result={'samples':total,'full_accuracy':full/max(total,1),'roi_accuracy':roi/max(total,1),'note':'BBox-derived weak pseudo-mask; not pixel-ground-truth'}; Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
