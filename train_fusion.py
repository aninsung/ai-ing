"""Train global+ROI feature fusion on the full NIH multiclass manifest."""
import argparse,csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from src.multiclass_dataset import read_manifest,DISEASES
from src.fusion_model import FusionResNet50

class FusionDataset(Dataset):
 def __init__(self,rows,root,masks,size=224): self.rows=rows; self.root=Path(root); self.masks=masks; self.size=size
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; p=self.root/r['image_path'].replace('\\','/')
  with Image.open(p) as im: arr=np.asarray(im.convert('L').resize((self.size,self.size)),dtype=np.float32)/255
  roi=arr.copy(); mp=self.masks.get(Path(r['image_path']).name)
  if mp and Path(mp).is_file():
   with Image.open(mp) as mm: roi*=np.asarray(mm.convert('L').resize((self.size,self.size)),dtype=np.float32)/255
  return {'image':torch.from_numpy(arr).unsqueeze(0),'roi':torch.from_numpy(roi).unsqueeze(0),'primary':torch.tensor(int(r['primary_class_id']) if r['primary_class_id'] else -1),'auxiliary':torch.tensor([int(r['label_'+d]) for d in DISEASES],dtype=torch.float32)}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--root',required=True); p.add_argument('--bbox-manifest',required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--output',required=True); a=p.parse_args()
 rows=read_manifest(a.manifest,'train'); masks={}
 with Path(a.bbox_manifest).open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f): masks[Path(r['image_path']).name]=r['mask_path']
 dl=DataLoader(FusionDataset(rows,a.root,masks),64,shuffle=True,num_workers=0,pin_memory=True); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=FusionResNet50().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
 counts=torch.bincount(torch.tensor([int(r['primary_class_id']) for r in rows if r['primary_class_id']]),minlength=15).float().clamp_min(1); weights=(1.0/counts.sqrt()); weights=weights/weights.mean()
 for e in range(a.epochs):
  model.train(); total=0
  for b in dl:
   with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
    pz,az=model(b['image'].to(dev),b['roi'].to(dev)); loss=torch.nn.functional.cross_entropy(pz,b['primary'].to(dev),weight=weights.to(dev),ignore_index=-1)+.5*torch.nn.functional.binary_cross_entropy_with_logits(az,b['auxiliary'].to(dev))
   if not torch.isfinite(loss): continue
   opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()*len(b['image'])
  print({'epoch':e+1,'loss':total/len(rows)},flush=True)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'epochs':a.epochs},a.output)
if __name__=='__main__': main()
