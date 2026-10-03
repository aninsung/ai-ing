"""Fine-tune the multiclass classifier on BBox-derived ROI images."""
import argparse,csv
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from src.multiclass_model import MultiTaskResNet50
from src.multiclass_dataset import CLASSES

class RoiDataset(Dataset):
 def __init__(self,rows,size=224): self.rows=rows; self.size=size
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['image_path']) as im: x=im.convert('L').resize((self.size,self.size))
  with Image.open(r['mask_path']) as mm: m=mm.convert('L').resize((self.size,self.size))
  x=torch.from_numpy(np.asarray(x,dtype=np.float32)/255).unsqueeze(0); m=torch.from_numpy(np.asarray(m,dtype=np.float32)/255).unsqueeze(0)
  return x*m, CLASSES.index(r['image_label'])

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--epochs',type=int,default=5); p.add_argument('--output',required=True); a=p.parse_args()
 with Path(a.manifest).open(encoding='utf-8-sig',newline='') as f: rows=[r for r in csv.DictReader(f) if r['image_label'] in CLASSES]
 rows.sort(key=lambda r:r['patient_id']); cut=int(len(rows)*.85); train,val=rows[:cut],rows[cut:]
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(a.checkpoint,map_location=dev); model=MultiTaskResNet50(15,14,False).to(dev); model.load_state_dict(ck['model']); opt=torch.optim.AdamW(model.parameters(),lr=2e-5); best=0
 tr=DataLoader(RoiDataset(train),8,shuffle=True); va=DataLoader(RoiDataset(val),8)
 for e in range(a.epochs):
  model.train()
  for x,y in tr:
   z,_=model(x.to(dev)); loss=torch.nn.functional.cross_entropy(z,y.to(dev)); opt.zero_grad(); loss.backward(); opt.step()
  model.eval(); good=n=0
  with torch.no_grad():
   for x,y in va: good+=(model(x.to(dev))[0].argmax(1).cpu()==y).sum().item(); n+=len(y)
  acc=good/max(n,1); print({'epoch':e+1,'val_accuracy':acc},flush=True)
  if acc>best: best=acc; Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'val_accuracy':acc},a.output)
if __name__=='__main__': main()
