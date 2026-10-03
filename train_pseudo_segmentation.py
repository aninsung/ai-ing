"""Weak segmentation smoke training using NIH BBox-derived pseudo-masks."""
import argparse, csv, hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.seg_dataset import SegDataset
from src.seg_network import build_model

def split_rows(rows):
    out = {'train': [], 'val': [], 'test': []}
    for row in rows:
        value = int(hashlib.sha256(row['patient_id'].encode()).hexdigest()[:8], 16) / 0xffffffff
        out['train' if value < .7 else 'val' if value < .85 else 'test'].append(row)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest', required=True); p.add_argument('--epochs',type=int,default=5); p.add_argument('--image-size',type=int,default=224); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--output',default='checkpoints/pseudo_unetpp.pth'); a=p.parse_args()
    with Path(a.manifest).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    groups=split_rows(rows); train=DataLoader(SegDataset(groups['train'],a.image_size,True),a.batch_size,shuffle=True); val=DataLoader(SegDataset(groups['val'],a.image_size),a.batch_size)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=build_model().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4); best=-1
    for epoch in range(a.epochs):
        model.train(); total=0
        for b in train:
            x,y=b['image'].to(device),b['mask'].to(device); z=model(x); prob=z.sigmoid(); inter=(prob*y).sum((1,2,3)); loss=(1-(2*inter+1e-6)/(prob.sum((1,2,3))+y.sum((1,2,3))+1e-6)).mean()+.3*torch.nn.functional.binary_cross_entropy_with_logits(z,y); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(x)
        model.eval(); score=n=0
        with torch.no_grad():
            for b in val:
                pr=model(b['image'].to(device)).sigmoid()>.5; gt=b['mask'].to(device)>.5; inter=(pr&gt).sum((1,2,3)).float(); score+=((2*inter+1e-6)/(pr.sum((1,2,3))+gt.sum((1,2,3))+1e-6)).sum().item(); n+=len(pr)
        dice=score/max(n,1); print({'epoch':epoch+1,'train_loss':total/max(len(groups['train']),1),'pseudo_val_dice':dice,'train':len(groups['train']),'val':len(groups['val'])},flush=True)
        if dice>best: best=dice; Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'epoch':epoch+1,'pseudo_val_dice':dice},a.output)
if __name__=='__main__': main()
