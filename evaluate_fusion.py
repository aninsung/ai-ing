"""Evaluate global+ROI fusion checkpoint on NIH validation/test splits."""
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from src.fusion_model import FusionResNet50
from src.multiclass_dataset import CLASSES,DISEASES

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--root',required=True); p.add_argument('--bbox-manifest',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 with Path(a.manifest).open(encoding='utf-8-sig',newline='') as f: allrows=list(csv.DictReader(f))
 masks={}
 with Path(a.bbox_manifest).open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f): masks[Path(r['image_path']).name]=r['mask_path']
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=FusionResNet50().to(dev); model.load_state_dict(torch.load(a.checkpoint,map_location=dev)['model']); model.eval(); result={}
 for split in ('val','test'):
  rows=[r for r in allrows if r['split']==split and r['primary_class_id']]; y=[]; pred=[]; probs=[]; aux_p=[]; aux_y=[]
  with torch.no_grad():
   for r in rows:
    with Image.open(Path(a.root)/r['image_path'].replace('\\','/')) as im: arr=np.asarray(im.convert('L').resize((224,224)),dtype=np.float32)/255
    roi=arr.copy(); mp=masks.get(Path(r['image_path']).name)
    if mp and Path(mp).is_file():
     with Image.open(mp) as mm: roi*=np.asarray(mm.convert('L').resize((224,224)),dtype=np.float32)/255
    x=torch.from_numpy(arr).float()[None,None].to(dev); z=torch.from_numpy(roi).float()[None,None].to(dev); logits,aux=model(x,z); y.append(int(r['primary_class_id'])); pred.append(int(logits.argmax(1))); probs.append(logits.softmax(1)[0].cpu().numpy()); aux_p.append(aux.sigmoid()[0].cpu().numpy()); aux_y.append([int(r['label_'+d]) for d in DISEASES])
  pmat=np.asarray(probs); metric={'samples':len(y),'accuracy':float(np.mean(np.asarray(y)==np.asarray(pred))),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred))}
  try: metric['macro_auroc']=float(roc_auc_score(np.eye(15)[y],pmat,multi_class='ovr',average='macro'))
  except ValueError: metric['macro_auroc']=None
  ap=np.asarray(aux_p); ay=np.asarray(aux_y); thresholds=[]
  for k in range(14):
   best_t,best_f=.5,-1
   for t in np.arange(.1,.91,.05):
    f=f1_score(ay[:,k],ap[:,k]>=t,zero_division=0)
    if f>best_f: best_t,best_f=float(t),float(f)
   thresholds.append(best_t)
  metric['auxiliary_thresholds']=thresholds; result[split]=metric
 Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
