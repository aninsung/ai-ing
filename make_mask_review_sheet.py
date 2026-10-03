"""Create image/mask overlays for quick pseudo-mask review."""
import argparse, csv, math
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True); p.add_argument('--limit',type=int,default=36); p.add_argument('--tile',type=int,default=224); a=p.parse_args()
    with Path(a.manifest).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))[:a.limit]
    cols=6; sheet=Image.new('RGB',(cols*a.tile,math.ceil(len(rows)/cols)*a.tile),'white')
    for i,row in enumerate(rows):
        with Image.open(row['image_path']) as im: image=im.convert('RGB').resize((a.tile,a.tile))
        with Image.open(row['mask_path']) as m: mask=m.convert('L').resize((a.tile,a.tile))
        overlay=image.copy(); red=Image.new('RGB',image.size,(255,0,0)); overlay=Image.composite(red,overlay,mask.point(lambda x:int(x*.45)))
        tile=Image.blend(image,overlay,.55); ImageDraw.Draw(tile).text((4,4),row['image_label'],fill='yellow')
        sheet.paste(tile,((i%cols)*a.tile,(i//cols)*a.tile))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); sheet.save(a.output); print({'images':len(rows),'output':str(a.output)})
if __name__=='__main__': main()
