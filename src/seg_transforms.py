"""Joint, aspect-preserving image/mask transforms and coordinate restoration."""
import numpy as np
from PIL import Image
import torch


def letterbox(image, mask, size, augment=False):
    width, height = image.size
    scale = size / max(width, height)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - rw) // 2, (size - rh) // 2
    im = np.asarray(image.resize((rw, rh), Image.Resampling.BILINEAR), dtype=np.float32).copy()
    lo, hi = float(im.min()), float(im.max())
    im = (im - lo) / (hi - lo) if hi > lo else np.zeros_like(im)
    ma = np.asarray(mask.resize((rw, rh), Image.Resampling.NEAREST), dtype=np.float32).copy()
    ma = (ma > 0).astype(np.float32)
    out_im, out_ma = np.zeros((size, size), np.float32), np.zeros((size, size), np.float32)
    out_im[top:top+rh, left:left+rw] = im
    out_ma[top:top+rh, left:left+rw] = ma
    if augment and torch.rand(()).item() < .5:
        out_im, out_ma = out_im[:, ::-1].copy(), out_ma[:, ::-1].copy()
    geometry = torch.tensor([width, height, left, top, rw, rh], dtype=torch.long)
    return torch.from_numpy(out_im.copy()).unsqueeze(0), torch.from_numpy(out_ma.copy()).unsqueeze(0), geometry


def restore_probability(probability, geometry):
    width, height, left, top, rw, rh = map(int, geometry)
    crop = probability[..., top:top+rh, left:left+rw]
    return torch.nn.functional.interpolate(crop.reshape(1, 1, rh, rw),
                                          size=(height, width), mode='bilinear',
                                          align_corners=False)[0, 0]
