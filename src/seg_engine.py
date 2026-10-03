import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .seg_transforms import restore_probability
from .seg_metrics import binary_scores, summarize


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_model(model, loader, device, threshold, output=None):
    model.eval()
    scores = []
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
    for batch in loader:
        probabilities = model(batch['image'].to(device)).sigmoid().cpu()
        for i, index in enumerate(batch['index'].tolist()):
            probability = restore_probability(probabilities[i, 0], batch['geometry'][i]).numpy()
            original, mask = loader.dataset.original(index)
            target = np.asarray(mask) > 0
            prediction = probability >= threshold
            score = binary_scores(prediction, target)
            score['image_path'] = loader.dataset.rows[index]['image_path']
            scores.append(score)
            if output is not None:
                name = f'{index:06d}'
                np.save(Path(output) / f'{name}_probability.npy', probability)
                Image.fromarray(prediction.astype(np.uint8)*255).save(Path(output) / f'{name}_prediction.png')
                base = np.asarray(original.convert('RGB')).copy()
                gt_view, pred_view = base.copy(), base.copy()
                gt_view[target] = (gt_view[target]*.5 + np.array([0,255,0])*.5).astype(np.uint8)
                pred_view[prediction] = (pred_view[prediction]*.5 + np.array([255,0,0])*.5).astype(np.uint8)
                Image.fromarray(np.concatenate([base, gt_view, pred_view], axis=1)).save(Path(output) / f'{name}_overlay.png')
    return summarize(scores), scores
