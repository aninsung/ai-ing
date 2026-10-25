"""Report positive Dice separately from verified-negative false positives."""
import numpy as np


def binary_scores(prediction, target):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape:
        raise ValueError('Prediction/target shape mismatch')
    p, t = int(prediction.sum()), int(target.sum())
    if t == 0:
        return {'positive': False, 'dice': None, 'iou': None,
                'false_positive': p > 0, 'false_positive_pixels': p,
                'pixels': int(target.size)}
    intersection = int(np.logical_and(prediction, target).sum())
    return {'positive': True, 'dice': 2 * intersection / (p+t),
            'iou': intersection / (p+t-intersection)}


def summarize(scores):
    positives = [s for s in scores if s['positive']]
    negatives = [s for s in scores if not s['positive']]
    return {'positive_images': len(positives), 'negative_images': len(negatives),
            'positive_dice': float(np.mean([s['dice'] for s in positives])) if positives else None,
            'positive_iou': float(np.mean([s['iou'] for s in positives])) if positives else None,
            'negative_image_fp_rate': sum(s['false_positive'] for s in negatives) / len(negatives) if negatives else None,
            'negative_pixel_fp_rate': sum(s['false_positive_pixels'] for s in negatives) / sum(s['pixels'] for s in negatives) if negatives else None,
            'empty_mask_policy': 'Negative images excluded from Dice/IoU and reported as false positives'}
