"""Validation metrics for the 14-label head."""
import numpy as np


def average_precision(target, score):
    positive = int(target.sum())
    if positive == 0:
        return None
    order = np.argsort(-score)
    hits = np.cumsum(target[order])
    precision = hits / np.arange(1, len(target) + 1)
    return float(precision[target[order] == 1].mean())


def auroc(target, score):
    positive = int(target.sum())
    negative = len(target) - positive
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[target == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def macro(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def summarize(target, score, names):
    ap = [average_precision(target[:, i], score[:, i]) for i in range(target.shape[1])]
    auc = [auroc(target[:, i], score[:, i]) for i in range(target.shape[1])]
    return {
        "macro_auprc": macro(ap),
        "macro_auroc": macro(auc),
        "auprc": {name: value for name, value in zip(names, ap)},
    }
