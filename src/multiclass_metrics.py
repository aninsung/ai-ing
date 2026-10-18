"""Ranking metrics for the 14 NIH disease labels."""
import numpy as np
from sklearn.metrics import roc_auc_score


def macro_binary_auroc(targets, probabilities):
    """Mean ROC AUC over disease columns that contain both classes.

    targets and probabilities are arrays of shape (images, diseases).
    A column with only one class is skipped; it is not scored as 0.5 or 1.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    scores = []
    for index in range(targets.shape[1]):
        labels = targets[:, index]
        if np.unique(labels).size < 2:
            scores.append(None)
            continue
        scores.append(float(roc_auc_score(labels, probabilities[:, index])))
    present = [score for score in scores if score is not None]
    macro = float(np.mean(present)) if present else None
    return macro, scores
