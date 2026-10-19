"""Decision and ranking metrics for the NIH primary and auxiliary heads."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

RARE_CLASSES = ("Nodule", "Pneumonia", "Edema", "Fibrosis")


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


def primary_decision_metrics(targets, predictions, class_names, rare_classes=RARE_CLASSES):
    """Accuracy, macro-F1, balanced accuracy, and rare-class recall.

    Balanced accuracy and rare recall skip classes that have no examples.
    Macro-F1 keeps a fixed label set, so a missing class contributes 0.
    """
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    recalls = {}
    present = []
    for index, name in enumerate(class_names):
        actual = np.flatnonzero(targets == index)
        if actual.size == 0:
            recalls[name] = None
            continue
        recall = float(np.mean(predictions[actual] == index))
        recalls[name] = recall
        present.append(recall)
    rare = [recalls[name] for name in rare_classes if recalls.get(name) is not None]
    if targets.size == 0:
        accuracy = None
        macro_f1 = None
    else:
        accuracy = float(np.mean(targets == predictions))
        macro_f1 = float(f1_score(
            targets, predictions, average="macro",
            labels=list(range(len(class_names))), zero_division=0,
        ))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": float(np.mean(present)) if present else None,
        "rare_recall": float(np.mean(rare)) if rare else None,
        "class_recall": recalls,
    }
