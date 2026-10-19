import torch
import torch.nn.functional as F


def asymmetric_bce(logits, targets, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
    """ASL-style auxiliary loss; targets are multi-hot and logits are raw."""
    probabilities = logits.sigmoid()
    if clip:
        probabilities = (probabilities + clip).clamp(max=1.0)
    positive = targets * torch.log(probabilities.clamp_min(1e-8))
    negative = (1 - targets) * torch.log((1 - probabilities).clamp_min(1e-8))
    weights = targets * (1 - probabilities).pow(gamma_pos) + (1 - targets) * probabilities.pow(gamma_neg)
    return -(weights * (positive + negative)).mean()


def effective_number_weights(counts, beta=0.9999):
    """Class-balanced weights from the effective number of samples.

    Rare classes receive a larger weight. beta is the usual CB hyperparameter
    and must lie strictly between 0 and 1.
    """
    if not 0.0 < float(beta) < 1.0:
        raise ValueError("beta must lie strictly between 0 and 1")
    counts = counts.detach().to(dtype=torch.float64).clamp_min(1)
    beta_tensor = torch.tensor(float(beta), dtype=torch.float64, device=counts.device)
    weights = (1.0 - float(beta)) / (1.0 - torch.pow(beta_tensor, counts))
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32)


def class_balanced_focal_loss(logits, targets, class_weights, gamma=2.0):
    """Multiclass focal loss scaled by a per-class weight."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    logits = logits.float()
    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    index = targets.unsqueeze(1)
    log_pt = log_probabilities.gather(1, index).squeeze(1)
    pt = probabilities.gather(1, index).squeeze(1)
    alpha = class_weights.to(device=logits.device, dtype=torch.float32)[targets]
    return -(alpha * (1.0 - pt).pow(gamma) * log_pt).mean()


def auxiliary_positive_weight(rows, diseases, clip=10.0):
    """neg/pos ratio per disease, clipped so rare labels cannot dominate the loss."""
    positive = torch.zeros(len(diseases))
    for row in rows:
        for index, name in enumerate(diseases):
            positive[index] += float(row[f"label_{name}"])
    negative = float(len(rows)) - positive
    return (negative / positive.clamp_min(1.0)).clamp(max=clip)


def multitask_loss(primary_logits, auxiliary_logits, primary, auxiliary, class_weights, auxiliary_weight,
                   auxiliary_loss_name="asymmetric_bce", pos_weight=None, primary_loss_name="cross_entropy",
                   focal_gamma=2.0):
    mask = primary >= 0
    if not mask.any():
        primary_loss = primary_logits.sum() * 0.0
    elif primary_loss_name == "cb_focal":
        primary_loss = class_balanced_focal_loss(
            primary_logits[mask], primary[mask], class_weights, gamma=focal_gamma)
    elif primary_loss_name == "cross_entropy":
        primary_loss = F.cross_entropy(primary_logits[mask], primary[mask], weight=class_weights)
    else:
        raise ValueError(f"Unsupported primary loss: {primary_loss_name}")
    if auxiliary_loss_name == "bce_pos_weight":
        auxiliary_loss = F.binary_cross_entropy_with_logits(auxiliary_logits, auxiliary, pos_weight=pos_weight)
    else:
        auxiliary_loss = asymmetric_bce(auxiliary_logits, auxiliary)
    return primary_loss + auxiliary_weight * auxiliary_loss, primary_loss.detach(), auxiliary_loss.detach()
