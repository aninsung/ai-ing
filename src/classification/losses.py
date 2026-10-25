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


def multitask_loss(primary_logits, auxiliary_logits, primary, auxiliary, class_weights, auxiliary_weight):
    mask = primary >= 0
    if not mask.any():
        primary_loss = primary_logits.sum() * 0.0
    else:
        primary_loss = F.cross_entropy(primary_logits[mask], primary[mask], weight=class_weights)
    auxiliary_loss = asymmetric_bce(auxiliary_logits, auxiliary)
    return primary_loss + auxiliary_weight * auxiliary_loss, primary_loss.detach(), auxiliary_loss.detach()
