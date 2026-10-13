"""Multitask training objective for the BGC detector.

L = L_det + lambda_cls * L_cls + lambda_evo * L_evo

with lambda_cls = 1.0 and lambda_evo = 0.5 (Supplementary Method 6.2).
L_det is position-wise binary cross-entropy computed only over valid (unpadded)
positions. L_cls is masked cross-entropy over product classes. L_evo is a
batch-hard triplet loss with margin 0.5 on the region-level evolutionary
representation; it is recorded as 0 for batches without valid anchors.
The MoE load-balancing loss (coefficient 0.01) is added separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

LAMBDA_CLASS = 1.0
LAMBDA_EVO = 0.5
TRIPLET_MARGIN = 0.5
LOAD_BALANCING_COEFFICIENT = 0.01


@dataclass(frozen=True)
class MultitaskLoss:
    total: torch.Tensor
    detection: torch.Tensor
    classification: torch.Tensor
    evolutionary: torch.Tensor
    load_balancing: torch.Tensor


def masked_detection_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="none")
    loss = loss * valid.to(logits.dtype)
    return loss.sum() / valid.to(logits.dtype).sum().clamp_min(1.0)


def classification_loss(class_logits: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(class_logits, class_labels)


def pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """Euclidean distance matrix over batch representations."""
    return torch.cdist(x, x, p=2.0)


def batch_hard_triplet_loss(representation: torch.Tensor, family_labels: torch.Tensor) -> torch.Tensor:
    """Batch-hard triplet loss with margin 0.5.

    Anchors require at least one positive and one negative partner in the batch.
    Returns 0 when no valid anchor exists (per manuscript specification).
    """
    distances = pairwise_distance_matrix(representation)
    same_family = family_labels.unsqueeze(0) == family_labels.unsqueeze(1)
    eye = torch.eye(same_family.shape[0], dtype=torch.bool, device=same_family.device)
    positive_mask = same_family & ~eye
    negative_mask = ~same_family
    losses = []
    for i in range(representation.shape[0]):
        if not bool(positive_mask[i].any()) or not bool(negative_mask[i].any()):
            continue
        hardest_positive = distances[i][positive_mask[i]].max()
        hardest_negative = distances[i][negative_mask[i]].min()
        losses.append(F.relu(hardest_positive - hardest_negative + TRIPLET_MARGIN))
    if not losses:
        return representation.new_zeros(())
    return torch.stack(losses).mean()


def multitask_loss(
    detection_logits: torch.Tensor,
    detection_labels: torch.Tensor,
    valid: torch.Tensor,
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    region_representation: torch.Tensor,
    family_labels: torch.Tensor,
    load_balancing_term: torch.Tensor,
    lambda_class: float = LAMBDA_CLASS,
    lambda_evo: float = LAMBDA_EVO,
    load_balancing_coefficient: float = LOAD_BALANCING_COEFFICIENT,
) -> MultitaskLoss:
    det = masked_detection_loss(detection_logits, detection_labels, valid)
    cls = classification_loss(class_logits, class_labels)
    evo = batch_hard_triplet_loss(region_representation, family_labels)
    total = det + lambda_class * cls + lambda_evo * evo + load_balancing_coefficient * load_balancing_term
    return MultitaskLoss(total=total, detection=det, classification=cls,
                         evolutionary=evo, load_balancing=load_balancing_term)
