"""Base MIC Regression Model training protocol (main text section 2.3;
optimiser protocol and freeze boundary from Supplementary Method 12).

Frozen protocol:

* continuous target log10(MIC); the loss is Huber (smooth L1 with
  delta = 1.0 log10 unit) - reconstruction decision RD-7;
* AdamW, weight decay 1e-4; learning rate 1e-5 for fine-tuned (pretrained)
  layers and 3e-4 for newly initialised layers; batch size 64; at most 100
  epochs; gradient clipping 1.0;
* model selection on validation RMSE; ties within 1e-4 are broken by the
  higher validation Spearman correlation (RD-13); early stopping after 12
  consecutive epochs without improvement;
* the production model is trained independently with 5 fixed seeds; Sbase is
  the arithmetic mean of the seed predictions (predicted log10(MIC)) and
  inter-model dispersion serves as the uncertainty indicator.

Prespecified MIC thresholds (<= 256 screening hit, <= 64 potent hit) define
experimental endpoints only; they never enter the training target (main text
section 2.3). Regression performance is assessed by MAE, RMSE, Spearman rank
correlation, observed-versus-predicted agreement and residual structure
(see ``bgc2activity.activity.evaluation``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

FINE_TUNED_LEARNING_RATE = 1e-5
NEW_LAYER_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
GRADIENT_CLIP_NORM = 1.0
SELECTION_TIE_TOLERANCE = 1e-4
EARLY_STOPPING_PATIENCE = 12
HUBER_DELTA = 1.0  # log10(MIC) units (RD-7)
ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)  # five fixed seeds (manuscript fixes the count)
DEFAULT_FINE_TUNED_PREFIXES = (
    "pretrainer.encoder.layers.2",
    "pretrainer.output_norm",
    "pooling",
)


def move_to_device(value, device: str | torch.device):
    """Recursively move tensors in nested model-input structures."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def huber_loss(predictions: torch.Tensor, targets: torch.Tensor,
               delta: float = HUBER_DELTA) -> torch.Tensor:
    """Smooth-L1 (Huber) loss on the log10(MIC) scale (RD-7)."""
    return nn.functional.smooth_l1_loss(predictions, targets, beta=delta)


def build_parameter_groups(
    model: nn.Module, fine_tuned_prefixes: tuple[str, ...] = DEFAULT_FINE_TUNED_PREFIXES,
) -> list[dict]:
    """AdamW parameter groups: pretrained-but-unfrozen parameters at 1e-5,
    newly initialised parameters at 3e-4.

    ``fine_tuned_prefixes`` are the name prefixes of parameters inherited from
    the pretraining checkpoint and kept trainable by the freeze boundary
    (layer 3, output LN and pooling of the reconstructed base model).
    """
    fine_tuned, newly_initialised = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in fine_tuned_prefixes):
            fine_tuned.append(parameter)
        else:
            newly_initialised.append(parameter)
    return [
        {"params": fine_tuned, "lr": FINE_TUNED_LEARNING_RATE},
        {"params": newly_initialised, "lr": NEW_LAYER_LEARNING_RATE},
    ]


@dataclass
class EpochRecord:
    epoch: int
    validation_rmse: float
    validation_spearman: float


def select_best_epoch(history: list[EpochRecord]) -> int:
    """Validation RMSE is primary (lower is better); ties within 1e-4 are
    broken by the higher validation Spearman (RD-13). Returns the epoch
    (1-based)."""
    best = None
    for record in history:
        if best is None:
            best = record
            continue
        if record.validation_rmse < best.validation_rmse - SELECTION_TIE_TOLERANCE:
            best = record
        elif (
            abs(record.validation_rmse - best.validation_rmse) <= SELECTION_TIE_TOLERANCE
            and record.validation_spearman > best.validation_spearman
        ):
            best = record
    return best.epoch


def train_one_seed(
    model: nn.Module,
    train_batches,
    evaluate,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    fine_tuned_prefixes: tuple[str, ...] = DEFAULT_FINE_TUNED_PREFIXES,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, list[EpochRecord]]:
    """Train one seed of the Base MIC Regression Model.

    ``train_batches()`` yields ``(batch, targets)`` with ``batch`` the keyword
    dictionary consumed by ``BaseActivityModel.forward`` and ``targets`` the
    log10(MIC) tensor. ``evaluate(model)`` returns
    ``(validation_rmse, validation_spearman)`` on the frozen validation split.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        build_parameter_groups(model, fine_tuned_prefixes), weight_decay=WEIGHT_DECAY,
    )
    history: list[EpochRecord] = []
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch, targets in train_batches():
            optimizer.zero_grad()
            predictions = model(**move_to_device(batch, device))
            loss = huber_loss(predictions, targets.to(predictions.device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_rmse, validation_spearman = evaluate(model)
        history.append(EpochRecord(epoch, float(validation_rmse), float(validation_spearman)))
        if select_best_epoch(history) == epoch:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def ensemble_sbase(predictions_by_seed: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Sbase = arithmetic mean of the five seed predicted log10(MIC) values;
    per-candidate dispersion is the uncertainty indicator."""
    stacked = torch.stack(predictions_by_seed, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)
