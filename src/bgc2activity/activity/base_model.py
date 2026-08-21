"""Base MIC Regression Model (main text section 2.3; freeze boundary and
feature fusion from Supplementary Method 12).

The Base MIC Regression Model takes the pretrained peptide encoder plus
predefined chemical features and outputs Sbase, the predicted log10(MIC)
(lower values denote greater predicted antibacterial activity). Supervised
training loads the full pretraining checkpoint and freezes the monomer
projections, positional encodings and the first two Transformer layers; only
the third layer, the output LayerNorm and the candidate pooling module are
fine-tuned. The pretraining chemical-recovery head does not enter the
downstream model; the scalar regression head and the chemical-feature fusion
layer are re-initialised.

Candidate-level continuous chemical features (length, molecular weight, net
charge at pH 7, GRAVY, aromatic/nonstandard monomer fractions, lipidation
descriptors) are imputed with training-set medians together with missingness
indicators, then standardised with training-set statistics. Categorical
states (N/C-terminal, lipidation, cyclization, each with an explicit unknown
level) use learnable embeddings. The fused chemistry vector is
32-dimensional.

The manuscript fixes the parameter census at 424,838 total parameters with
172,550 trainable and 252,288 frozen; exact internal padding is resolved by
the frozen configuration in ``configs/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bgc2activity.representation.pretraining import (
    HIDDEN_DIMENSION,
    CandidatePooling,
    ChemicalAwarePretrainer,
)

CHEMISTRY_DIMENSION = 32
CONTINUOUS_FEATURES = (
    "length",
    "molecular_weight",
    "net_charge_pH7",
    "GRAVY",
    "aromatic_fraction",
    "nonstandard_monomer_fraction",
    "lipidation_descriptor",
)
CATEGORICAL_LEVELS = {
    "n_terminal": 4,
    "c_terminal": 3,  # OH, NH2, unknown
    "lipidation": 3,  # none, explicit, unknown
    "cyclization": 3,  # none, cyclized, unknown
}
CATEGORICAL_EMBEDDING_DIMENSION = 4
MANUSCRIPT_TOTAL_PARAMETERS = 424_838
MANUSCRIPT_TRAINABLE_PARAMETERS = 172_550
MANUSCRIPT_FROZEN_PARAMETERS = 252_288


@dataclass(frozen=True)
class ChemistryFeatureSpec:
    medians: dict[str, float]
    means: dict[str, float]
    standard_deviations: dict[str, float]


class ChemistryEncoder(nn.Module):
    """Projects continuous + missingness indicators and categorical embeddings
    into the 32-dimensional chemistry representation."""

    def __init__(self) -> None:
        super().__init__()
        n_continuous_inputs = 2 * len(CONTINUOUS_FEATURES)
        categorical_input = len(CATEGORICAL_LEVELS) * CATEGORICAL_EMBEDDING_DIMENSION
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(levels, CATEGORICAL_EMBEDDING_DIMENSION) for name, levels in CATEGORICAL_LEVELS.items()}
        )
        self.projection = nn.Linear(n_continuous_inputs + categorical_input, CHEMISTRY_DIMENSION)

    def forward(self, continuous: torch.Tensor, missingness: torch.Tensor,
                categorical: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [torch.cat([continuous, missingness], dim=-1)]
        for name in CATEGORICAL_LEVELS:
            parts.append(self.embeddings[name](categorical[name]))
        return self.projection(torch.cat(parts, dim=-1))


class BaseActivityModel(nn.Module):
    """Base MIC Regression Model: scalar regression head over the fused
    candidate + chemistry representation, predicting log10(MIC)."""

    def __init__(self, pretrainer: ChemicalAwarePretrainer) -> None:
        super().__init__()
        self.pretrainer = pretrainer
        self.pooling = CandidatePooling()
        self.chemistry = ChemistryEncoder()
        self.regression_head = nn.Sequential(
            nn.Linear(HIDDEN_DIMENSION + CHEMISTRY_DIMENSION, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )
        # The pretraining chemical-recovery head does not enter the downstream model.
        self.pretrainer.recovery_head = nn.Identity()

    def freeze_for_supervised_training(self) -> None:
        """Freeze monomer projections, positional encodings and the first two
        Transformer layers; keep layer 3, output LN and pooling trainable."""
        for parameter in self.pretrainer.monomer_projection.parameters():
            parameter.requires_grad = False
        for parameter in self.pretrainer.input_fusion.parameters():
            parameter.requires_grad = False
        self.pretrainer.positional_encoding.requires_grad = False
        self.pretrainer.mask_token.requires_grad = False
        for layer_index, layer in enumerate(self.pretrainer.encoder.layers.layers):
            for parameter in layer.parameters():
                parameter.requires_grad = layer_index >= 2
        for parameter in self.pretrainer.output_norm.parameters():
            parameter.requires_grad = True

    def encode_candidate(self, text_vectors: torch.Tensor, descriptors: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.pretrainer.embed(text_vectors, descriptors, attention_mask)
        encoded = self.pretrainer.encoder(hidden, attention_mask)
        encoded = self.pretrainer.output_norm(encoded)
        return self.pooling(encoded, attention_mask)

    def forward(
        self,
        text_vectors: torch.Tensor,
        descriptors: torch.Tensor,
        attention_mask: torch.Tensor,
        continuous: torch.Tensor,
        missingness: torch.Tensor,
        categorical: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Returns Sbase, the predicted log10(MIC) (no probability transform)."""
        candidate_representation = self.encode_candidate(text_vectors, descriptors, attention_mask)
        chemistry_representation = self.chemistry(continuous, missingness, categorical)
        return self.regression_head(
            torch.cat([candidate_representation, chemistry_representation], dim=-1)
        ).squeeze(-1)


def encode_sequences_to_ensemble_sbase(models: list[BaseActivityModel], batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Average predicted log10(MIC) over independently trained seeds;
    dispersion is the uncertainty indicator (main text section 2.3)."""
    predictions = torch.stack([model(**batch) for model in models], dim=0)
    return predictions.mean(dim=0), predictions.std(dim=0, unbiased=False)
