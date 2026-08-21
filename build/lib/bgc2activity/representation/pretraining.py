"""Multimodal chemical-aware peptide pretraining (Supplementary Method 10).

Each monomer carries a chemical-text signature (BLAKE2b-hashed to a
24-dimensional L2-normalised vector) and an 8-dimensional numeric descriptor
vector (hydrophobicity, charge, polarity, aromaticity, mass, configuration,
lipid-chain carbon count, cyclic indicator). The two are each projected to 96
dimensions and concatenated. Learnable positional encodings (max_len = 64) are
added, and a 3-layer bidirectional Transformer (hidden 96, 4 heads, FFN 384,
GELU, pre-LN, dropout 0.10) predicts the 32-dimensional chemical target at
masked positions.

Masking samples 30% of valid positions (at least one per sequence); 80% of the
selected positions are replaced by a mask token, 10% by a random substitution
and 10% are left unchanged. The objective is SmoothL1 + 0.20 * cosine distance
at the selected positions. Training uses AdamW for 30 epochs (batch 128,
lr 8e-4, weight decay 1e-3, gradient clipping 1.0, seed 20260802).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

TEXT_VECTOR_DIMENSION = 24
DESCRIPTOR_DIMENSION = 8
CHEMICAL_TARGET_DIMENSION = TEXT_VECTOR_DIMENSION + DESCRIPTOR_DIMENSION
MONOMER_PROJECTION_DIMENSION = 96
HIDDEN_DIMENSION = 96
N_HEADS = 4
FFN_DIMENSION = 384
N_LAYERS = 3
DROPOUT = 0.10
MAX_LENGTH = 64
MASK_FRACTION = 0.30
COSINE_LOSS_WEIGHT = 0.20
PRETRAINING_SEED = 20260802
PRETRAINING_EPOCHS = 30
PRETRAINING_BATCH_SIZE = 128
PRETRAINING_LEARNING_RATE = 8e-4
PRETRAINING_WEIGHT_DECAY = 1e-3


def blake2b_text_vector(text: str, dimension: int = TEXT_VECTOR_DIMENSION) -> torch.Tensor:
    """Deterministic L2-normalised chemical-text signature via BLAKE2b hashing.

    Bytes are drawn from counter-mode BLAKE2b digests (BLAKE2b caps a single
    digest at 64 bytes), which keeps the mapping exact for any dimension.
    """
    raw = bytearray()
    counter = 0
    while len(raw) < dimension * 4:
        raw.extend(hashlib.blake2b(f"{counter}:{text}".encode("utf-8"), digest_size=64).digest())
        counter += 1
    values = torch.tensor(
        [int.from_bytes(raw[i * 4:(i + 1) * 4], "little") / 2**32 - 1.0 for i in range(dimension)],
        dtype=torch.float32,
    )
    norm = values.norm()
    if norm == 0:
        return values
    return values / norm


class MonomerInputProjection(nn.Module):
    def __init__(
        self,
        text_dimension: int = TEXT_VECTOR_DIMENSION,
        descriptor_dimension: int = DESCRIPTOR_DIMENSION,
        projection_dimension: int = MONOMER_PROJECTION_DIMENSION,
    ) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_dimension, projection_dimension)
        self.descriptor_projection = nn.Linear(descriptor_dimension, projection_dimension)

    def forward(self, text_vectors: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.text_projection(text_vectors), self.descriptor_projection(descriptors)], dim=-1)


class PretrainingTransformer(nn.Module):
    """3-layer bidirectional Transformer encoder with pre-layer-norm."""

    def __init__(
        self,
        hidden_dimension: int = HIDDEN_DIMENSION,
        n_heads: int = N_HEADS,
        ffn_dimension: int = FFN_DIMENSION,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=n_heads,
            dim_feedforward=ffn_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~attention_mask.bool()
        return self.layers(x, src_key_padding_mask=key_padding_mask)


class ChemicalAwarePretrainer(nn.Module):
    """Masked chemical-context pretraining model."""

    def __init__(self) -> None:
        super().__init__()
        self.monomer_projection = MonomerInputProjection()
        self.input_fusion = nn.Linear(2 * MONOMER_PROJECTION_DIMENSION, HIDDEN_DIMENSION)
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_LENGTH, HIDDEN_DIMENSION))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIMENSION))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.encoder = PretrainingTransformer()
        self.output_norm = nn.LayerNorm(HIDDEN_DIMENSION)
        self.recovery_head = nn.Linear(HIDDEN_DIMENSION, CHEMICAL_TARGET_DIMENSION)

    def embed(self, text_vectors: torch.Tensor, descriptors: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        fused = self.input_fusion(self.monomer_projection(text_vectors, descriptors))
        length = fused.shape[1]
        return fused + self.positional_encoding[:, :length, :]

    def forward(self, text_vectors: torch.Tensor, descriptors: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(text_vectors, descriptors, attention_mask)
        encoded = self.encoder(hidden, attention_mask)
        return self.recovery_head(self.output_norm(encoded))


def apply_masking(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token: torch.Tensor,
    mask_fraction: float = MASK_FRACTION,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ~30% of valid positions (>=1 per sequence); 80/10/10 scheme.

    Returns (masked_hidden, selection_mask).
    """
    batch_size, length, _ = hidden.shape
    uniform = torch.rand(batch_size, length, generator=generator, device=hidden.device)
    valid = attention_mask.bool()
    selected = valid & (uniform < mask_fraction)
    for row in range(batch_size):
        if valid[row].any() and not selected[row].any():
            valid_positions = valid[row].nonzero(as_tuple=True)[0]
            pick = int(torch.randint(0, len(valid_positions), (1,), generator=generator).item())
            selected[row, valid_positions[pick]] = True
    scheme = torch.rand(batch_size, length, generator=generator, device=hidden.device)
    out = hidden.clone()
    replace_mask = selected & (scheme < 0.8)
    substitute = selected & (scheme >= 0.8) & (scheme < 0.9)
    out[replace_mask] = mask_token.reshape(1, 1, -1).expand_as(out)[replace_mask]
    if substitute.any():
        donor = torch.randperm(batch_size * length, generator=generator, device=hidden.device)
        flat = out.reshape(batch_size * length, -1)
        flat[substitute.reshape(-1)] = flat[donor][: int(substitute.sum())]
        out = flat.reshape(batch_size, length, -1)
    return out, selected


def pretraining_loss(prediction: torch.Tensor, target: torch.Tensor, selection: torch.Tensor) -> dict[str, torch.Tensor]:
    smooth_l1 = F.smooth_l1_loss(prediction[selection], target[selection], reduction="mean")
    cosine = (1.0 - F.cosine_similarity(prediction[selection], target[selection], dim=-1)).mean()
    total = smooth_l1 + COSINE_LOSS_WEIGHT * cosine
    return {"total": total, "smooth_l1": smooth_l1, "cosine": cosine}


class CandidatePooling(nn.Module):
    """Attention-weighted mean, weighted std and per-dimension maximum pooling.

    Concatenates the three statistics, then projects with linear + GELU + LN to
    a 96-dimensional candidate representation (Supplementary Method 12).
    """

    def __init__(self, hidden_dimension: int = HIDDEN_DIMENSION, output_dimension: int = 96) -> None:
        super().__init__()
        self.attention_query = nn.Linear(hidden_dimension, 1)
        self.projection = nn.Linear(3 * hidden_dimension, output_dimension)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(output_dimension)

    def forward(self, encoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.attention_query(encoded).squeeze(-1)
        logits = logits.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = weights.masked_fill(~attention_mask.bool(), 0.0)
        mean = torch.einsum("bt,btd->bd", weights, encoded)
        variance = torch.einsum("bt,btd->bd", weights, (encoded - mean.unsqueeze(1)) ** 2)
        std = variance.clamp_min(1e-12).sqrt()
        maximum = encoded.masked_fill(~attention_mask.bool().unsqueeze(-1), float("-inf")).max(dim=1).values
        return self.norm(self.activation(self.projection(torch.cat([mean, std, maximum], dim=-1))))
