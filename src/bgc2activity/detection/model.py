"""Multitask BGC detection model (final-manuscript specification).

Implements Supplementary Method 6 of the final manuscript: a shared encoder of
four stacked Time-mixing / Channel-mixing blocks over 102-dimensional frozen
pfam2vec inputs projected to 128 dimensions, followed by an eight-expert
mixture-of-experts module and three output heads (per-position detection,
auxiliary product class, evolutionary contrastive representation).

The Time-mixing sublayer uses causal linear attention along sequence positions;
the Channel-mixing sublayer is a two-layer FFN with expansion ratio 4 and GELU
activation. Every sublayer uses pre-layer-norm with residual connections and
dropout 0.15. The MoE uses top-2 sparse routing with a load-balancing loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PFAM2VEC_DIMENSION = 102
HIDDEN_DIMENSION = 128
N_LAYERS = 4
DROPOUT = 0.15
FFN_EXPANSION = 4
MOE_EXPERTS = 8
MOE_EXPERT_HIDDEN = 256
MOE_LOAD_BALANCING_COEFFICIENT = 0.01
DETECTION_THRESHOLD = 0.45
N_PRODUCT_CLASSES = 5


@dataclass(frozen=True)
class DetectorOutput:
    detection_logits: torch.Tensor  # (batch, positions)
    class_logits: torch.Tensor  # (batch, n_classes)
    region_representation: torch.Tensor  # (batch, hidden)
    router_probabilities: torch.Tensor  # (batch, positions, n_experts)
    expert_fraction: torch.Tensor  # (n_experts,)


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class CausalLinearAttention(nn.Module):
    """Causal linear attention used by the Time-mixing sublayer."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = _elu_feature_map(self.query(x))
        k = _elu_feature_map(self.key(x))
        v = self.value(x)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        k = k * mask
        v = v * mask
        numerator = torch.cumsum(torch.einsum("bti,btj->btij", k, v), dim=1)
        denominator = torch.cumsum(k, dim=1)
        out = torch.einsum("btij,bti->btj", numerator, q)
        normalizer = torch.einsum("bti,bti->bt", denominator, q).unsqueeze(-1)
        return out / normalizer.clamp_min(1e-6)


class TimeMixingSublayer(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.attention = CausalLinearAttention(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.attention(self.norm(x), attention_mask))


class ChannelMixingSublayer(nn.Module):
    def __init__(self, dimension: int, expansion: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * expansion),
            nn.GELU(),
            nn.Linear(dimension * expansion, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.ffn(self.norm(x)))
        return (x + out) * attention_mask.unsqueeze(-1).to(x.dtype)


class MixtureOfExperts(nn.Module):
    """Eight-expert MoE with top-2 sparse routing and load-balancing loss."""

    def __init__(self, dimension: int, n_experts: int, expert_hidden: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dimension, expert_hidden),
                nn.GELU(),
                nn.Linear(expert_hidden, dimension),
            )
            for _ in range(n_experts)
        )
        self.router = nn.Linear(dimension, n_experts)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        router_logits = self.router(x)
        router_probabilities = torch.softmax(router_logits, dim=-1)
        top2_weights, top2_indices = router_probabilities.topk(2, dim=-1)
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x)
        for slot in range(2):
            for expert_index in range(self.n_experts):
                selected = top2_indices[..., slot] == expert_index
                if not bool(selected.any()):
                    continue
                expert_input = x[selected]
                out[selected] = out[selected] + top2_weights[..., slot][selected].unsqueeze(-1) * self.experts[expert_index](expert_input)
        valid = attention_mask.bool()
        dispatched = top2_indices[valid]
        expert_fraction = torch.zeros(self.n_experts, device=x.device)
        if dispatched.numel():
            expert_fraction = torch.bincount(dispatched.reshape(-1), minlength=self.n_experts).to(x.dtype)
            expert_fraction = expert_fraction / expert_fraction.sum().clamp_min(1.0)
        return out, router_probabilities, expert_fraction


class MultitaskBGCDetector(nn.Module):
    """Shared-encoder multi-head BGC detector per Supplementary Method 6."""

    def __init__(
        self,
        pfam2vec_dimension: int = PFAM2VEC_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        ffn_expansion: int = FFN_EXPANSION,
        n_experts: int = MOE_EXPERTS,
        expert_hidden: int = MOE_EXPERT_HIDDEN,
        n_classes: int = N_PRODUCT_CLASSES,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(pfam2vec_dimension, hidden_dimension)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "time_mixing": TimeMixingSublayer(hidden_dimension, dropout),
                    "channel_mixing": ChannelMixingSublayer(hidden_dimension, ffn_expansion, dropout),
                }
            )
            for _ in range(n_layers)
        )
        self.moe = MixtureOfExperts(hidden_dimension, n_experts, expert_hidden)
        self.detection_head = nn.Linear(hidden_dimension, 1)
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Linear(hidden_dimension // 2, n_classes),
        )

    def encode(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.input_projection(embeddings) * attention_mask.unsqueeze(-1).to(embeddings.dtype)
        for layer in self.layers:
            hidden = layer["time_mixing"](hidden, attention_mask)
            hidden = layer["channel_mixing"](hidden, attention_mask)
        moe_out, router_probabilities, expert_fraction = self.moe(hidden, attention_mask)
        return moe_out, router_probabilities, expert_fraction

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> DetectorOutput:
        encoded, router_probabilities, expert_fraction = self.encode(embeddings, attention_mask)
        detection_logits = self.detection_head(encoded).squeeze(-1)
        pooled = masked_mean_pool(encoded, attention_mask)
        class_logits = self.class_head(pooled)
        return DetectorOutput(
            detection_logits=detection_logits,
            class_logits=class_logits,
            region_representation=pooled,
            router_probabilities=router_probabilities,
            expert_fraction=expert_fraction,
        )


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def load_balancing_loss(router_probabilities: torch.Tensor, expert_fraction: torch.Tensor) -> torch.Tensor:
    """Switch-style auxiliary loss (coefficient applied outside)."""
    mean_probability = router_probabilities.mean(dim=tuple(range(router_probabilities.ndim - 1)))
    return float(router_probabilities.shape[-1]) * torch.sum(expert_fraction * mean_probability)
