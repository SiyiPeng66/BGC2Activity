"""Regression evaluation for the Base MIC Regression Model.

The main text (section 2.3) assesses regression performance by MAE, RMSE,
Spearman rank correlation, observed-versus-predicted agreement and residual
structure on structural-cluster-disjoint partitions (Supplementary Figs 7-8).
All functions are dependency-free and operate on plain sequences of floats.
"""

from __future__ import annotations

import math
from typing import Sequence


def _as_floats(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def mean_absolute_error(observed: Sequence[float], predicted: Sequence[float]) -> float:
    observed, predicted = _as_floats(observed), _as_floats(predicted)
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted must be equal-length non-empty sequences")
    return sum(abs(o - p) for o, p in zip(observed, predicted)) / len(observed)


def root_mean_squared_error(observed: Sequence[float], predicted: Sequence[float]) -> float:
    observed, predicted = _as_floats(observed), _as_floats(predicted)
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted must be equal-length non-empty sequences")
    return math.sqrt(sum((o - p) ** 2 for o, p in zip(observed, predicted)) / len(observed))


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        block_end = index
        while block_end + 1 < len(order) and values[order[block_end + 1]] == values[order[index]]:
            block_end += 1
        average_rank = (index + block_end) / 2.0 + 1.0
        for position in range(index, block_end + 1):
            ranks[order[position]] = average_rank
        index = block_end + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    variance_x = sum((a - mean_x) ** 2 for a in x)
    variance_y = sum((b - mean_y) ** 2 for b in y)
    if variance_x == 0.0 or variance_y == 0.0:
        return 0.0
    return covariance / math.sqrt(variance_x * variance_y)


def spearman(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Spearman rank correlation between observed and predicted log10(MIC)."""
    observed, predicted = _as_floats(observed), _as_floats(predicted)
    if len(observed) != len(predicted) or len(observed) < 2:
        raise ValueError("spearman requires equal-length sequences with >= 2 elements")
    return _pearson(_ranks(observed), _ranks(predicted))


def observed_predicted_agreement(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Least-squares fit observed = intercept + slope * predicted plus Pearson
    correlation (the observed-versus-predicted agreement panel of
    Supplementary Fig 7; identity corresponds to slope 1, intercept 0)."""
    observed, predicted = _as_floats(observed), _as_floats(predicted)
    n = len(observed)
    if n != len(predicted) or n < 2:
        raise ValueError("agreement requires equal-length sequences with >= 2 elements")
    mean_o = sum(observed) / n
    mean_p = sum(predicted) / n
    s_pp = sum((p - mean_p) ** 2 for p in predicted)
    s_op = sum((o - mean_o) * (p - mean_p) for o, p in zip(observed, predicted))
    slope = s_op / s_pp if s_pp > 0 else 0.0
    intercept = mean_o - slope * mean_p
    return {"slope": slope, "intercept": intercept, "pearson": _pearson(observed, predicted)}


def residual_structure(observed: Sequence[float], predicted: Sequence[float],
                       n_bins: int = 5) -> list[dict]:
    """Mean residual (observed - predicted) within quantile bins of the
    observed log10(MIC) (residual-structure panel of Supplementary Fig 7)."""
    observed, predicted = _as_floats(observed), _as_floats(predicted)
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted must be equal-length non-empty sequences")
    order = sorted(range(len(observed)), key=lambda index: observed[index])
    bins: list[dict] = []
    for start in range(0, len(order), max(1, len(order) // n_bins)):
        block = order[start:start + max(1, len(order) // n_bins)]
        if not block:
            continue
        residuals = [observed[i] - predicted[i] for i in block]
        bins.append({
            "observed_low": min(observed[i] for i in block),
            "observed_high": max(observed[i] for i in block),
            "mean_residual": sum(residuals) / len(residuals),
            "n": len(block),
        })
    return bins


def regression_report(observed: Sequence[float], predicted: Sequence[float]) -> dict:
    """The full main-text regression metric set for one frozen partition."""
    return {
        "mae": mean_absolute_error(observed, predicted),
        "rmse": root_mean_squared_error(observed, predicted),
        "spearman": spearman(observed, predicted),
        "agreement": observed_predicted_agreement(observed, predicted),
        "residual_bins": residual_structure(observed, predicted),
    }
