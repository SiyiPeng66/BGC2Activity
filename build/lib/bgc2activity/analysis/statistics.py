"""Statistical analysis, calibration assessment and reproducibility
(Supplementary Methods 24-25).

Implemented in code (Method 24):

* Wilson 95% confidence intervals for the two-round hit rates;
* risk difference with its confidence interval (primary cohort effect size)
  and Fisher's exact test (secondary);
* Spearman rank correlation with right-censored MIC handling (MIC > 256
  assigned the weakest rank in the primary analysis; 512 substitution in the
  sensitivity analysis);
* paired bootstrap confidence intervals for the correlation difference
  between the Base Model and the Calibrated Model, with candidate/BGC/
  core-sequence clusters as the resampling unit;
* probability reliability: Brier score, log loss, five-bin reliability curve
  and ECE.

GraphPad Prism 9 / IBM SPSS tests of Method 25 (normality screening, t-test,
ANOVA families, Mann-Whitney / Kruskal-Wallis) are not re-implemented; the
technical-replicate averaging rule that precedes them is provided here.
"""

from __future__ import annotations

import math

MIC_CENSORING_THRESHOLD = 256.0
MIC_SENSITIVITY_SUBSTITUTION = 512.0
RELIABILITY_BINS = 5
Z_95 = 1.959963984540054


def wilson_confidence_interval(n_successes: int, n_total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion (Method 24)."""
    if n_total <= 0 or not 0 <= n_successes <= n_total:
        raise ValueError("invalid binomial counts")
    centre = n_successes / n_total
    denominator = 1.0 + z * z / n_total
    centre_adjusted = centre + z * z / (2.0 * n_total)
    spread = z * math.sqrt(centre * (1.0 - centre) / n_total + z * z / (4.0 * n_total * n_total))
    lower = (centre_adjusted - spread) / denominator
    upper = (centre_adjusted + spread) / denominator
    return max(0.0, lower), min(1.0, upper)


def newcombe_risk_difference_interval(
    successes_a: int, total_a: int, successes_b: int, total_b: int, z: float = Z_95,
) -> tuple[float, float]:
    """Risk difference (p_a - p_b) with the Newcombe hybrid score confidence
    interval, the primary effect size for the two-cohort hit-rate comparison
    (Method 24)."""
    if total_a <= 0 or total_b <= 0:
        raise ValueError("both cohorts must be non-empty")
    p_a = successes_a / total_a
    p_b = successes_b / total_b

    def wilson_bounds(successes: int, total: int) -> tuple[float, float]:
        centre = successes / total
        denominator = 1.0 + z * z / total
        centre_adjusted = centre + z * z / (2.0 * total)
        spread = z * math.sqrt(centre * (1.0 - centre) / total + z * z / (4.0 * total * total))
        return (centre_adjusted - spread) / denominator, (centre_adjusted + spread) / denominator

    lower_a, upper_a = wilson_bounds(successes_a, total_a)
    lower_b, upper_b = wilson_bounds(successes_b, total_b)
    difference = p_a - p_b
    lower = difference - math.sqrt((p_a - lower_a) ** 2 + (upper_b - p_b) ** 2)
    upper = difference + math.sqrt((upper_a - p_a) ** 2 + (p_b - lower_b) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact test for a 2x2 table [[a, b], [c, d]]
    (secondary analysis of the cohort hit-rate difference, Method 24).

    The two-sided p-value sums all tables with the same margins whose
    hypergeometric probability does not exceed the observed table's.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("table entries must be non-negative")
    n = a + b + c + d
    row_1, column_1 = a + b, a + c

    def table_probability(x: int) -> float:
        return math.exp(
            math.lgamma(row_1 + 1) + math.lgamma(n - row_1 + 1) + math.lgamma(column_1 + 1)
            + math.lgamma(n - column_1 + 1) - math.lgamma(n + 1) - math.lgamma(x + 1)
            - math.lgamma(row_1 - x + 1) - math.lgamma(column_1 - x + 1)
            - math.lgamma(n - row_1 - column_1 + x + 1)
        )

    observed = table_probability(a)
    low = max(0, column_1 - (n - row_1))
    high = min(row_1, column_1)
    return min(1.0, sum(table_probability(x) for x in range(low, high + 1) if table_probability(x) <= observed + 1e-12))


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks (ties averaged) of a numeric list."""
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


def spearman_with_censoring(
    mic_values: list[float], scores: list[float], substitute_censored: bool = False,
) -> float:
    """Spearman correlation between model scores and MIC values (Method 24).

    Records with MIC > 256 are right-censored: the primary analysis assigns
    them the weakest rank (they sort last, ties averaged among themselves);
    the sensitivity analysis substitutes 512 before ranking.
    """
    if len(mic_values) != len(scores):
        raise ValueError("mic_values and scores must have the same length")
    if substitute_censored:
        adjusted = [MIC_SENSITIVITY_SUBSTITUTION if value > MIC_CENSORING_THRESHOLD else value for value in mic_values]
    else:
        capped = [value if value <= MIC_CENSORING_THRESHOLD else math.inf for value in mic_values]
        adjusted = capped
    mic_ranks = _ranks(adjusted)
    score_ranks = _ranks(scores)
    return _pearson(mic_ranks, score_ranks)


def _pearson(values_x: list[float], values_y: list[float]) -> float:
    n = len(values_x)
    mean_x = sum(values_x) / n
    mean_y = sum(values_y) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    variance_x = sum((x - mean_x) ** 2 for x in values_x)
    variance_y = sum((y - mean_y) ** 2 for y in values_y)
    if variance_x == 0.0 or variance_y == 0.0:
        return 0.0
    return covariance / math.sqrt(variance_x * variance_y)


def paired_bootstrap_delta_rho(
    scores_base: list[float],
    scores_calibrated: list[float],
    mic_values: list[float],
    cluster_ids: list[int],
    n_bootstrap: int = 2000,
    seed: int = 42,
    substitute_censored: bool = False,
) -> tuple[float, float]:
    """Paired bootstrap 95% CI for delta-rho = rho(Calibrated) - rho(Base)
    (Method 24).

    Whole clusters (candidate, BGC or core-sequence clusters for
    highly-correlated candidates) are the resampling unit; the point estimate
    is returned alongside the percentile interval.
    """
    import random

    if not (len(scores_base) == len(scores_calibrated) == len(mic_values) == len(cluster_ids)):
        raise ValueError("all inputs must have the same length")
    members: dict[int, list[int]] = {}
    for index, cluster in enumerate(cluster_ids):
        members.setdefault(cluster, []).append(index)
    cluster_keys = sorted(members)

    def delta_rho(indices: list[int]) -> float:
        mics = [mic_values[i] for i in indices]
        rho_base = spearman_with_censoring(mics, [scores_base[i] for i in indices], substitute_censored)
        rho_calibrated = spearman_with_censoring(mics, [scores_calibrated[i] for i in indices], substitute_censored)
        return rho_calibrated - rho_base

    observed = delta_rho(list(range(len(mic_values))))
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(n_bootstrap):
        sampled_indices: list[int] = []
        for _ in cluster_keys:
            sampled_indices.extend(members[rng.choice(cluster_keys)])
        bootstrap.append(delta_rho(sampled_indices))
    bootstrap.sort()
    lower = bootstrap[int(0.025 * len(bootstrap))]
    upper = bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))]
    return lower, upper


def brier_score(probabilities: list[float], labels: list[int]) -> float:
    if not probabilities:
        raise ValueError("empty input")
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities)


def log_loss(probabilities: list[float], labels: list[int], epsilon: float = 1e-15) -> float:
    if not probabilities:
        raise ValueError("empty input")
    total = 0.0
    for probability, label in zip(probabilities, labels):
        clipped = min(max(probability, epsilon), 1.0 - epsilon)
        total += -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))
    return total / len(probabilities)


def reliability_curve(probabilities: list[float], labels: list[int], n_bins: int = RELIABILITY_BINS) -> list[dict[str, float]]:
    """Five-bin reliability curve (Method 24): per-bin mean predicted
    probability, observed frequency and sample count."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    bin_labels: list[list[int]] = [[] for _ in range(n_bins)]
    for probability, label in zip(probabilities, labels):
        index = min(int(probability * n_bins), n_bins - 1)
        bins[index].append(probability)
        bin_labels[index].append(label)
    curve = []
    for index in range(n_bins):
        count = len(bins[index])
        curve.append(
            {
                "bin": index,
                "mean_predicted": sum(bins[index]) / count if count else float("nan"),
                "observed_frequency": sum(bin_labels[index]) / count if count else float("nan"),
                "count": count,
            }
        )
    return curve


def expected_calibration_error(probabilities: list[float], labels: list[int], n_bins: int = RELIABILITY_BINS) -> float:
    """ECE with equal-width bins (Method 24)."""
    curve = reliability_curve(probabilities, labels, n_bins)
    total = sum(entry["count"] for entry in curve)
    if total == 0:
        raise ValueError("empty input")
    return sum(
        entry["count"] / total * abs(entry["observed_frequency"] - entry["mean_predicted"])
        for entry in curve
        if entry["count"]
    )


def average_technical_replicates(replicates_by_experiment: dict[str, list[float]]) -> dict[str, float]:
    """Technical replicates are averaged within each experiment before
    between-group statistics (Supplementary Method 25)."""
    averaged = {}
    for experiment, replicates in replicates_by_experiment.items():
        if not replicates:
            raise ValueError(f"experiment {experiment} has no replicates")
        averaged[experiment] = sum(replicates) / len(replicates)
    return averaged
