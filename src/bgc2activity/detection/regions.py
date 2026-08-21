"""Windowed inference aggregation and candidate region generation.

Implements Supplementary Methods 6.3-6.4 of the final manuscript:

- Inference uses overlapping windows of length 256 with stride 128; multi-window
  probabilities at the same position are averaged arithmetically before
  thresholding.
- Candidate regions receive five-class probabilities from the multitask module;
  for regions exceeding 256 positions, window class probabilities are averaged
  with weights proportional to the number of valid positions in each window.
- The gene-level score is the maximum position probability over the gene's Pfam
  domains; genes without Pfam hits score 0. A gene containing at least one
  position with probability >= 0.45 is a positive gene.
- Adjacent positive genes separated by at most two negative or Pfam-less genes
  are joined into initial regions; initial regions require >= 3 positive genes
  and >= 5 predicted genes, are extended by two genes on each side, and
  overlapping or closely adjacent (gap <= 3 genes) regions are merged, with the
  minimum requirements re-validated after merging.
- A region is NRPS-positive when NRPS is the highest-probability product class
  and at least one NRPS signature Pfam domain is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WINDOW_LENGTH = 256
WINDOW_STRIDE = 128
POSITION_THRESHOLD = 0.45
MAX_GAP_GENES = 2
MIN_POSITIVE_GENES = 3
MIN_PREDICTED_GENES = 5
EXTENSION_GENES = 2
MERGE_GAP_GENES = 3
NRPS_SIGNATURE_PFAMS = frozenset({"PF00668", "PF00501", "PF00550", "PF13193", "PF00975", "PF13745"})
PRODUCT_CLASSES = ("NRPS", "PKS", "RiPP", "Terpene", "Other")


def strip_pfam_version(pfam_id: str) -> str:
    return pfam_id.split(".", 1)[0]


def window_starts(n_positions: int, window_length: int = WINDOW_LENGTH, stride: int = WINDOW_STRIDE) -> list[int]:
    if n_positions <= window_length:
        return [0]
    starts = list(range(0, n_positions - window_length, stride))
    if starts[-1] != n_positions - window_length:
        starts.append(n_positions - window_length)
    return starts


def aggregate_window_probabilities(window_probabilities: list[tuple[int, np.ndarray]], n_positions: int) -> np.ndarray:
    """Average arithmetic over windows covering each position."""
    totals = np.zeros(n_positions, dtype=np.float64)
    counts = np.zeros(n_positions, dtype=np.float64)
    for start, probabilities in window_probabilities:
        end = start + len(probabilities)
        totals[start:end] += probabilities
        counts[start:end] += 1.0
    counts[counts == 0.0] = 1.0
    return totals / counts


def aggregate_window_class_probabilities(
    window_class_probabilities: list[tuple[np.ndarray, int]],
) -> np.ndarray:
    """Region-level five-class probabilities (Supplementary Method 6.4).

    Each entry is ``(class_probability_vector, n_valid_positions)`` for one
    window covering the region. Windows are averaged with weights
    proportional to their number of valid positions; a single-window region
    therefore returns its window probabilities unchanged.
    """
    if not window_class_probabilities:
        raise ValueError("at least one window is required")
    total_weight = 0
    aggregated = None
    for class_probabilities, n_valid in window_class_probabilities:
        if n_valid <= 0:
            continue
        if aggregated is None:
            aggregated = np.zeros_like(np.asarray(class_probabilities, dtype=np.float64))
        aggregated += np.asarray(class_probabilities, dtype=np.float64) * n_valid
        total_weight += n_valid
    if aggregated is None or total_weight == 0:
        raise ValueError("at least one window must contain valid positions")
    return aggregated / total_weight


@dataclass(frozen=True)
class GeneRecord:
    gene_id: str
    gene_start: int
    gene_end: int
    pfam_positions: tuple[int, ...]  # model position indices covered by this gene
    pfam_ids: tuple[str, ...]


@dataclass
class CalledRegion:
    start_index: int
    end_index: int
    positive_genes: int
    predicted_genes: int
    gene_ids: list[str] = field(default_factory=list)


def call_regions(genes: list[GeneRecord], position_probabilities: np.ndarray) -> list[CalledRegion]:
    """Apply the manuscript region-calling rules over an ordered gene list."""
    gene_positive: list[bool] = []
    for gene in genes:
        if not gene.pfam_positions:
            gene_positive.append(False)
            continue
        gene_score = float(max(position_probabilities[p] for p in gene.pfam_positions))
        gene_positive.append(any(position_probabilities[p] >= POSITION_THRESHOLD for p in gene.pfam_positions))

    def requirements_met(start: int, end: int) -> bool:
        positives = sum(gene_positive[start:end])
        return positives >= MIN_POSITIVE_GENES and (end - start) >= MIN_PREDICTED_GENES

    n = len(genes)
    positive_indices = [i for i, positive in enumerate(gene_positive) if positive]
    initial_regions: list[tuple[int, int]] = []
    if positive_indices:
        segment_start = positive_indices[0]
        segment_end = positive_indices[0]
        for idx in positive_indices[1:]:
            if idx - segment_end - 1 <= MAX_GAP_GENES:
                segment_end = idx
            else:
                initial_regions.append((segment_start, segment_end + 1))
                segment_start = idx
                segment_end = idx
        initial_regions.append((segment_start, segment_end + 1))

    kept = [(s, e) for (s, e) in initial_regions if requirements_met(s, e)]

    extended: list[tuple[int, int]] = []
    for start, end in kept:
        extended.append((max(0, start - EXTENSION_GENES), min(n, end + EXTENSION_GENES)))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(extended):
        if merged and start - merged[-1][1] <= MERGE_GAP_GENES:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    regions: list[CalledRegion] = []
    for start, end in merged:
        if not requirements_met(start, end):
            continue
        regions.append(
            CalledRegion(
                start_index=start,
                end_index=end,
                positive_genes=sum(gene_positive[start:end]),
                predicted_genes=end - start,
                gene_ids=[g.gene_id for g in genes[start:end]],
            )
        )
    return regions


def region_is_nrps(class_probabilities: dict[str, float], region_pfam_ids: set[str]) -> bool:
    best_class = max(class_probabilities, key=lambda k: class_probabilities[k])
    if best_class != "NRPS":
        return False
    return any(strip_pfam_version(p) in NRPS_SIGNATURE_PFAMS for p in region_pfam_ids)
