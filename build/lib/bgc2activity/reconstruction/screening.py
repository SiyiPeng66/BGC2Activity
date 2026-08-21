"""Screening of NRPS-type BGCs and chemical-resolvability filtering
(Supplementary Method 8).

Three gates precede candidate-structure reconstruction:

8.1 Dereplication / MIBiG filtering
    Core NRPS proteins of each candidate BGC are aligned against the core NRPS
    proteins of experimentally characterized NRPS-type BGCs in MIBiG. A
    candidate is designated MIBiG-high-similarity and removed from
    reconstruction when it shows similarity > 80% to any MIBiG reference
    sequence with alignment coverage >= 80% of both sequences.

8.2 A-domain substrate prediction
    NRPStransformer predicts A-domain substrate specificity; A-domains are
    indexed by their order within NRPS proteins and biosynthetic modules and
    mapped to a unified substrate nomenclature. Only assignments reaching the
    confidence threshold (0.8, Supplementary Method 13 / Table S1) are
    accepted; the rest are marked unresolved.

8.3 Chemical resolvability
    Only candidates whose core sequences contain at least five monomers and
    whose building blocks all map to well-defined chemical entities are
    retained. Broad-category-only or unmappable substrates are not manually
    completed; candidates containing such unresolved positions are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

MIBIG_IDENTITY_THRESHOLD = 0.80  # strict: similarity must exceed 80%
MIBIG_COVERAGE_THRESHOLD = 0.80  # inclusive: coverage of both sequences >= 80%
ADOMAIN_CONFIDENCE_THRESHOLD = 0.8
MINIMUM_CORE_MONOMERS = 5


def is_mibig_high_similarity(
    n_identical: int,
    alignment_length: int,
    query_length: int,
    reference_length: int,
    identity_threshold: float = MIBIG_IDENTITY_THRESHOLD,
    coverage_threshold: float = MIBIG_COVERAGE_THRESHOLD,
) -> bool:
    """MIBiG-high-similarity rule of Supplementary Method 8.1.

    Similarity is the fraction of identical positions within the alignment
    (strictly greater than the threshold); coverage is the alignment length
    relative to each sequence (at least the threshold on both).
    """
    if alignment_length <= 0 or query_length <= 0 or reference_length <= 0:
        return False
    similarity = n_identical / alignment_length
    query_coverage = alignment_length / query_length
    reference_coverage = alignment_length / reference_length
    return (
        similarity > identity_threshold
        and query_coverage >= coverage_threshold
        and reference_coverage >= coverage_threshold
    )


def accept_adomain_prediction(confidence: float) -> bool:
    """Accept an A-domain substrate assignment only at confidence >= 0.8;
    lower-confidence assignments are marked unresolved (Method 8.2)."""
    return confidence >= ADOMAIN_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class ADomainPrediction:
    protein_index: int
    module_index: int
    adomain_index: int
    substrate_raw: str
    confidence: float


def reconstruct_core_sequence(
    predictions: list[ADomainPrediction],
    substrate_mapping: dict[str, str],
) -> tuple[list[str], list[int]]:
    """Reconstruct the ordered core monomer sequence (Method 8.2).

    Predictions are ordered by NRPS protein index, module index and A-domain
    index (gene orientation is already encoded in the module arrangement by
    the upstream workflow). Substrates are mapped through the unified
    nomenclature table; indices of unresolved positions (below the confidence
    threshold or absent from the mapping) are returned alongside so that
    chemical-resolvability filtering can discard the candidate.
    """
    ordered = sorted(
        predictions,
        key=lambda p: (p.protein_index, p.module_index, p.adomain_index),
    )
    monomers: list[str] = []
    unresolved: list[int] = []
    for position, prediction in enumerate(ordered):
        if not accept_adomain_prediction(prediction.confidence):
            unresolved.append(position)
            monomers.append("?")
            continue
        standardized = substrate_mapping.get(prediction.substrate_raw.strip().lower())
        if standardized is None:
            unresolved.append(position)
            monomers.append("?")
        else:
            monomers.append(standardized)
    return monomers, unresolved


def is_chemically_resolvable(monomers: list[str], unresolved_positions: list[int]) -> bool:
    """Chemical-resolvability gate of Supplementary Method 8.3: at least five
    monomers and every building block mapped to a well-defined chemical
    entity (no unresolved positions)."""
    return len(monomers) >= MINIMUM_CORE_MONOMERS and not unresolved_positions
