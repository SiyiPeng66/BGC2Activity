"""Structural clustering and leakage-free partitioning of MIC-supervised
sequences (Supplementary Methods 10.1 and 12).

The curated MIC-supervised sequences are grouped into mutually exclusive
structural clusters by chemical entity and structural similarity: global
sequence similarity >= 0.80 and length ratio >= 0.80, or Morgan-fingerprint
Tanimoto >= 0.60. Complete clusters are assigned 70/15/15 to
training/validation/test sets with seed 42. The 4,739 sequences carrying
nonstandard monomers or explicit modification signals form the effective
training subset in the final project inventory; strict-label counts remain a
descriptive curation funnel and are not regression labels.

The same similarity rule underlies the pretraining corpus exclusion of
Supplementary Method 10.1; `assign_structural_clusters` is shared by both.
"""

from __future__ import annotations

from difflib import SequenceMatcher

SEQUENCE_SIMILARITY_THRESHOLD = 0.80
LENGTH_RATIO_THRESHOLD = 0.80
TANIMOTO_THRESHOLD = 0.60
CLUSTER_PARTITION_SEED = 42
PARTITION_FRACTIONS = (0.70, 0.15, 0.15)


def global_sequence_similarity(sequence_a: str, sequence_b: str) -> float:
    """Global similarity in [0, 1] (ratio of matching blocks); deterministic
    for fixed inputs."""
    return SequenceMatcher(None, sequence_a, sequence_b, autojunk=False).ratio()


def length_ratio(sequence_a: str, sequence_b: str) -> float:
    if not sequence_a or not sequence_b:
        return 0.0
    return min(len(sequence_a), len(sequence_b)) / max(len(sequence_a), len(sequence_b))


def tanimoto_similarity(fingerprint_a: frozenset[int], fingerprint_b: frozenset[int]) -> float:
    """Tanimoto coefficient of two bit fingerprints (Morgan fingerprints are
    supplied by the caller to stay dependency-free)."""
    if not fingerprint_a and not fingerprint_b:
        return 1.0
    union = len(fingerprint_a | fingerprint_b)
    if union == 0:
        return 1.0
    return len(fingerprint_a & fingerprint_b) / union


def sequences_are_clustered(
    sequence_a: str,
    sequence_b: str,
    fingerprint_a: frozenset[int] | None = None,
    fingerprint_b: frozenset[int] | None = None,
) -> bool:
    """Cluster-membership rule of Supplementary Method 12: similarity >= 0.80
    with length ratio >= 0.80, or Tanimoto >= 0.60 when fingerprints exist."""
    if fingerprint_a is not None and fingerprint_b is not None:
        if tanimoto_similarity(fingerprint_a, fingerprint_b) >= TANIMOTO_THRESHOLD:
            return True
    return (
        global_sequence_similarity(sequence_a, sequence_b) >= SEQUENCE_SIMILARITY_THRESHOLD
        and length_ratio(sequence_a, sequence_b) >= LENGTH_RATIO_THRESHOLD
    )


def assign_structural_clusters(
    sequences: list[str], fingerprints: list[frozenset[int]] | None = None,
) -> list[int]:
    """Union-find structural clustering over the frozen similarity rule."""
    if fingerprints is not None and len(fingerprints) != len(sequences):
        raise ValueError("fingerprints must align with sequences")
    parent = list(range(len(sequences)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            fingerprint_i = fingerprints[i] if fingerprints is not None else None
            fingerprint_j = fingerprints[j] if fingerprints is not None else None
            if sequences_are_clustered(sequences[i], sequences[j], fingerprint_i, fingerprint_j):
                union(i, j)
    roots = [find(i) for i in range(len(sequences))]
    relabel: dict[int, int] = {}
    labels = []
    for root in roots:
        relabel.setdefault(root, len(relabel))
        labels.append(relabel[root])
    return labels


def split_clusters(
    cluster_labels: list[int],
    seed: int = CLUSTER_PARTITION_SEED,
    fractions: tuple[float, float, float] = PARTITION_FRACTIONS,
) -> dict[str, set[int]]:
    """Assign whole clusters to train/validation/test (70/15/15, seed 42).

    Clusters are drawn in a deterministic shuffled order and placed into the
    partition that is furthest below its target sample share, so no cluster
    crosses a partition boundary.
    """
    import random

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("partition fractions must sum to 1")
    members: dict[int, list[int]] = {}
    for index, label in enumerate(cluster_labels):
        members.setdefault(label, []).append(index)
    cluster_ids = sorted(members)
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)
    total = len(cluster_labels)
    targets = {name: fraction * total for name, fraction in zip(("train", "validation", "test"), fractions)}
    assignment: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    counts = {"train": 0, "validation": 0, "test": 0}
    for cluster_id in cluster_ids:
        deficit = {name: targets[name] - counts[name] for name in assignment}
        chosen = max(deficit, key=lambda name: deficit[name])
        assignment[chosen].update(members[cluster_id])
        counts[chosen] += len(members[cluster_id])
    return assignment


def effective_training_mask(has_nonstandard_or_modification: list[bool]) -> list[bool]:
    """Effective training set: only sequences carrying nonstandard monomers or
    explicit modification signals are used for training (Method 12); the
    remainder serve validation and testing only."""
    return list(has_nonstandard_or_modification)
