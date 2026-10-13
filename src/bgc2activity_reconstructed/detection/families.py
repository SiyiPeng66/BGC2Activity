"""Evolutionary distance, family clustering and low-leakage partitioning.

Implements Supplementary Method 5 of the final manuscript:

    d_BGC(p, q) = 1/2 * (1 - |D_p ∩ D_q| / |D_p ∪ D_q|)
                + 1/2 * (1 - cos(c_p, c_q))

where D is the Pfam set and c the pfam2vec centroid vector. Empty sets or zero
vectors are treated as maximal distance. Families are formed by single-linkage
clustering at threshold d_BGC <= 0.35. The evolutionary family-held-out
partition assigns whole families as units (seed 42) so that no family crosses
the train/held-out boundary; the manuscript-fixed sizes are 2,162 training and
1,806 held-out samples (0% family overlap), versus 2,778 / 1,190 for the random
partition (12.0% of families crossing partitions).
"""

from __future__ import annotations

import numpy as np

FAMILY_DISTANCE_THRESHOLD = 0.35
PARTITION_SEED = 42
RANDOM_SPLIT_SIZES = (2778, 1190)
FAMILY_HELDOUT_SPLIT_SIZES = (2162, 1806)


def pfam_set_jaccard_distance(set_p: frozenset[str], set_q: frozenset[str]) -> float:
    if not set_p or not set_q:
        return 1.0
    intersection = len(set_p & set_q)
    union = len(set_p | set_q)
    return 1.0 - intersection / union


def centroid_cosine_distance(centroid_p: np.ndarray, centroid_q: np.ndarray) -> float:
    norm_p = np.linalg.norm(centroid_p)
    norm_q = np.linalg.norm(centroid_q)
    if norm_p == 0.0 or norm_q == 0.0:
        return 1.0
    cosine = float(np.dot(centroid_p, centroid_q) / (norm_p * norm_q))
    return 1.0 - max(-1.0, min(1.0, cosine))


def composite_bgc_distance(
    set_p: frozenset[str], set_q: frozenset[str],
    centroid_p: np.ndarray, centroid_q: np.ndarray,
) -> float:
    return 0.5 * pfam_set_jaccard_distance(set_p, set_q) + 0.5 * centroid_cosine_distance(centroid_p, centroid_q)


def pairwise_distance_matrix(
    pfam_sets: list[frozenset[str]], centroids: np.ndarray,
) -> np.ndarray:
    n = len(pfam_sets)
    matrix = np.zeros((n, n), dtype=np.float64)
    norms = np.linalg.norm(centroids, axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            distance = composite_bgc_distance(pfam_sets[i], pfam_sets[j], centroids[i], centroids[j])
            matrix[i, j] = matrix[j, i] = distance
    return matrix


def single_linkage_families(distance_matrix: np.ndarray, threshold: float = FAMILY_DISTANCE_THRESHOLD) -> np.ndarray:
    """Assign family labels by single-linkage clustering (union-find)."""
    n = distance_matrix.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    upper_i, upper_j = np.triu_indices(n, k=1)
    for i, j in zip(upper_i, upper_j):
        if distance_matrix[i, j] <= threshold:
            union(int(i), int(j))
    roots = [find(i) for i in range(n)]
    relabel: dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for i, root in enumerate(roots):
        relabel.setdefault(root, len(relabel))
        labels[i] = relabel[root]
    return labels


def family_heldout_partition(
    family_labels: np.ndarray, n_heldout: int, seed: int = PARTITION_SEED,
) -> np.ndarray:
    """Select whole families for the held-out partition (returns boolean mask).

    Families are drawn in a deterministic shuffled order until the held-out
    sample budget is reached as closely as the family structure allows.
    """
    rng = np.random.default_rng(seed)
    unique_families = np.unique(family_labels)
    order = rng.permutation(unique_families)
    mask = np.zeros(len(family_labels), dtype=bool)
    selected = 0
    for family in order:
        members = family_labels == family
        if selected >= n_heldout:
            break
        mask |= members
        selected = int(mask.sum())
    return mask


def random_partition(n_samples: int, n_heldout: int, seed: int = PARTITION_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    mask = np.zeros(n_samples, dtype=bool)
    mask[indices[:n_heldout]] = True
    return mask


def family_overlap_fraction(family_labels: np.ndarray, heldout_mask: np.ndarray) -> float:
    """Fraction of families with members on both sides of a partition."""
    crossing = 0
    total = 0
    for family in np.unique(family_labels):
        members = family_labels == family
        total += 1
        if heldout_mask[members].any() and (~heldout_mask[members]).any():
            crossing += 1
    return crossing / total if total else 0.0
