"""Pretraining evaluation and downstream representation quality
(Supplementary Method 10.3).

Pretraining evaluation comprises masked-recovery performance (smooth L1,
cosine similarity at masked positions) and candidate-level representation
quality (chemical-neighbour retrieval, k-NN enrichment with structural-
cluster filtering, and linear probes). UMAP is used only for two-dimensional
display and does not participate in training or model selection, so it is not
implemented here.
"""

from __future__ import annotations

import numpy as np


def masked_recovery_metrics(
    predictions: np.ndarray, targets: np.ndarray, selection: np.ndarray,
) -> dict[str, float]:
    """Masked-recovery performance at the selected positions (Method 10.3).

    ``predictions``/``targets`` are (batch, length, target_dimension) chemical
    targets; ``selection`` is the boolean mask of predicted positions.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    selection = np.asarray(selection, dtype=bool)
    if predictions.shape != targets.shape or predictions.shape[:2] != selection.shape:
        raise ValueError("predictions, targets and selection shapes must align")
    if not selection.any():
        raise ValueError("at least one selected position is required")
    residual = predictions[selection] - targets[selection]
    abs_residual = np.abs(residual)
    smooth_l1 = float(np.where(abs_residual < 1.0, 0.5 * residual ** 2, abs_residual - 0.5).mean())
    norm_product = np.linalg.norm(predictions[selection], axis=-1) * np.linalg.norm(targets[selection], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.where(
            norm_product > 0,
            (predictions[selection] * targets[selection]).sum(axis=-1) / np.where(norm_product > 0, norm_product, 1.0),
            0.0,
        )
    return {"smooth_l1": smooth_l1, "cosine_similarity": float(cosine.mean())}


def _pairwise_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query_squared = (query ** 2).sum(axis=1)[:, None]
    gallery_squared = (gallery ** 2).sum(axis=1)[None, :]
    cross = query @ gallery.T
    squared = np.maximum(query_squared + gallery_squared - 2.0 * cross, 0.0)
    return np.sqrt(squared)


def chemical_neighbour_retrieval(
    query_representations: np.ndarray,
    query_cluster_ids: np.ndarray,
    gallery_representations: np.ndarray,
    gallery_cluster_ids: np.ndarray,
    top_k: int = 1,
) -> float:
    """Chemical-neighbour retrieval hit rate (Method 10.3).

    For each query, the ``top_k`` nearest gallery representations are
    retrieved; the query counts as a hit when at least one retrieved
    neighbour belongs to the same structural cluster.
    """
    if top_k < 1 or top_k > len(gallery_representations):
        raise ValueError("top_k must be within the gallery size")
    distances = _pairwise_distances(np.asarray(query_representations, float), np.asarray(gallery_representations, float))
    order = np.argsort(distances, axis=1, kind="stable")[:, :top_k]
    hits = 0
    for row, neighbours in enumerate(order):
        if any(gallery_cluster_ids[neighbour] == query_cluster_ids[row] for neighbour in neighbours):
            hits += 1
    return hits / len(order)


def knn_enrichment(
    query_representations: np.ndarray,
    query_labels: np.ndarray,
    gallery_representations: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
    excluded_gallery_mask_by_query: np.ndarray | None = None,
) -> float:
    """k-NN label enrichment with structural-cluster filtering (Method 10.3).

    ``excluded_gallery_mask_by_query[i]`` marks gallery members excluded for
    query ``i`` (its own structural cluster and neighbours), so enrichment is
    measured beyond near-duplicates. Majority vote over the ``k`` nearest
    admissible neighbours predicts each query label; returns accuracy.
    """
    if k < 1:
        raise ValueError("k must be positive")
    distances = _pairwise_distances(np.asarray(query_representations, float), np.asarray(gallery_representations, float))
    if excluded_gallery_mask_by_query is not None:
        excluded_gallery_mask_by_query = np.asarray(excluded_gallery_mask_by_query, dtype=bool)
        if excluded_gallery_mask_by_query.shape != distances.shape:
            raise ValueError("exclusion mask must align with (queries, gallery)")
        distances = np.where(excluded_gallery_mask_by_query, np.inf, distances)
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    correct = 0
    for row, neighbours in enumerate(order):
        admissible = [neighbour for neighbour in neighbours if np.isfinite(distances[row, neighbour])]
        if not admissible:
            continue
        votes = int(np.bincount(np.asarray(gallery_labels)[admissible].astype(int), minlength=2).argmax())
        correct += int(votes == int(query_labels[row]))
    return correct / len(order)


def linear_probe_accuracy(
    train_features: np.ndarray, train_labels: np.ndarray,
    test_features: np.ndarray, test_labels: np.ndarray,
) -> float:
    """Frozen-representation linear probe (Method 10.3; requires scikit-learn)."""
    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(max_iter=2000)
    probe.fit(np.asarray(train_features, float), np.asarray(train_labels, int))
    return float(probe.score(np.asarray(test_features, float), np.asarray(test_labels, int)))
