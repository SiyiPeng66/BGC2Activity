"""Contract tests for pretraining representation evaluation (Method 10.3)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - dependency-free environments
    raise unittest.SkipTest("numpy is required for this test module")

from bgc2activity.representation.evaluation import (
    chemical_neighbour_retrieval,
    knn_enrichment,
    masked_recovery_metrics,
)


class TestMaskedRecovery(unittest.TestCase):
    def test_perfect_recovery(self):
        predictions = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        targets = predictions.copy()
        selection = np.array([[True, False]])
        metrics = masked_recovery_metrics(predictions, targets, selection)
        self.assertAlmostEqual(metrics["smooth_l1"], 0.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)

    def test_requires_selected_positions(self):
        zeros = np.zeros((1, 2, 2))
        with self.assertRaises(ValueError):
            masked_recovery_metrics(zeros, zeros, np.zeros((1, 2), dtype=bool))


class TestRetrievalAndEnrichment(unittest.TestCase):
    def test_retrieval_returns_same_cluster_neighbour(self):
        queries = np.array([[0.0, 0.0], [10.0, 10.0]])
        gallery = np.array([[0.1, 0.0], [10.1, 10.0], [5.0, 5.0]])
        query_clusters = np.array([0, 1])
        gallery_clusters = np.array([0, 1, 2])
        self.assertAlmostEqual(
            chemical_neighbour_retrieval(queries, query_clusters, gallery, gallery_clusters, top_k=1), 1.0)
        self.assertAlmostEqual(
            chemical_neighbour_retrieval(queries, query_clusters, gallery, gallery_clusters, top_k=3), 1.0)

    def test_knn_enrichment_with_cluster_filtering(self):
        queries = np.array([[0.0, 0.0], [1.0, 0.0]])
        gallery = np.array([[0.0, 0.1], [0.9, 0.0], [5.0, 5.0], [5.1, 5.0]])
        gallery_labels = np.array([1, 0, 0, 0])
        exclusion = np.zeros((2, 4), dtype=bool)
        exclusion[0, 0] = True  # same structural cluster as query 0
        filtered = knn_enrichment(queries, np.array([0, 0]), gallery, gallery_labels, k=1,
                                  excluded_gallery_mask_by_query=exclusion)
        self.assertAlmostEqual(filtered, 1.0)  # same-cluster wrong-label neighbour removed
        unfiltered = knn_enrichment(queries, np.array([0, 0]), gallery, gallery_labels, k=1)
        self.assertAlmostEqual(unfiltered, 0.5)


if __name__ == "__main__":
    unittest.main()
