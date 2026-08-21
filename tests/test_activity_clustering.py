"""Contract tests for structural clustering and leakage-free partitioning
(Supplementary Methods 10.1 and 12)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgc2activity.activity.clustering import (
    CLUSTER_PARTITION_SEED,
    LENGTH_RATIO_THRESHOLD,
    PARTITION_FRACTIONS,
    SEQUENCE_SIMILARITY_THRESHOLD,
    TANIMOTO_THRESHOLD,
    assign_structural_clusters,
    effective_training_mask,
    global_sequence_similarity,
    length_ratio,
    sequences_are_clustered,
    split_clusters,
    tanimoto_similarity,
)


class TestSimilarityRules(unittest.TestCase):
    def test_manuscript_thresholds(self):
        self.assertEqual(SEQUENCE_SIMILARITY_THRESHOLD, 0.80)
        self.assertEqual(LENGTH_RATIO_THRESHOLD, 0.80)
        self.assertEqual(TANIMOTO_THRESHOLD, 0.60)
        self.assertEqual(CLUSTER_PARTITION_SEED, 42)
        self.assertEqual(PARTITION_FRACTIONS, (0.70, 0.15, 0.15))

    def test_global_similarity_and_length_ratio(self):
        self.assertAlmostEqual(global_sequence_similarity("ABCD", "ABCD"), 1.0)
        self.assertAlmostEqual(length_ratio("ABCD", "ABCDEF"), 4 / 6)

    def test_length_ratio_gate_blocks_short_matches(self):
        # similarity 0.80 but length ratio 0.667 < 0.80 -> not clustered
        self.assertFalse(sequences_are_clustered("AAAA", "AAAAAA"))

    def test_tanimoto_path(self):
        self.assertAlmostEqual(tanimoto_similarity(frozenset({1, 2}), frozenset({2, 3})), 1 / 3)
        self.assertTrue(
            sequences_are_clustered("ABCD", "WXYZ", frozenset({1, 2, 3}), frozenset({1, 2, 3, 4}))
        )


class TestClusteringAndSplit(unittest.TestCase):
    def test_cluster_assignment(self):
        sequences = [
            "ALKGES", "ALKGES",      # identical
            "ALKGDS",                # similar to the first group (ratio 5/6)
            "WWWWFF", "WWWFFY",      # second group
        ]
        labels = assign_structural_clusters(sequences)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertNotEqual(labels[0], labels[3])

    def test_split_whole_clusters_deterministically(self):
        cluster_labels = [0, 0, 0, 1, 1, 2, 2, 3, 4, 5]
        assignment = split_clusters(cluster_labels)
        self.assertEqual(set(assignment), {"train", "validation", "test"})
        union = assignment["train"] | assignment["validation"] | assignment["test"]
        self.assertEqual(union, set(range(10)))
        self.assertFalse(assignment["train"] & assignment["validation"])
        # no cluster crosses a boundary
        for cluster_id in {0, 1, 2, 3, 4, 5}:
            members = {index for index, label in enumerate(cluster_labels) if label == cluster_id}
            placement = [name for name, indices in assignment.items() if members & indices]
            self.assertEqual(len(placement), 1)
        repeat = split_clusters(cluster_labels)
        self.assertEqual(assignment, repeat)

    def test_effective_training_mask(self):
        mask = effective_training_mask([True, False, True, False])
        self.assertEqual(mask, [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
