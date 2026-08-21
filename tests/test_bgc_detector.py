"""Contract tests for the reconstructed BGC detection module.

All fixtures are synthetic and deterministic; the tests assert the
manuscript-fixed architecture and rule parameters rather than trained
performance.
"""

import sys
import unittest
from pathlib import Path

import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - dependency-free environments
    raise unittest.SkipTest("numpy is required for this test module")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency-free environments
    raise unittest.SkipTest("torch is required for this test module")

from bgc2activity.detection import (
    MultitaskBGCDetector,
    RegionSample,
    aggregate_window_probabilities,
    call_regions,
    collate_regions,
    composite_bgc_distance,
    family_heldout_partition,
    family_overlap_fraction,
    load_balancing_loss,
    multitask_loss,
    pairwise_distance_matrix,
    random_partition,
    region_is_nrps,
    single_linkage_families,
    window_starts,
)
from bgc2activity.detection.data import OOV_TOKEN
from bgc2activity.detection.model import (
    DETECTION_THRESHOLD,
    HIDDEN_DIMENSION,
    MOE_EXPERTS,
    N_LAYERS,
    PFAM2VEC_DIMENSION,
)
from bgc2activity.detection.regions import GeneRecord


def make_model() -> MultitaskBGCDetector:
    torch.manual_seed(42)
    return MultitaskBGCDetector()


def make_batch():
    vocabulary = {OOV_TOKEN: 0, "PF00001": 1, "PF00002": 2, "PF00003": 3}
    matrix = torch.randn(len(vocabulary), PFAM2VEC_DIMENSION)
    samples = [
        RegionSample("a", ("PF00001", "PF00002", "PF00003"), (1, 1, 0), 0, 0),
        RegionSample("b", ("PF00001", "PF99999"), (1, 0), 1, 1),  # PF99999 -> OOV
    ]
    batch = collate_regions(samples, vocabulary, matrix)
    return batch


class ArchitectureTests(unittest.TestCase):
    def test_manuscript_fixed_hyperparameters(self):
        self.assertEqual(PFAM2VEC_DIMENSION, 102)
        self.assertEqual(HIDDEN_DIMENSION, 128)
        self.assertEqual(N_LAYERS, 4)
        self.assertEqual(MOE_EXPERTS, 8)
        self.assertAlmostEqual(DETECTION_THRESHOLD, 0.45)

    def test_forward_shapes_and_padding_mask(self):
        model = make_model()
        batch = make_batch()
        output = model(batch["embeddings"], batch["valid"])
        self.assertEqual(output.detection_logits.shape, (2, 3))
        self.assertEqual(output.class_logits.shape, (2, 5))
        self.assertEqual(output.region_representation.shape, (2, HIDDEN_DIMENSION))
        # padding positions must not influence the pooled representation:
        vocabulary = {OOV_TOKEN: 0, "PF00001": 1, "PF00002": 2, "PF00003": 3}
        matrix = torch.randn(len(vocabulary), PFAM2VEC_DIMENSION)
        padded = collate_regions(
            [
                RegionSample("a", ("PF00001", "PF00002"), (1, 1), 0, 0),
                RegionSample("b", ("PF00001", "PF00002", "PF00003"), (1, 1, 0), 0, 0),
            ],
            vocabulary, matrix,
        )
        alone = collate_regions(
            [RegionSample("a", ("PF00001", "PF00002"), (1, 1), 0, 0)],
            vocabulary, matrix,
        )
        model.eval()
        out_padded = model(padded["embeddings"][:1], padded["valid"][:1])
        out_alone = model(alone["embeddings"], alone["valid"])
        self.assertTrue(torch.allclose(out_padded.region_representation, out_alone.region_representation, atol=1e-6))

    def test_oov_token_maps_to_zero_row(self):
        vocabulary = {OOV_TOKEN: 0, "PF00001": 1}
        matrix = torch.randn(len(vocabulary), PFAM2VEC_DIMENSION)
        matrix[0] = 0.0
        batch = collate_regions(
            [RegionSample("a", ("PF00001", "PF99999"), (1, 0), 0, 0)], vocabulary, matrix)
        self.assertTrue(torch.equal(batch["embeddings"][0, 1], matrix[0]))


class LossTests(unittest.TestCase):
    def test_padding_positions_are_excluded_from_detection_loss(self):
        logits = torch.tensor([[2.0, -2.0, 100.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])
        valid = torch.tensor([[1.0, 1.0, 0.0]])
        loss = multitask_loss(
            logits, labels, valid,
            torch.zeros(1, 5), torch.zeros(1, dtype=torch.long),
            torch.randn(1, 8), torch.arange(1), torch.tensor(0.0),
        )
        reference = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, :2], labels[:, :2])
        self.assertAlmostEqual(float(loss.detection), float(reference), places=5)

    def test_triplet_loss_is_zero_without_valid_anchors(self):
        loss = multitask_loss(
            torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 3),
            torch.zeros(2, 5), torch.zeros(2, dtype=torch.long),
            torch.randn(2, 8), torch.tensor([0, 0]), torch.tensor(0.0),
        )
        self.assertEqual(float(loss.evolutionary), 0.0)

    def test_load_balancing_term_is_added_with_manuscript_coefficient(self):
        lb = torch.tensor(2.5)
        loss = multitask_loss(
            torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 3),
            torch.zeros(1, 5), torch.zeros(1, dtype=torch.long),
            torch.randn(1, 8), torch.arange(1), lb,
        )
        self.assertAlmostEqual(float(loss.total),
                               float(loss.detection + loss.classification + loss.evolutionary + 0.01 * lb),
                               places=5)


class FamilyPartitionTests(unittest.TestCase):
    def test_composite_distance_extremes(self):
        zero = np.zeros(8)
        vector = np.ones(8)
        self.assertEqual(composite_bgc_distance(frozenset(), frozenset({"PF1"}), zero, vector), 1.0)
        self.assertAlmostEqual(composite_bgc_distance(frozenset({"PF1"}), frozenset({"PF1"}), vector, vector), 0.0)

    def test_single_linkage_families_and_heldout_has_no_family_overlap(self):
        sets = [frozenset({"A", "B"}), frozenset({"A", "B", "C"}), frozenset({"X"}), frozenset({"X", "Y"})]
        centroids = np.array([[1, 1, 0], [1, 1, 0.1], [0, 0, 1], [0, 0.1, 1]], dtype=float)
        matrix = pairwise_distance_matrix(sets, centroids)
        labels = single_linkage_families(matrix, threshold=0.35)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        mask = family_heldout_partition(labels, n_heldout=2)
        self.assertEqual(family_overlap_fraction(labels, mask), 0.0)

    def test_random_partition_sizes_match_manuscript(self):
        mask = random_partition(3968, 1190)
        self.assertEqual(int(mask.sum()), 1190)
        self.assertEqual(len(mask), 3968)


class RegionCallingTests(unittest.TestCase):
    def make_genes(self, n: int) -> list[GeneRecord]:
        return [
            GeneRecord(f"g{i}", i * 1000, i * 1000 + 900, (i,), (f"PF{i:05d}",))
            for i in range(n)
        ]

    def test_minimum_region_requirements(self):
        genes = self.make_genes(10)
        probabilities = np.zeros(10)
        probabilities[[1, 3, 5]] = 0.9  # 3 positives spanning a 5-gene segment
        regions = call_regions(genes, probabilities)
        self.assertEqual(len(regions), 1)
        self.assertGreaterEqual(regions[0].positive_genes, 3)
        self.assertGreaterEqual(regions[0].predicted_genes, 5)

    def test_two_positives_do_not_form_a_region(self):
        genes = self.make_genes(10)
        probabilities = np.zeros(10)
        probabilities[[2, 3]] = 0.9
        self.assertEqual(call_regions(genes, probabilities), [])

    def test_gap_of_two_joins_three_blocks_merge_and_extension(self):
        genes = self.make_genes(16)
        probabilities = np.zeros(16)
        probabilities[[1, 2, 3, 6, 7, 9]] = 0.9
        regions = call_regions(genes, probabilities)
        self.assertEqual(len(regions), 1)
        self.assertLessEqual(regions[0].start_index, 1)
        self.assertGreaterEqual(regions[0].end_index, 10)

    def test_window_aggregation_averages_overlaps(self):
        windows = [(0, np.array([0.2, 0.4, 0.6, 0.8])), (2, np.array([1.0, 1.0, 0.0, 0.0]))]
        aggregated = aggregate_window_probabilities(windows, n_positions=6)
        self.assertAlmostEqual(aggregated[0], 0.2)
        self.assertAlmostEqual(aggregated[2], 0.8)  # (0.6 + 1.0) / 2
        self.assertAlmostEqual(aggregated[4], 0.0)

    def test_window_starts_cover_long_sequences(self):
        starts = window_starts(600)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 600 - 256)
        self.assertTrue(all(b - a == 128 for a, b in zip(starts, starts[1:-1])))
        self.assertLessEqual(starts[-1] - starts[-2], 128)

    def test_nrps_rule_requires_class_and_signature_pfam(self):
        good = {"NRPS": 0.7, "PKS": 0.2, "RiPP": 0.05, "Terpene": 0.03, "Other": 0.02}
        self.assertTrue(region_is_nrps(good, {"PF00668.12", "PF99999"}))
        self.assertFalse(region_is_nrps(good, {"PF99999"}))
        bad = dict(good); bad["NRPS"], bad["PKS"] = 0.2, 0.7
        self.assertFalse(region_is_nrps(bad, {"PF00668"}))


if __name__ == "__main__":
    unittest.main()
