"""Contract tests for the nine-genome benchmark evaluation (Method 7).

Fixtures are synthetic; tests assert the manuscript-fixed evaluation rules
and benchmark constants rather than trained performance.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgc2activity.detection.evaluation import (
    BENCHMARK_GENE_THRESHOLD,
    BENCHMARK_PFAM_ANNOTATED_GENES,
    BENCHMARK_POSITIVE_GENES,
    BENCHMARK_TOTAL_GENES,
    GENOME_IDENTIFIERS,
    PRODUCT_CLASSES,
    REGION_POSITION_THRESHOLD,
    BenchmarkGene,
    PredictedRegion,
    TruthRegion,
    aggregate_position_scores_to_genes,
    assign_gene_labels,
    auroc,
    auprc,
    binary_metrics_at_threshold,
    classify_recovery,
    evaluate_gene_level,
    merge_product_class,
    project_regions_to_gene_scores,
    product_class_metrics,
    region_level_metrics,
    stratified_counts,
)


def make_genes():
    return [
        BenchmarkGene("g0_gene1", "genome_0", 100, 300, True, 0),
        BenchmarkGene("g0_gene2", "genome_0", 400, 600, True, 0),
        BenchmarkGene("g0_gene3", "genome_0", 700, 900, False, 0),
        BenchmarkGene("g1_gene1", "genome_1", 50, 250, True, 0),
    ]


class TestBenchmarkConstants(unittest.TestCase):
    def test_nine_anonymous_genomes(self):
        self.assertEqual(len(GENOME_IDENTIFIERS), 9)
        self.assertEqual(GENOME_IDENTIFIERS[0], "genome_0")
        self.assertEqual(GENOME_IDENTIFIERS[-1], "genome_8")

    def test_bgc_prophet_benchmark_counts(self):
        self.assertEqual(BENCHMARK_TOTAL_GENES, 67_072)
        self.assertEqual(BENCHMARK_POSITIVE_GENES, 3_696)
        self.assertEqual(BENCHMARK_PFAM_ANNOTATED_GENES, 54_285)

    def test_thresholds(self):
        self.assertEqual(BENCHMARK_GENE_THRESHOLD, 0.5)
        self.assertEqual(REGION_POSITION_THRESHOLD, 0.45)

    def test_product_class_merging(self):
        self.assertEqual(PRODUCT_CLASSES, ("NRPS", "PKS", "RiPP", "terpene", "other"))
        self.assertEqual(merge_product_class("NRPS-PKS-like"), "NRPS")
        self.assertEqual(merge_product_class("Type I Polyketide"), "PKS")
        self.assertEqual(merge_product_class("ribosomally synthesized peptide"), "RiPP")
        self.assertEqual(merge_product_class("terpenoid"), "terpene")
        self.assertEqual(merge_product_class("siderophore"), "other")
        self.assertEqual(merge_product_class(""), "other")


class TestGeneLevelProjection(unittest.TestCase):
    def test_labels_from_truth_overlap(self):
        genes = assign_gene_labels(make_genes(), [TruthRegion("genome_0", 350, 650, "NRPS")])
        self.assertEqual([gene.label for gene in genes], [0, 1, 0, 0])

    def test_region_projection_by_coordinate_overlap(self):
        genes = make_genes()
        scores = project_regions_to_gene_scores(
            [PredictedRegion("genome_0", 250, 450)], genes,
        )
        self.assertEqual(scores["g0_gene1"], 1.0)  # overlaps at 250-300
        self.assertEqual(scores["g0_gene2"], 1.0)  # overlaps at 400-450
        self.assertEqual(scores["g0_gene3"], 0.0)
        self.assertEqual(scores["g1_gene1"], 0.0)

    def test_position_scores_reduced_to_gene_maximum(self):
        genes = make_genes()
        position_scores = {
            ("genome_0", 150): 0.30,
            ("genome_0", 200): 0.72,
            ("genome_0", 1000): 0.99,  # outside every gene
        }
        scores = aggregate_position_scores_to_genes(position_scores, genes, default_score=0.0)
        self.assertAlmostEqual(scores["g0_gene1"], 0.72)
        self.assertAlmostEqual(scores["g0_gene2"], 0.0)  # gene without Pfam -> 0.0
        self.assertAlmostEqual(scores["g1_gene1"], 0.0)


class TestGeneLevelMetrics(unittest.TestCase):
    def test_binary_metrics_known_example(self):
        scores = [0.9, 0.6, 0.4, 0.2]
        labels = [1, 1, 0, 0]
        metrics = binary_metrics_at_threshold(scores, labels, threshold=0.5)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fn"], 0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_binary_metrics_mixed_example(self):
        scores = [0.9, 0.8, 0.3, 0.2]
        labels = [1, 0, 1, 0]
        metrics = binary_metrics_at_threshold(scores, labels)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_auroc_perfect_and_chance(self):
        self.assertAlmostEqual(auroc([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(auroc([0.9, 0.8, 0.7, 0.6], [1, 0, 0, 1]), 0.5)
        self.assertAlmostEqual(auroc([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]), 0.75)

    def test_auprc_perfect(self):
        self.assertAlmostEqual(auprc([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0]), 1.0)

    def test_evaluate_gene_level_block(self):
        metrics = evaluate_gene_level([0.9, 0.6, 0.4, 0.2], [1, 1, 0, 0])
        for key in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "threshold"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["threshold"], 0.5)


class TestRecoveryClassification(unittest.TestCase):
    def setUp(self):
        self.truths = [
            TruthRegion("genome_0", 100, 500, "NRPS"),   # model only
            TruthRegion("genome_1", 100, 500, "PKS"),    # comparator only
            TruthRegion("genome_2", 100, 500, "RiPP"),   # both
            TruthRegion("genome_3", 100, 500, "other"),  # none
        ]
        self.predictions = {
            "proposed_model": [
                PredictedRegion("genome_0", 300, 700),
                PredictedRegion("genome_2", 400, 600),
                PredictedRegion("genome_5", 10, 90),  # unmatched extra prediction
            ],
            "comparator_a": [
                PredictedRegion("genome_1", 200, 300),
                PredictedRegion("genome_2", 450, 550),
            ],
        }

    def test_categories(self):
        report = classify_recovery(self.truths, self.predictions, "proposed_model")
        self.assertEqual(
            report.categories,
            ["model_exclusive", "comparator_exclusive", "jointly_recovered", "missed_by_all"],
        )
        self.assertEqual(report.recovering_methods[0], ("proposed_model",))
        self.assertEqual(report.recovering_methods[2], ("comparator_a", "proposed_model"))

    def test_unmatched_predictions_reported_separately(self):
        report = classify_recovery(self.truths, self.predictions, "proposed_model")
        self.assertEqual(len(report.unmatched_predictions["proposed_model"]), 1)
        self.assertEqual(report.unmatched_predictions["proposed_model"][0].genome_id, "genome_5")
        self.assertEqual(report.unmatched_predictions["comparator_a"], [])

    def test_stratification_by_product_class(self):
        report = classify_recovery(self.truths, self.predictions, "proposed_model")
        counts = stratified_counts(report, self.truths)
        self.assertEqual(counts[("model_exclusive", "NRPS")], 1)
        self.assertEqual(counts[("jointly_recovered", "RiPP")], 1)

    def test_region_level_reference_metrics(self):
        metrics = region_level_metrics(
            self.predictions["proposed_model"], self.truths,
        )
        self.assertEqual(metrics["tp"], 2.0)
        self.assertEqual(metrics["fp"], 1.0)
        self.assertEqual(metrics["fn"], 2.0)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 0.5)


class TestDistanceAnalysis(unittest.TestCase):
    def test_minimum_training_distance(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise unittest.SkipTest("numpy is required for distance analysis")
        from bgc2activity.detection.evaluation import minimum_training_distance

        training_sets = [frozenset({"PF00001", "PF00002"})]
        training_centroids = np.array([[1.0, 0.0]])
        query_sets = [frozenset({"PF00001", "PF00002"}), frozenset({"PF99999"})]
        query_centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        minima = minimum_training_distance(query_sets, query_centroids, training_sets, training_centroids)
        self.assertAlmostEqual(minima[0], 0.0)
        self.assertAlmostEqual(minima[1], 1.0)


class TestProductClassMetrics(unittest.TestCase):
    def test_macro_f1_and_confusion_matrix(self):
        labels = ["NRPS", "NRPS", "PKS", "RiPP", "other"]
        predictions = ["NRPS", "PKS", "PKS", "RiPP", "other"]
        metrics = product_class_metrics(predictions, labels)
        self.assertAlmostEqual(metrics["per_class"]["NRPS"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["PKS"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["RiPP"]["f1"], 1.0)
        self.assertAlmostEqual(metrics["confusion"]["NRPS"]["PKS"], 1)
        # terpene has no support -> per-class F1 0 in the macro average
        expected_macro = (2 / 3 + 2 / 3 + 1.0 + 0.0 + 1.0) / 5
        self.assertAlmostEqual(metrics["macro_f1"], expected_macro)

    def test_rejects_classes_outside_five_categories(self):
        with self.assertRaises(ValueError):
            product_class_metrics(["NRPS"], ["hybrid"])


if __name__ == "__main__":
    unittest.main()
