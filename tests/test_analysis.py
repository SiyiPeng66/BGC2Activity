"""Contract tests for experimental endpoint formulas (Methods 20-23) and
statistical assessment (Methods 24-25)."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgc2activity.analysis.experimental import (
    IMPLANT_DETECTION_LIMIT_LOG10,
    biofilm_inhibition_rate,
    cell_viability,
    haemolysis_rate,
    histopathology_score,
    implant_burden_log10,
    mature_biofilm_reduction_rate,
    mhc5,
    resistance_change,
    viable_burden_log10,
)
from bgc2activity.analysis.statistics import (
    MIC_CENSORING_THRESHOLD,
    MIC_SENSITIVITY_SUBSTITUTION,
    average_technical_replicates,
    brier_score,
    expected_calibration_error,
    fisher_exact_two_sided,
    log_loss,
    newcombe_risk_difference_interval,
    paired_bootstrap_delta_rho,
    reliability_curve,
    spearman_with_censoring,
    wilson_confidence_interval,
)


class TestHaemolysisAndViability(unittest.TestCase):
    def test_haemolysis_rate_formula(self):
        self.assertAlmostEqual(haemolysis_rate(0.3, 0.1, 0.9), 25.0)
        self.assertAlmostEqual(haemolysis_rate(0.1, 0.1, 0.9), 0.0)
        self.assertAlmostEqual(haemolysis_rate(0.9, 0.1, 0.9), 100.0)

    def test_mhc5_highest_concentration_within_5_percent(self):
        concentrations = [8, 16, 32, 64, 128, 256, 512]
        rates = [0.5, 1.2, 4.9, 6.0, 12.0, 30.0, 80.0]
        self.assertEqual(mhc5(concentrations, rates), 32.0)
        self.assertIsNone(mhc5(concentrations, [6.0] * 7))

    def test_cell_viability_formula(self):
        self.assertAlmostEqual(cell_viability(0.9, 0.1, 0.9), 100.0)
        self.assertAlmostEqual(cell_viability(0.5, 0.1, 0.9), 50.0)


class TestBiofilm(unittest.TestCase):
    def test_inhibition_rate_formula(self):
        self.assertAlmostEqual(biofilm_inhibition_rate(0.3, 0.1, 0.9), 75.0)
        self.assertAlmostEqual(biofilm_inhibition_rate(0.9, 0.1, 0.9), 0.0)

    def test_mature_reduction_formula(self):
        self.assertAlmostEqual(mature_biofilm_reduction_rate(0.3, 0.1, 0.9), 75.0)
        self.assertAlmostEqual(mature_biofilm_reduction_rate(0.5, 0.1, 0.5), 0.0)


class TestTimeKillAndResistance(unittest.TestCase):
    def test_viable_burden_log10_cfu_plus_one(self):
        self.assertAlmostEqual(viable_burden_log10(99.0), 2.0)
        self.assertAlmostEqual(viable_burden_log10(0.0), 0.0)

    def test_resistance_change_ratio(self):
        self.assertAlmostEqual(resistance_change(8.0, 4.0), 2.0)
        self.assertAlmostEqual(resistance_change(4.0, 4.0), 1.0)
        with self.assertRaises(ValueError):
            resistance_change(4.0, 0.0)


class TestImplantModel(unittest.TestCase):
    def test_implant_burden_and_detection_limit(self):
        self.assertEqual(IMPLANT_DETECTION_LIMIT_LOG10, 1.0)
        self.assertAlmostEqual(implant_burden_log10(5, 1.0), math.log10(50.0))
        self.assertAlmostEqual(implant_burden_log10(1, 1.0), 1.0)  # exactly 10 CFU
        self.assertAlmostEqual(implant_burden_log10(0, 1.0), 1.0)  # below limit
        self.assertAlmostEqual(implant_burden_log10(10, 10.0), math.log10(1000.0))

    def test_histopathology_scores(self):
        self.assertEqual(histopathology_score([3, 2, 1, 0], expected_items=4), 6)
        self.assertEqual(histopathology_score([1, 1, 1], expected_items=3), 3)
        with self.assertRaises(ValueError):
            histopathology_score([3, 2, 1], expected_items=4)
        with self.assertRaises(ValueError):
            histopathology_score([4, 0, 0, 0], expected_items=4)


class TestHitRateStatistics(unittest.TestCase):
    def test_wilson_interval_round_one(self):
        lower, upper = wilson_confidence_interval(8, 40)
        self.assertAlmostEqual(lower, 0.1050, places=3)
        self.assertAlmostEqual(upper, 0.3475, places=3)

    def test_wilson_interval_bounds(self):
        lower, upper = wilson_confidence_interval(0, 40)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_newcombe_risk_difference(self):
        lower, upper = newcombe_risk_difference_interval(8, 40, 3, 40)
        difference = 8 / 40 - 3 / 40
        self.assertLess(lower, difference)
        self.assertGreater(upper, difference)

    def test_fisher_exact_known_table(self):
        p_value = fisher_exact_two_sided(1, 9, 11, 3)
        self.assertAlmostEqual(p_value, 0.002759, places=5)
        self.assertAlmostEqual(fisher_exact_two_sided(5, 5, 5, 5), 1.0, places=6)


class TestRankingAndCalibration(unittest.TestCase):
    def test_censoring_constants(self):
        self.assertEqual(MIC_CENSORING_THRESHOLD, 256.0)
        self.assertEqual(MIC_SENSITIVITY_SUBSTITUTION, 512.0)

    def test_spearman_perfect_and_censored_weakest_rank(self):
        # lower MIC is better activity, so high scores anti-correlate with MIC
        self.assertAlmostEqual(spearman_with_censoring([8, 16, 64], [0.9, 0.5, 0.1]), -1.0)
        self.assertAlmostEqual(spearman_with_censoring([8, 16, 64], [0.1, 0.5, 0.9]), 1.0)
        # censored >256 values take the weakest ranks in the primary analysis
        rho = spearman_with_censoring([8, 300, 300], [3.0, 2.0, 1.0])
        self.assertLess(rho, 0.0)
        # sensitivity analysis substitutes 512
        rho_sensitivity = spearman_with_censoring([8, 300, 300], [3.0, 2.0, 1.0], substitute_censored=True)
        self.assertAlmostEqual(rho, rho_sensitivity)

    def test_paired_bootstrap_interval_ordered(self):
        lower, upper = paired_bootstrap_delta_rho(
            scores_base=[0.9, 0.8, 0.2, 0.1, 0.5],
            scores_calibrated=[0.85, 0.75, 0.25, 0.15, 0.45],
            mic_values=[8, 16, 128, 300, 64],
            cluster_ids=[0, 0, 1, 2, 3],
            n_bootstrap=200,
            seed=42,
        )
        self.assertLessEqual(lower, upper)

    def test_brier_and_log_loss(self):
        self.assertAlmostEqual(brier_score([0.8, 0.3], [1, 0]), (0.04 + 0.09) / 2)
        self.assertGreater(log_loss([0.99, 0.01], [1, 0]), 0.0)

    def test_reliability_curve_five_bins(self):
        curve = reliability_curve([0.05, 0.15, 0.9], [0, 0, 1])
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve[0]["count"], 2)
        self.assertEqual(curve[4]["count"], 1)
        self.assertAlmostEqual(curve[4]["observed_frequency"], 1.0)

    def test_ece_perfect_calibration(self):
        probabilities = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]
        ece = expected_calibration_error(probabilities, labels)
        self.assertAlmostEqual(ece, 0.0, places=6)

    def test_technical_replicate_averaging(self):
        averaged = average_technical_replicates({"exp1": [1.0, 2.0, 3.0], "exp2": [4.0, 6.0]})
        self.assertEqual(averaged, {"exp1": 2.0, "exp2": 5.0})


if __name__ == "__main__":
    unittest.main()
