"""Contract tests for pretraining, Base MIC Regression Model, calibration
and selection.

Tests assert manuscript-fixed rules (main text section 2.3; Methods 10-12,
14, 17) on synthetic fixtures; they do not claim to reproduce trained
performance.
"""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - dependency-free environments
    raise unittest.SkipTest("numpy is required for this test module")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency-free environments
    raise unittest.SkipTest("torch is required for this test module")

from bgc2activity.representation import (
    ChemicalAwarePretrainer,
    blake2b_text_vector,
    apply_masking,
    pretraining_loss,
)
from bgc2activity.activity import (
    BaseActivityModel,
    MicMeasurement,
    aggregate_sequence_labels,
    aggregate_sequence_targets,
    log10_mic_target,
    replicate_consistency_groups,
    strict_label_for_measurement,
)
from bgc2activity.activity.evaluation import (
    mean_absolute_error,
    observed_predicted_agreement,
    residual_structure,
    root_mean_squared_error,
    spearman,
)
from bgc2activity.calibration import (
    cross_fit_out_of_fold,
    cross_fit_out_of_fold_calibrated,
    fit_monotone_regression_calibration,
    fit_residual_update,
    stratified_fold_indices,
)
from bgc2activity.screening.selection import SelectionConstraints, select_cohort


class PretrainingTests(unittest.TestCase):
    def test_blake2b_vector_is_deterministic_and_normalized(self):
        v1 = blake2b_text_vector("lys(myr)")
        v2 = blake2b_text_vector("lys(myr)")
        self.assertTrue(torch.equal(v1, v2))
        self.assertAlmostEqual(float(v1.norm()), 1.0, places=5)
        self.assertEqual(v1.shape, (24,))

    def test_masking_selects_at_least_one_position_per_sequence(self):
        torch.manual_seed(0)
        model = ChemicalAwarePretrainer()
        hidden = torch.randn(4, 10, 96)
        mask = torch.ones(4, 10)
        mask[0, 5:] = 0
        _masked, selected = apply_masking(hidden, mask, model.mask_token, generator=torch.Generator().manual_seed(1))
        for row in range(4):
            self.assertGreaterEqual(int(selected[row].sum()), 1)
        self.assertFalse(bool(selected[0, 5:].any()))

    def test_pretraining_loss_structure(self):
        prediction = torch.randn(6, 32)
        target = torch.randn(6, 32)
        selection = torch.tensor([True, True, False, True, False, True])
        losses = pretraining_loss(prediction, target, selection)
        self.assertIn("total", losses)
        expected = losses["smooth_l1"] + 0.20 * losses["cosine"]
        self.assertAlmostEqual(float(losses["total"]), float(expected), places=5)

    def test_forward_output_shape(self):
        torch.manual_seed(0)
        model = ChemicalAwarePretrainer()
        text = torch.randn(2, 12, 24)
        desc = torch.randn(2, 12, 8)
        mask = torch.ones(2, 12)
        out = model(text, desc, mask)
        self.assertEqual(out.shape, (2, 12, 32))


class BaseModelTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(0)
        return BaseActivityModel(ChemicalAwarePretrainer())

    def test_recovery_head_removed_from_downstream_model(self):
        model = self.make_model()
        self.assertIsInstance(model.pretrainer.recovery_head, torch.nn.Identity)

    def test_freeze_boundary_matches_method_12(self):
        model = self.make_model()
        model.freeze_for_supervised_training()
        self.assertFalse(model.pretrainer.positional_encoding.requires_grad)
        for parameter in model.pretrainer.monomer_projection.parameters():
            self.assertFalse(parameter.requires_grad)
        layers = model.pretrainer.encoder.layers.layers
        for parameter in layers[0].parameters():
            self.assertFalse(parameter.requires_grad)
        for parameter in layers[1].parameters():
            self.assertFalse(parameter.requires_grad)
        for parameter in layers[2].parameters():
            self.assertTrue(parameter.requires_grad)
        for parameter in model.pretrainer.output_norm.parameters():
            self.assertTrue(parameter.requires_grad)
        for parameter in model.pooling.parameters():
            self.assertTrue(parameter.requires_grad)

    def test_sbase_is_scalar_prediction_without_probability_transform(self):
        model = self.make_model()
        model.eval()
        batch = dict(
            text_vectors=torch.randn(3, 9, 24),
            descriptors=torch.randn(3, 9, 8),
            attention_mask=torch.ones(3, 9),
            continuous=torch.randn(3, 7),
            missingness=torch.zeros(3, 7),
            categorical={name: torch.zeros(3, dtype=torch.long)
                         for name in ("n_terminal", "c_terminal", "lipidation", "cyclization")},
        )
        out = model(**batch)
        self.assertEqual(out.shape, (3,))  # scalar predicted log10(MIC) per candidate


class QuantitativeTargetTests(unittest.TestCase):
    def measurement(self, relation="=", value=16.0, bucket="ug_mL", upper=None, sequence="SEQ"):
        return MicMeasurement("DBAASP", "r1", sequence, "S. aureus", bucket, relation, value, upper)

    def test_point_and_bound_targets(self):
        self.assertAlmostEqual(log10_mic_target(self.measurement("=", 16.0)).log10_mic, np.log10(16.0))
        self.assertEqual(log10_mic_target(self.measurement("=", 16.0)).censoring, "none")
        self.assertEqual(log10_mic_target(self.measurement("<=", 8.0)).censoring, "left")
        self.assertEqual(log10_mic_target(self.measurement(">=", 64.0)).censoring, "right")

    def test_interval_uses_geometric_mean(self):
        target = log10_mic_target(self.measurement("<=", 8.0, upper=32.0))
        self.assertAlmostEqual(target.log10_mic, np.log10(np.sqrt(8.0 * 32.0)))

    def test_unknown_bucket_excluded(self):
        self.assertIsNone(log10_mic_target(self.measurement(bucket="mg_L")))

    def test_entity_aggregation_prefers_ugml_then_minimum(self):
        measurements = [
            self.measurement("=", 32.0, bucket="uM"),       # fallback bucket
            self.measurement("=", 64.0, bucket="ug_mL"),
            self.measurement("=", 16.0, bucket="ug_mL"),
        ]
        targets = aggregate_sequence_targets(measurements)
        self.assertAlmostEqual(targets["SEQ"], np.log10(16.0))

    def test_strict_labels_remain_descriptive_only(self):
        self.assertEqual(strict_label_for_measurement(self.measurement("=", 4.0)), 1)
        self.assertEqual(strict_label_for_measurement(self.measurement("=", 128.0)), 0)
        self.assertIsNone(strict_label_for_measurement(self.measurement("=", 32.0)))
        labels = aggregate_sequence_labels([
            self.measurement("=", 128.0), self.measurement("=", 4.0, sequence="SEQ"),
        ])
        self.assertEqual(labels["SEQ"], 1)
        groups = replicate_consistency_groups([
            self.measurement("=", 16.0), self.measurement("=", 32.0),
        ])
        self.assertEqual(len(groups[("r1", "S. aureus", "ug_mL")]), 2)


class RegressionMetricTests(unittest.TestCase):
    def test_metrics_on_known_values(self):
        observed = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 4.0]
        self.assertAlmostEqual(mean_absolute_error(observed, predicted), 1.0 / 3.0)
        self.assertAlmostEqual(root_mean_squared_error(observed, predicted), np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(spearman(observed, predicted), 1.0)

    def test_agreement_identity_and_residual_bins(self):
        observed = [0.0, 1.0, 2.0, 3.0]
        agreement = observed_predicted_agreement(observed, observed)
        self.assertAlmostEqual(agreement["slope"], 1.0)
        self.assertAlmostEqual(agreement["intercept"], 0.0)
        bins = residual_structure(observed, [o + 0.5 for o in observed], n_bins=2)
        self.assertTrue(all(abs(bin["mean_residual"] + 0.5) < 1e-9 for bin in bins))


class CalibrationTests(unittest.TestCase):
    def make_round1(self, n=40, seed=7):
        rng = np.random.default_rng(seed)
        targets = rng.normal(1.6, 0.6, n)
        sbase = targets + rng.normal(0, 0.5, n)
        features = targets[:, None] * rng.normal(0.5, 1.0, (n, 9)) + rng.normal(0, 1, (n, 9))
        return sbase, features, targets

    def test_residual_update_keeps_sbase_coefficient_one(self):
        sbase, features, targets = self.make_round1()
        model = fit_residual_update(sbase, features, targets)
        supd = model.predict_supd(sbase, features)
        shifted = model.predict_supd(sbase + 1.0, features)
        self.assertTrue(np.allclose(shifted - supd, 1.0))

    def test_development_oof_calibrated_uses_fold_specific_mappings(self):
        sbase, features, targets = self.make_round1()
        supd_oof, scal_dev_oof = cross_fit_out_of_fold_calibrated(sbase, features, targets)
        self.assertEqual(supd_oof.shape, (40,))
        self.assertEqual(scal_dev_oof.shape, (40,))
        self.assertTrue(np.isfinite(scal_dev_oof).all())
        # RD-12: development calibrated values equal manual per-fold refits
        for training_index, validation_index in stratified_fold_indices(targets):
            fold_calibration = fit_monotone_regression_calibration(
                supd_oof[training_index], targets[training_index])
            expected = fold_calibration.predict_calibrated(
                supd_oof[validation_index], training_supd=supd_oof[training_index])
            self.assertTrue(np.allclose(scal_dev_oof[validation_index], expected))

    def test_development_calibration_is_not_the_final_mapping_reapplied(self):
        sbase, features, targets = self.make_round1()
        supd_oof, scal_dev_oof = cross_fit_out_of_fold_calibrated(sbase, features, targets)
        final = fit_monotone_regression_calibration(supd_oof, targets)
        scal_final = final.predict_calibrated(supd_oof, training_supd=supd_oof)
        self.assertFalse(np.allclose(scal_dev_oof, scal_final))

    def test_monotone_calibration_is_monotone_and_truncated(self):
        sbase, features, targets = self.make_round1()
        model = fit_residual_update(sbase, features, targets)
        supd = model.predict_supd(sbase, features)
        calibration = fit_monotone_regression_calibration(supd, targets)
        grid = np.linspace(supd.min() - 2, supd.max() + 2, 200)
        calibrated = calibration.predict_calibrated(grid, training_supd=supd)
        self.assertTrue(bool(np.all(np.diff(calibrated) >= -1e-12)))
        # outside the training range the mapping saturates (truncation)
        self.assertEqual(calibrated[0], calibrated[1])
        self.assertEqual(calibrated[-1], calibrated[-2])

    def test_fold_indices_are_a_disjoint_stratified_cover(self):
        _sbase, _features, targets = self.make_round1()
        folds = stratified_fold_indices(targets)
        self.assertEqual(len(folds), 5)
        covered = []
        for training_index, validation_index in folds:
            self.assertEqual(len(validation_index), 8)
            self.assertEqual(len(training_index), 32)
            self.assertFalse(set(training_index) & set(validation_index))
            covered.extend(validation_index.tolist())
        self.assertEqual(sorted(covered), list(range(40)))

    def test_oof_supd_reproducible_from_cross_fit(self):
        sbase, features, targets = self.make_round1()
        supd_oof, _ = cross_fit_out_of_fold_calibrated(sbase, features, targets)
        self.assertTrue(np.allclose(supd_oof, cross_fit_out_of_fold(sbase, features, targets)))


class SelectionTests(unittest.TestCase):
    def make_pool(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "candidate_id": [f"C{i:04d}" for i in range(n)],
            "score": rng.normal(1.5, 0.5, n),  # predicted log10(MIC)
            "chemical_entity_id": [f"E{i % 150}" for i in range(n)],
            "core_sequence_cluster_id": [f"K{i % 40}" for i in range(n)],
            "parent_bgc_id": [f"B{i % 90}" for i in range(n)],
            "synthesis_feasible": [i % 17 != 0 for i in range(n)],
        })
        return df

    def test_cohort_size_and_constraints(self):
        pool = self.make_pool()
        constraints = SelectionConstraints(max_per_core_sequence_cluster=3, max_per_bgc=2)
        record = select_cohort(pool, constraints)
        self.assertEqual(len(record.selected), 40)
        selected_rows = pool[pool["candidate_id"].isin(record.selected)]
        self.assertLessEqual(selected_rows["chemical_entity_id"].value_counts().max(), 1)
        self.assertLessEqual(selected_rows["core_sequence_cluster_id"].value_counts().max(), 3)
        self.assertLessEqual(selected_rows["parent_bgc_id"].value_counts().max(), 2)
        self.assertTrue(bool(selected_rows["synthesis_feasible"].all()))

    def test_lowest_predicted_mic_ranked_first(self):
        pool = self.make_pool()
        record = select_cohort(pool, SelectionConstraints(max_per_chemical_entity=300,
                                                          max_per_core_sequence_cluster=300,
                                                          max_per_bgc=300))
        selected_scores = pool.set_index("candidate_id").loc[record.selected, "score"]
        feasible = pool[pool["synthesis_feasible"]]
        self.assertAlmostEqual(selected_scores.min(), feasible["score"].min())

    def test_round2_excludes_round1_candidates(self):
        pool = self.make_pool()
        constraints = SelectionConstraints(max_per_core_sequence_cluster=3, max_per_bgc=2)
        round1 = select_cohort(pool, constraints)
        round2 = select_cohort(pool, constraints, excluded_candidate_ids=frozenset(round1.selected))
        self.assertFalse(set(round1.selected) & set(round2.selected))
        self.assertEqual(len(round2.selected), 40)


if __name__ == "__main__":
    unittest.main()
