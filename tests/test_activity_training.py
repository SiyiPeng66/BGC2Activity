"""Contract tests for the Base MIC Regression Model training protocol
(main text section 2.3; Supplementary Method 12). Requires torch."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency-free environments
    raise unittest.SkipTest("torch is required for this test module")

from bgc2activity.activity.training import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    ENSEMBLE_SEEDS,
    FINE_TUNED_LEARNING_RATE,
    GRADIENT_CLIP_NORM,
    HUBER_DELTA,
    MAX_EPOCHS,
    NEW_LAYER_LEARNING_RATE,
    SELECTION_TIE_TOLERANCE,
    WEIGHT_DECAY,
    EpochRecord,
    build_parameter_groups,
    ensemble_sbase,
    huber_loss,
    move_to_device,
    select_best_epoch,
)


class TestProtocolConstants(unittest.TestCase):
    def test_manuscript_fixed_constants(self):
        self.assertEqual(FINE_TUNED_LEARNING_RATE, 1e-5)
        self.assertEqual(NEW_LAYER_LEARNING_RATE, 3e-4)
        self.assertEqual(WEIGHT_DECAY, 1e-4)
        self.assertEqual(BATCH_SIZE, 64)
        self.assertEqual(MAX_EPOCHS, 100)
        self.assertEqual(GRADIENT_CLIP_NORM, 1.0)
        self.assertEqual(SELECTION_TIE_TOLERANCE, 1e-4)
        self.assertEqual(EARLY_STOPPING_PATIENCE, 12)
        self.assertEqual(len(ENSEMBLE_SEEDS), 5)

    def test_huber_delta_is_one_log10_unit(self):
        self.assertEqual(HUBER_DELTA, 1.0)


class TestHuberLoss(unittest.TestCase):
    def test_quadratic_then_linear(self):
        targets = torch.zeros(2)
        small = huber_loss(torch.tensor([0.1, -0.1]), targets)
        large = huber_loss(torch.tensor([4.0, -4.0]), targets)
        self.assertGreater(float(large), float(small))
        # within the delta the loss is 0.5 * r^2
        self.assertAlmostEqual(float(huber_loss(torch.tensor([0.5]), torch.tensor([0.0]))), 0.125, places=5)

    def test_zero_for_perfect_prediction(self):
        self.assertAlmostEqual(float(huber_loss(torch.tensor([1.7, 0.3]), torch.tensor([1.7, 0.3]))), 0.0, places=6)

    def test_nested_inputs_move_to_device(self):
        nested = {"x": torch.ones(1), "categorical": {"state": torch.zeros(1, dtype=torch.long)}}
        moved = move_to_device(nested, "cpu")
        self.assertEqual(moved["x"].device.type, "cpu")
        self.assertEqual(moved["categorical"]["state"].device.type, "cpu")


class TestParameterGroups(unittest.TestCase):
    def test_learning_rate_split(self):
        model = torch.nn.Sequential()
        model.add_module("pretrainer_layer", torch.nn.Linear(4, 4))
        model.add_module("regression_head", torch.nn.Linear(4, 1))
        groups = build_parameter_groups(model, ("pretrainer_layer",))
        self.assertEqual(groups[0]["lr"], FINE_TUNED_LEARNING_RATE)
        self.assertEqual(groups[1]["lr"], NEW_LAYER_LEARNING_RATE)
        self.assertEqual(len(groups[0]["params"]), 2)
        self.assertEqual(len(groups[1]["params"]), 2)


class TestSelectionAndEnsemble(unittest.TestCase):
    def test_selection_primary_rmse_lower_is_better(self):
        history = [
            EpochRecord(1, 0.50, 0.60),
            EpochRecord(2, 0.40, 0.55),
            EpochRecord(3, 0.42, 0.90),
        ]
        self.assertEqual(select_best_epoch(history), 2)

    def test_selection_tie_broken_by_spearman(self):
        history = [
            EpochRecord(1, 0.4000, 0.55),
            EpochRecord(2, 0.40005, 0.80),  # within 1e-4 -> higher Spearman wins
        ]
        self.assertEqual(select_best_epoch(history), 2)

    def test_ensemble_mean_and_dispersion(self):
        predictions = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([2.0, 3.0])]
        mean, dispersion = ensemble_sbase(predictions)
        self.assertTrue(torch.allclose(mean, torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.all(dispersion >= 0))


if __name__ == "__main__":
    unittest.main()
