"""Contract tests for windowed class-probability aggregation (Method 6.4)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - dependency-free environments
    raise unittest.SkipTest("numpy is required for this test module")

from bgc2activity.detection.regions import (
    aggregate_window_class_probabilities,
    aggregate_window_probabilities,
    window_starts,
)


class TestPositionAggregation(unittest.TestCase):
    def test_overlap_positions_averaged_arithmetically(self):
        windows = [
            (0, np.array([0.2, 0.4, 0.6, 0.8])),
            (2, np.array([0.8, 0.6])),
        ]
        aggregated = aggregate_window_probabilities(windows, n_positions=6)
        self.assertAlmostEqual(aggregated[0], 0.2)
        self.assertAlmostEqual(aggregated[2], 0.7)  # (0.6 + 0.8) / 2
        self.assertAlmostEqual(aggregated[3], 0.7)  # (0.8 + 0.6) / 2
        self.assertAlmostEqual(aggregated[5], 0.0)  # uncovered tail stays 0

    def test_window_tiling_long_sequences(self):
        starts = window_starts(600)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 600 - 256)
        # stride 128 except the final window, which slides to the sequence end
        self.assertTrue(all(b - a == 128 for a, b in zip(starts[:-2], starts[1:-1])))
        self.assertLessEqual(starts[-1] - starts[-2], 128)


class TestClassProbabilityAggregation(unittest.TestCase):
    def test_weights_proportional_to_valid_positions(self):
        window_a = (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 200)
        window_b = (np.array([0.0, 1.0, 0.0, 0.0, 0.0]), 56)
        aggregated = aggregate_window_class_probabilities([window_a, window_b])
        self.assertAlmostEqual(aggregated[0], 200 / 256)
        self.assertAlmostEqual(aggregated[1], 56 / 256)
        self.assertAlmostEqual(float(aggregated.sum()), 1.0)

    def test_single_window_returned_unchanged(self):
        probabilities = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
        aggregated = aggregate_window_class_probabilities([(probabilities, 256)])
        self.assertTrue(np.allclose(aggregated, probabilities))

    def test_requires_valid_positions(self):
        with self.assertRaises(ValueError):
            aggregate_window_class_probabilities([])
        with self.assertRaises(ValueError):
            aggregate_window_class_probabilities([(np.array([0.5, 0.5]), 0)])


if __name__ == "__main__":
    unittest.main()
