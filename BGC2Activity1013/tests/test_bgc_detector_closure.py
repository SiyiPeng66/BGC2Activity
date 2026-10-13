import csv
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgc2activity_reconstructed.detection.data import load_pfam2vec  # noqa: E402
from bgc2activity_reconstructed.detection.model import MultitaskBGCDetector  # noqa: E402


class BGCDetectorClosureTests(unittest.TestCase):
    def test_recovered_resource_and_smoke_checkpoint_are_loadable(self):
        vocabulary, matrix = load_pfam2vec(ROOT / "models/pfam2vec.csv.gz")
        self.assertGreater(len(vocabulary), 1)
        self.assertEqual(matrix.shape[1], 100)
        checkpoint = ROOT / "models/bgc_detector.pt"
        self.assertTrue(checkpoint.exists())
        model = MultitaskBGCDetector(pfam2vec_dimension=matrix.shape[1])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        output = model(torch.zeros(1, 4, matrix.shape[1]), torch.ones(1, 4))
        self.assertEqual(tuple(output.detection_logits.shape), (1, 4))
        self.assertEqual(tuple(output.class_logits.shape), (1, 5))

    def test_prediction_table_has_public_output_fields(self):
        path = ROOT / "examples/bgc/example_bgc_regions.tsv"
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle, delimiter="\t"))
        self.assertIn("start", row)
        self.assertIn("product_class", row)
        self.assertIn("end", row)


if __name__ == "__main__":
    unittest.main()
