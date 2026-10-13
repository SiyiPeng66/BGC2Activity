"""Regression checks for standalone imports and complete windowed outputs."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from predict_bgc_regions import predict_region, parse_region
from bgc2activity_reconstructed.detection.data import RegionSample


class PositionModel(torch.nn.Module):
    def forward(self, embeddings, valid):
        width = embeddings.shape[1]
        return SimpleNamespace(
            detection_logits=torch.arange(width, dtype=torch.float32).unsqueeze(0) / 256,
            class_logits=torch.tensor([[0., 1., 0., 0., 0.]]),
        )


class InferenceContractTests(unittest.TestCase):
    def test_missing_domain_slot_matches_training_and_keeps_position(self):
        from bgc2activity_reconstructed.detection.data import collate_regions
        row = {"sample_id": "missing", "pfam_ids": "PF1;;PF1"}
        training = RegionSample("missing", ("PF1", "", "PF1"), (0, 0, 0), 0, 0)
        inference = parse_region(row)
        vocabulary, matrix = {"<OOV>": 0, "PF1": 1}, torch.arange(200.).reshape(2, 100)
        self.assertEqual(len(inference.pfam_ids), 3)
        torch.testing.assert_close(collate_regions([training], vocabulary, matrix)["embeddings"],
                                   collate_regions([inference], vocabulary, matrix)["embeddings"])

    def test_blank_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no Pfam positions"):
            parse_region({"sample_id": "empty", "pfam_ids": ""})

    def test_package_export_is_from_reconstruction(self):
        from bgc2activity_reconstructed.detection import MultitaskBGCDetector
        self.assertEqual(MultitaskBGCDetector.__module__,
                         "bgc2activity_reconstructed.detection.model")

    def test_long_region_preserves_and_averages_all_positions(self):
        sample = RegionSample("long", ("PF1",) * 363, (0,) * 363, 0, 0)
        result = predict_region(PositionModel(), sample, {"<OOV>": 0, "PF1": 1},
                                torch.zeros(2, 100))
        scores = np.array([float(v) for v in result["position_probabilities"].split(";")])
        self.assertEqual(len(scores), 363)
        window = torch.sigmoid(torch.arange(256) / 256).numpy()
        expected = np.concatenate([window[:107], (window[107:] + window[:149]) / 2,
                                   window[149:]])
        np.testing.assert_allclose(scores, expected, atol=1e-7)
        self.assertEqual(result["product_class"], "PKS")

    def test_empty_region_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "no Pfam positions"):
            predict_region(PositionModel(), RegionSample("empty", (), (), 0, 0),
                           {"<OOV>": 0}, torch.zeros(1, 100))


if __name__ == "__main__":
    unittest.main()
