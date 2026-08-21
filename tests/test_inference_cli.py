"""End-to-end smoke tests for the two public model-inference CLIs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    raise unittest.SkipTest("torch is required for inference CLI tests")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgc2activity.activity.base_model import CONTINUOUS_FEATURES, BaseActivityModel
from bgc2activity.detection.model import MultitaskBGCDetector
from bgc2activity.representation.pretraining import ChemicalAwarePretrainer


class InferenceCliTests(unittest.TestCase):
    def test_activity_scoring_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            checkpoint = work / "activity.pt"
            manifest = work / "manifest.json"
            input_table = work / "candidates.tsv"
            output_table = work / "scores.tsv"
            torch.save(BaseActivityModel(ChemicalAwarePretrainer()).state_dict(), checkpoint)
            feature_spec = {
                key: {feature: (1.0 if key == "standard_deviations" else 0.0)
                      for feature in CONTINUOUS_FEATURES}
                for key in ("medians", "means", "standard_deviations")
            }
            manifest.write_text(json.dumps({"feature_spec": feature_spec}), encoding="utf-8")
            fields = ["candidate_library_id", "monomers", "descriptors", *CONTINUOUS_FEATURES,
                      "n_terminal", "c_terminal", "lipidation", "cyclization"]
            row = {field: "0" for field in fields}
            row.update({
                "candidate_library_id": "CAND0001",
                "monomers": "A;K",
                "descriptors": "0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0",
            })
            with input_table.open("wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/score_activity_candidates.py"),
                 "--input-table", str(input_table), "--checkpoints", str(checkpoint),
                 "--manifests", str(manifest), "--output", str(output_table)],
                check=True, capture_output=True, text=True,
            )
            with output_table.open("rt", encoding="utf-8") as handle:
                result = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(result["candidate_library_id"], "CAND0001")
            self.assertEqual(result["n_seed_models"], "1")
            self.assertTrue(float(result["sbase"]) == float(result["sbase"]))

    def test_bgc_scoring_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            checkpoint = work / "detector.pt"
            pfam2vec = work / "pfam2vec.csv"
            regions = work / "regions.tsv"
            output_table = work / "scores.tsv"
            torch.save(MultitaskBGCDetector().state_dict(), checkpoint)
            pfam2vec.write_text(
                "pfam_id," + ",".join(f"v{i}" for i in range(102)) + "\n" +
                "PF00668," + ",".join(["0.1"] * 102) + "\n",
                encoding="utf-8",
            )
            regions.write_text("sample_id\tpfam_ids\nR1\tPF00668;PF00668\n", encoding="utf-8")
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/score_bgc_regions.py"),
                 "--pfam2vec", str(pfam2vec), "--checkpoint", str(checkpoint),
                 "--regions", str(regions), "--output", str(output_table)],
                check=True, capture_output=True, text=True,
            )
            with output_table.open("rt", encoding="utf-8") as handle:
                result = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(result["sample_id"], "R1")
            self.assertEqual(len(result["position_probabilities"].split(";")), 2)
            self.assertIn(result["product_class"], {"NRPS", "PKS", "RiPP", "Terpene", "Other"})


if __name__ == "__main__":
    unittest.main()
