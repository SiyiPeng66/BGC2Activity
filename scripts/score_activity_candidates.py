#!/usr/bin/env python3
"""Score candidate peptides with the frozen Base MIC Regression ensemble.

The command loads one or more seed checkpoints plus their training manifests,
applies each seed's frozen training-set preprocessing, and writes the ensemble
mean predicted log10(MIC) (Sbase; lower is more active) and inter-model
standard deviation. Input rows require no MIC labels.

Input TSV columns:
  candidate_library_id, monomers, descriptors,
  length, molecular_weight, net_charge_pH7, GRAVY, aromatic_fraction,
  nonstandard_monomer_fraction, lipidation_descriptor,
  n_terminal, c_terminal, lipidation, cyclization.

``monomers`` and ``descriptors`` use the formats documented by
``train_activity_model.py``. Checkpoints and manifests must be supplied in
matching order (normally seeds 42-46).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from bgc2activity.activity.base_model import CATEGORICAL_LEVELS, CONTINUOUS_FEATURES, BaseActivityModel
from bgc2activity.activity.training import move_to_device
from bgc2activity.representation.pretraining import DESCRIPTOR_DIMENSION, ChemicalAwarePretrainer, blake2b_text_vector


def parse_candidate(row: dict, feature_spec: dict) -> dict:
    monomers = row["monomers"].split(";")
    descriptor_groups = []
    for group in row["descriptors"].split(";"):
        if group.strip().upper() == "NA":
            descriptor_groups.append([float("nan")] * DESCRIPTOR_DIMENSION)
        else:
            values = [float(value) for value in group.split(",")]
            if len(values) != DESCRIPTOR_DIMENSION:
                raise ValueError(f"{row['candidate_library_id']}: descriptor groups must contain 8 values")
            descriptor_groups.append(values)
    if len(monomers) != len(descriptor_groups):
        raise ValueError(f"{row['candidate_library_id']}: monomer/descriptor length mismatch")

    continuous, missingness = [], []
    for feature in CONTINUOUS_FEATURES:
        raw = (row.get(feature) or "").strip()
        missing = raw == "" or raw.upper() == "NA"
        value = feature_spec["medians"][feature] if missing else float(raw)
        value = (value - feature_spec["means"][feature]) / feature_spec["standard_deviations"][feature]
        continuous.append(value)
        missingness.append(float(missing))
    return {
        "text_vectors": torch.stack([blake2b_text_vector(monomer) for monomer in monomers]).unsqueeze(0),
        "descriptors": torch.tensor(descriptor_groups, dtype=torch.float32).unsqueeze(0),
        "attention_mask": torch.ones(1, len(monomers)),
        "continuous": torch.tensor([continuous], dtype=torch.float32),
        "missingness": torch.tensor([missingness], dtype=torch.float32),
        "categorical": {
            name: torch.tensor([int(row[name])], dtype=torch.long) for name in CATEGORICAL_LEVELS
        },
    }


def impute_descriptor_nans(batch: dict) -> None:
    descriptors = batch["descriptors"]
    batch["descriptors"] = torch.nan_to_num(descriptors, nan=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--checkpoints", required=True, nargs="+")
    parser.add_argument("--manifests", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.manifests):
        parser.error("--checkpoints and --manifests must contain the same number of paths")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    specs = []
    for checkpoint_path, manifest_path in zip(args.checkpoints, args.manifests):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        model = BaseActivityModel(ChemicalAwarePretrainer())
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model.to(device).eval()
        models.append(model)
        specs.append(manifest["feature_spec"])

    with Path(args.input_table).open("rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    output_rows = []
    with torch.no_grad():
        for row in rows:
            seed_predictions = []
            for model, feature_spec in zip(models, specs):
                batch = parse_candidate(row, feature_spec)
                impute_descriptor_nans(batch)
                seed_predictions.append(float(model(**move_to_device(batch, device)).item()))
            tensor = torch.tensor(seed_predictions)
            output_rows.append({
                "candidate_library_id": row["candidate_library_id"],
                "sbase": float(tensor.mean()),
                "sbase_uncertainty": float(tensor.std(unbiased=False)),
                "n_seed_models": len(seed_predictions),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys() if output_rows else
                                ["candidate_library_id", "sbase", "sbase_uncertainty", "n_seed_models"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} candidate scores to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
