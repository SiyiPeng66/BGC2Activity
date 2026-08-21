#!/usr/bin/env python3
"""Train one seed of the Base MIC Regression Model under the frozen protocol.

Entry point for the main-text activity model (section 2.3; freeze boundary,
chemical-feature fusion and optimiser protocol from Supplementary Method 12).
The model loads the chemical-aware pretraining checkpoint, freezes the
monomer projections, positional encodings and the first two Transformer
layers, fine-tunes layer 3, the output LN and the candidate pooling module,
and trains a scalar regression head on the continuous log10(MIC) target with
the Huber loss. Model selection uses validation RMSE (ties broken by higher
validation Spearman); early stopping after 12 epochs without improvement.
Run once per seed in ENSEMBLE_SEEDS (42-46); Sbase is the arithmetic mean of
the five seed predictions (see bgc2activity.activity.training).

Input table TSV columns (one row per strictly curated entity):
  entity_id, partition (train|validation|test),
  monomers (';'-separated), descriptors (';'-separated groups of 8
  comma-separated values; 'NA' groups are median-imputed per batch column),
  log10_mic_target,
  the seven continuous chemistry features (length, molecular_weight,
  net_charge_pH7, GRAVY, aromatic_fraction, nonstandard_monomer_fraction,
  lipidation_descriptor; missing as empty/NA),
  and integer codes for n_terminal (0-3), c_terminal (0-2), lipidation (0-2),
  cyclization (0-2), with the highest code denoting the explicit unknown level.

Imputation and standardisation statistics are computed on the training split
only and remain frozen for validation/test (Method 12).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, Dataset

from bgc2activity.activity.base_model import (
    CATEGORICAL_LEVELS,
    CONTINUOUS_FEATURES,
    BaseActivityModel,
)
from bgc2activity.activity.evaluation import root_mean_squared_error, spearman
from bgc2activity.activity.training import BATCH_SIZE, MAX_EPOCHS, move_to_device, train_one_seed
from bgc2activity.representation.pretraining import (
    DESCRIPTOR_DIMENSION,
    ChemicalAwarePretrainer,
    blake2b_text_vector,
)

PROTOCOL = {
    "target": "log10_mic",
    "loss": "huber (smooth_l1, delta=1.0)",
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "lr_fine_tuned": 1e-5,
    "lr_new_layers": 3e-4,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "gradient_clip": 1.0,
    "selection": "validation_rmse (ties: higher validation spearman)",
    "early_stopping_patience": 12,
    "ensemble_seeds": [42, 43, 44, 45, 46],
}


class ActivityEntity:
    def __init__(self, row: dict):
        self.entity_id = row["entity_id"]
        self.monomers = row["monomers"].split(";")
        self.descriptors = []
        for group in row["descriptors"].split(";"):
            if group.strip().upper() == "NA":
                self.descriptors.append([float("nan")] * DESCRIPTOR_DIMENSION)
            else:
                self.descriptors.append([float(value) for value in group.split(",")])
        if len(self.descriptors) != len(self.monomers):
            raise ValueError(f"{self.entity_id}: monomer/descriptor length mismatch")
        self.target = float(row["log10_mic_target"])
        self.continuous = []
        self.missingness = []
        for feature in CONTINUOUS_FEATURES:
            raw = (row.get(feature) or "").strip()
            if raw == "" or raw.upper() == "NA":
                self.continuous.append(float("nan"))
                self.missingness.append(1.0)
            else:
                self.continuous.append(float(raw))
                self.missingness.append(0.0)
        self.categorical = {name: int(row[name]) for name in CATEGORICAL_LEVELS}


def load_table(path: Path) -> dict[str, list[ActivityEntity]]:
    partitions: dict[str, list[ActivityEntity]] = {"train": [], "validation": [], "test": []}
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            partitions[row["partition"]].append(ActivityEntity(row))
    return partitions


class ActivityDataset(Dataset):
    def __init__(self, entities: list[ActivityEntity], feature_spec: dict):
        self.entities = entities
        self.feature_spec = feature_spec
        self.text_vectors = [
            torch.stack([blake2b_text_vector(monomer) for monomer in entity.monomers])
            for entity in entities
        ]

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        entity = self.entities[index]
        spec = self.feature_spec
        continuous = []
        for feature, value in zip(CONTINUOUS_FEATURES, entity.continuous):
            if value != value:  # NaN: impute with the frozen training median
                value = spec["medians"][feature]
            continuous.append((value - spec["means"][feature]) / spec["standard_deviations"][feature])
        return {
            "text_vectors": self.text_vectors[index],
            "descriptors": torch.tensor(entity.descriptors, dtype=torch.float32),
            "continuous": torch.tensor(continuous, dtype=torch.float32),
            "missingness": torch.tensor(entity.missingness, dtype=torch.float32),
            "categorical": {name: torch.tensor(entity.categorical[name], dtype=torch.long)
                            for name in CATEGORICAL_LEVELS},
        }, torch.tensor(entity.target, dtype=torch.float32)


def collate_activity(batch):
    items, targets = zip(*batch)
    length = max(item["text_vectors"].shape[0] for item in items)
    text_dim = items[0]["text_vectors"].shape[1]
    padded_text = torch.zeros(len(items), length, text_dim)
    padded_descriptors = torch.zeros(len(items), length, DESCRIPTOR_DIMENSION)
    attention_mask = torch.zeros(len(items), length)
    for row, item in enumerate(items):
        n = item["text_vectors"].shape[0]
        padded_text[row, :n] = item["text_vectors"]
        padded_descriptors[row, :n] = item["descriptors"]
        attention_mask[row, :n] = 1.0
    missing = torch.isnan(padded_descriptors)
    if missing.any():
        column_medians = torch.nanmedian(padded_descriptors, dim=0).values
        column_medians = torch.where(torch.isnan(column_medians), torch.zeros_like(column_medians), column_medians)
        padded_descriptors = torch.where(missing, column_medians.expand_as(padded_descriptors), padded_descriptors)
    batch_dict = {
        "text_vectors": padded_text,
        "descriptors": padded_descriptors,
        "attention_mask": attention_mask,
        "continuous": torch.stack([item["continuous"] for item in items]),
        "missingness": torch.stack([item["missingness"] for item in items]),
        "categorical": {name: torch.stack([item["categorical"][name] for item in items])
                        for name in CATEGORICAL_LEVELS},
    }
    return batch_dict, torch.stack(targets)


def compute_feature_spec(entities: list[ActivityEntity]) -> dict:
    medians, means, standard_deviations = {}, {}, {}
    for feature_index, feature in enumerate(CONTINUOUS_FEATURES):
        values = [entity.continuous[feature_index] for entity in entities
                  if entity.continuous[feature_index] == entity.continuous[feature_index]]
        tensor = torch.tensor(values, dtype=torch.float32)
        medians[feature] = float(torch.median(tensor)) if len(values) else 0.0
        means[feature] = float(tensor.mean()) if len(values) else 0.0
        standard_deviations[feature] = float(tensor.std()) if len(values) > 1 else 1.0
        if standard_deviations[feature] == 0.0:
            standard_deviations[feature] = 1.0
    return {"medians": medians, "means": means, "standard_deviations": standard_deviations}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", required=True, help="entity table with partition column")
    parser.add_argument("--seed", type=int, default=42, help="one of the five ensemble seeds")
    parser.add_argument("--pretrainer-checkpoint", default=None,
                        help="optional ChemicalAwarePretrainer state dict to initialise from")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partitions = load_table(Path(args.input_table))
    if not partitions["train"] or not partitions["validation"]:
        raise SystemExit("input table must contain train and validation partitions")
    feature_spec = compute_feature_spec(partitions["train"])

    train_dataset = ActivityDataset(partitions["train"], feature_spec)
    validation_dataset = ActivityDataset(partitions["validation"], feature_spec)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_activity)

    pretrainer = ChemicalAwarePretrainer()
    if args.pretrainer_checkpoint:
        pretrainer.load_state_dict(
            torch.load(args.pretrainer_checkpoint, map_location="cpu"), strict=False,
        )
    model = BaseActivityModel(pretrainer)
    model.freeze_for_supervised_training()

    def train_batches():
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_activity,
                            generator=torch.Generator().manual_seed(args.seed))
        for batch_dict, targets in loader:
            yield batch_dict, targets

    def evaluate(model):
        model.eval()
        observed, predicted = [], []
        for batch_dict, targets in validation_loader:
            predictions = model(**move_to_device(batch_dict, device))
            observed.extend(targets.tolist())
            predicted.extend(predictions.cpu().tolist())
        return (root_mean_squared_error(observed, predicted),
                spearman(observed, predicted))

    model, history = train_one_seed(model, train_batches, evaluate, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"base_regression_model_seed{args.seed}.pt")
    manifest = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "feature_spec": feature_spec,
        "history": [record.__dict__ for record in history],
    }
    (output_dir / f"training_manifest_seed{args.seed}.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote seed-{args.seed} checkpoint and manifest to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
