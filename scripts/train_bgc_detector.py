#!/usr/bin/env python3
"""Train the multitask BGC detector under the manuscript-fixed protocol.

Entry point for Supplementary Methods 3-7. All inputs are explicit CLI paths
(no legacy defaults). The protocol is frozen by the final manuscript: AdamW
(lr 5e-4, weight decay 1e-3), gradient clipping 1.0, batch size 64, up to 80
epochs with early stopping after 10 epochs without validation-AUPRC
improvement, seed 42.

Input sample TSV columns:
  sample_id, family_label, class_label, pfam_ids (';'-separated),
  position_labels (';'-separated 0/1)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from bgc2activity.detection.data import PfamSequenceDataset, RegionSample, collate_regions, load_pfam2vec
from bgc2activity.detection.losses import multitask_loss
from bgc2activity.detection.model import MultitaskBGCDetector, load_balancing_loss

PROTOCOL = {
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "gradient_clip": 1.0,
    "batch_size": 64,
    "max_epochs": 80,
    "early_stopping_patience": 10,
    "selection_criterion": "validation_auprc",
    "seed": 42,
}


def load_samples(path: Path) -> list[RegionSample]:
    samples: list[RegionSample] = []
    with path.open("rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            samples.append(
                RegionSample(
                    sample_id=row["sample_id"],
                    pfam_ids=tuple(row["pfam_ids"].split(";")),
                    position_labels=tuple(int(v) for v in row["position_labels"].split(";")),
                    class_label=int(row["class_label"]),
                    family_label=int(row["family_label"]),
                )
            )
    return samples


def average_precision(scores: list[float], labels: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfam2vec", required=True, help="frozen pfam2vec CSV")
    parser.add_argument("--train-samples", required=True, help="training sample TSV")
    parser.add_argument("--validation-samples", required=True, help="validation sample TSV")
    parser.add_argument("--output-dir", required=True, help="checkpoint and manifest output directory")
    args = parser.parse_args()

    torch.manual_seed(PROTOCOL["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary, matrix = load_pfam2vec(args.pfam2vec)
    embedding_matrix = torch.as_tensor(matrix, dtype=torch.float32)
    training = load_samples(Path(args.train_samples))
    validation = load_samples(Path(args.validation_samples))
    train_loader = DataLoader(
        PfamSequenceDataset(training, vocabulary, matrix),
        batch_size=PROTOCOL["batch_size"], shuffle=True,
        collate_fn=lambda batch: collate_regions(batch, vocabulary, embedding_matrix),
    )

    model = MultitaskBGCDetector()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])

    best_auprc = -1.0
    patience = 0
    for epoch in range(PROTOCOL["max_epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch["embeddings"], batch["valid"])
            losses = multitask_loss(
                output.detection_logits, batch["position_labels"], batch["valid"],
                output.class_logits, batch["class_labels"],
                output.region_representation, batch["family_labels"],
                load_balancing_loss(output.router_probabilities, output.expert_fraction),
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip"])
            optimizer.step()

        model.eval()
        scores: list[float] = []
        labels: list[int] = []
        with torch.no_grad():
            for start in range(0, len(validation), PROTOCOL["batch_size"]):
                batch = collate_regions(validation[start:start + PROTOCOL["batch_size"]],
                                        vocabulary, embedding_matrix)
                output = model(batch["embeddings"], batch["valid"])
                probabilities = torch.sigmoid(output.detection_logits)
                for row in range(probabilities.shape[0]):
                    valid_count = int(batch["valid"][row].sum())
                    scores.extend(probabilities[row, :valid_count].tolist())
                    labels.extend(batch["position_labels"][row, :valid_count].int().tolist())
        auprc = average_precision(scores, labels)
        print(f"epoch {epoch + 1:03d} validation AUPRC {auprc:.4f}")
        if auprc > best_auprc + 1e-6:
            best_auprc = auprc
            patience = 0
            torch.save(model.state_dict(), output_dir / "detector_checkpoint.pt")
        else:
            patience += 1
            if patience >= PROTOCOL["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch + 1}")
                break

    manifest = {
        "protocol": PROTOCOL,
        "best_validation_auprc": best_auprc,
        "n_train_samples": len(training),
        "n_validation_samples": len(validation),
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote checkpoint and manifest to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
