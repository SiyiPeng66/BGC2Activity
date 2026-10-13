#!/usr/bin/env python3
"""Train the reconstructed multitask BGC detector."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgc2activity_reconstructed.detection.data import (  # noqa: E402
    PfamSequenceDataset,
    RegionSample,
    collate_regions,
    load_pfam2vec,
)
from bgc2activity_reconstructed.detection.losses import multitask_loss  # noqa: E402
from bgc2activity_reconstructed.detection.model import (  # noqa: E402
    MultitaskBGCDetector,
    load_balancing_loss,
)


def load_samples(path: Path, limit: int | None = None) -> list[RegionSample]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(RegionSample(
                sample_id=row["sample_id"],
                pfam_ids=tuple(row["pfam_ids"].split(";")),
                position_labels=tuple(int(v) for v in row["position_labels"].split(";")),
                class_label=int(row["class_label"]),
                family_label=int(row["family_label"]),
            ))
    return rows if limit is None else rows[:limit]


def average_precision(scores: list[float], labels: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    positives = sum(labels)
    if positives == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positives


def evaluate(model, samples, vocabulary, matrix):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for start in range(0, len(samples), 32):
            batch = collate_regions(samples[start:start + 32], vocabulary, matrix)
            output = model(batch["embeddings"], batch["valid"])
            probabilities = torch.sigmoid(output.detection_logits)
            for row in range(probabilities.shape[0]):
                n = int(batch["valid"][row].sum())
                scores.extend(probabilities[row, :n].tolist())
                labels.extend(batch["position_labels"][row, :n].int().tolist())
    return average_precision(scores, labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfam2vec", type=Path, required=True)
    parser.add_argument("--train-samples", type=Path, required=True)
    parser.add_argument("--validation-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    vocabulary, matrix_np = load_pfam2vec(args.pfam2vec)
    matrix = torch.as_tensor(matrix_np, dtype=torch.float32)
    train = load_samples(args.train_samples, args.limit_train)
    validation = load_samples(args.validation_samples, args.limit_validation)
    if not train or not validation:
        raise SystemExit("training and validation tables must contain rows")

    loader = DataLoader(
        PfamSequenceDataset(train, vocabulary, matrix_np),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda batch: collate_regions(batch, vocabulary, matrix),
    )
    model = MultitaskBGCDetector(pfam2vec_dimension=matrix.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    best = -1.0
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch["embeddings"], batch["valid"])
            losses = multitask_loss(
                output.detection_logits, batch["position_labels"], batch["valid"],
                output.class_logits, batch["class_labels"],
                output.region_representation, batch["family_labels"],
                load_balancing_loss(output.router_probabilities, output.expert_fraction),
            )
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        auprc = evaluate(model, validation, vocabulary, matrix)
        history.append({"epoch": epoch + 1, "validation_auprc": auprc})
        print(f"epoch {epoch + 1:03d} validation AUPRC {auprc:.4f}")
        if auprc > best + 1e-6:
            best, stale = auprc, 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output_dir / "detector_checkpoint.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": {
            "optimizer": "AdamW",
            "learning_rate": 5e-4,
            "weight_decay": 1e-3,
            "batch_size": args.batch_size,
            "gradient_clip": 1.0,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "selection_metric": "validation_AUPRC",
            "seed": args.seed,
        },
        "pfam2vec_dimension": int(matrix.shape[1]),
        "n_train": len(train),
        "n_validation": len(validation),
        "best_validation_auprc": best,
        "history": history,
    }
    (args.output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote detector checkpoint and manifest to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
