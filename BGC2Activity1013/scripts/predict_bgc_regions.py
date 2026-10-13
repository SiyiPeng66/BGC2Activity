#!/usr/bin/env python3
"""Run the reconstructed BGC detector on ordered Pfam-region TSV rows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgc2activity_reconstructed.detection.data import RegionSample, collate_regions, load_pfam2vec  # noqa: E402
from bgc2activity_reconstructed.detection.model import MultitaskBGCDetector  # noqa: E402
from bgc2activity_reconstructed.detection.regions import (  # noqa: E402
    WINDOW_LENGTH, POSITION_THRESHOLD, window_starts,
    aggregate_window_probabilities, aggregate_window_class_probabilities,
)

CLASSES = ("NRPS", "PKS", "RiPP", "Terpene", "Other")


def parse_region(row):
    raw = row["pfam_ids"]
    if not raw.strip():
        raise ValueError(f"{row['sample_id']}: no Pfam positions")
    # Preserve missing-domain slots as OOV, matching training collation.
    # Deleting them shifts all later position probabilities.
    pfams = tuple(token if token else "<OOV>" for token in raw.split(";"))
    return RegionSample(row["sample_id"], pfams, (0,) * len(pfams), 0, 0)


@torch.no_grad()
def predict_region(model, sample, vocabulary, matrix):
    """Score every position using 256-position windows with 128-position stride."""
    if not sample.pfam_ids:
        raise ValueError(f"{sample.sample_id}: no Pfam positions")
    positions, classes = [], []
    for start in window_starts(len(sample.pfam_ids)):
        pfams = sample.pfam_ids[start:start + WINDOW_LENGTH]
        window = RegionSample(sample.sample_id, pfams, (0,) * len(pfams), 0, 0)
        batch = collate_regions([window], vocabulary, matrix)
        output = model(batch["embeddings"], batch["valid"])
        positions.append((start, torch.sigmoid(output.detection_logits[0]).cpu().numpy()))
        classes.append((torch.softmax(output.class_logits[0], dim=-1).cpu().numpy(), len(pfams)))
    scores = aggregate_window_probabilities(positions, len(sample.pfam_ids))
    class_scores = aggregate_window_class_probabilities(classes)
    return {
        "sample_id": sample.sample_id,
        "max_bgc_probability": float(scores.max()),
        "positive_position_fraction": float((scores >= POSITION_THRESHOLD).mean()),
        "product_class": CLASSES[int(class_scores.argmax())],
        "position_probabilities": ";".join(f"{value:.8g}" for value in scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfam2vec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vocabulary, matrix_np = load_pfam2vec(args.pfam2vec)
    matrix = torch.as_tensor(matrix_np, dtype=torch.float32)
    model = MultitaskBGCDetector(pfam2vec_dimension=matrix.shape[1])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    rows = []
    with args.regions.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(parse_region(row))
    if not rows:
        raise SystemExit("regions table contains no rows")

    output_rows = [predict_region(model, sample, vocabulary, matrix) for sample in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} BGC predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
