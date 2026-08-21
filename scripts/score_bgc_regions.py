#!/usr/bin/env python3
"""Run the frozen BGC detector on ordered Pfam-region sequences.

Input TSV columns: ``sample_id`` and ``pfam_ids`` (semicolon-separated,
ordered by genomic coordinate). The command writes the maximum position-level
NRPS/BGC probability, the fraction of positions at or above the manuscript
threshold 0.45, the predicted product class and the complete position-score
vector. It is the model-inference counterpart to
``scripts/train_bgc_detector.py``; genome FASTA annotation is performed first
with ``bgc2activity.detection.annotation`` when using raw genomes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from bgc2activity.detection.data import RegionSample, collate_regions, load_pfam2vec
from bgc2activity.detection.model import DETECTION_THRESHOLD, MultitaskBGCDetector
from bgc2activity.detection.regions import (
    WINDOW_LENGTH,
    aggregate_window_class_probabilities,
    aggregate_window_probabilities,
    window_starts,
)

PRODUCT_CLASSES = ("NRPS", "PKS", "RiPP", "Terpene", "Other")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfam2vec", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    vocabulary, matrix = load_pfam2vec(args.pfam2vec)
    embedding_matrix = torch.as_tensor(matrix, dtype=torch.float32)
    rows = []
    with Path(args.regions).open("rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pfam_ids = tuple(token for token in row["pfam_ids"].split(";") if token)
            rows.append(RegionSample(row["sample_id"], pfam_ids, tuple([0] * len(pfam_ids)), 0, 0))
    if not rows:
        raise SystemExit("regions table contains no rows")

    model = MultitaskBGCDetector()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    output_rows = []
    with torch.no_grad():
        for sample in rows:
            probability_windows = []
            class_windows = []
            for window_start in window_starts(len(sample.pfam_ids)):
                window_pfams = sample.pfam_ids[window_start:window_start + WINDOW_LENGTH]
                window = RegionSample(sample.sample_id, window_pfams, tuple([0] * len(window_pfams)), 0, 0)
                batch = collate_regions([window], vocabulary, embedding_matrix)
                output = model(batch["embeddings"], batch["valid"])
                valid = batch["valid"][0].bool()
                position_scores = torch.sigmoid(output.detection_logits[0][valid]).cpu().numpy()
                class_scores = torch.softmax(output.class_logits[0], dim=-1).cpu().numpy()
                probability_windows.append((window_start, position_scores))
                class_windows.append((class_scores, len(window_pfams)))
            scores_array = aggregate_window_probabilities(probability_windows, len(sample.pfam_ids))
            region_class_probabilities = aggregate_window_class_probabilities(class_windows)
            scores = scores_array.tolist()
            class_index = int(region_class_probabilities.argmax())
            output_rows.append({
                "sample_id": sample.sample_id,
                "max_bgc_probability": max(scores, default=0.0),
                "positive_position_fraction": sum(score >= DETECTION_THRESHOLD for score in scores) / max(len(scores), 1),
                "product_class": PRODUCT_CLASSES[class_index],
                "position_probabilities": ";".join(f"{score:.8g}" for score in scores),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} BGC-region scores to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
