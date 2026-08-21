#!/usr/bin/env python3
"""Apply the frozen activity-independent selection stack (Methods 14, 17).

Reads a score-ranked candidate table and writes the selected cohort plus the
rejection ledger. The same command serves Round 1 (Base Model scores) and
Round 2 (Calibrated Model scores, with Round-1 candidates excluded via
--exclude).

Input TSV columns: candidate_id, score, chemical_entity_id,
core_sequence_cluster_id, parent_bgc_id, synthesis_feasible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bgc2activity.screening.selection import SelectionConstraints, select_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-candidates", required=True)
    parser.add_argument("--exclude", default=None,
                        help="TSV with candidate_id column already selected (e.g. Round 1)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-cluster", type=int, default=2)
    parser.add_argument("--max-per-bgc", type=int, default=1)
    parser.add_argument("--cohort-size", type=int, default=40)
    parser.add_argument("--longlist-size", type=int, default=100)
    args = parser.parse_args()

    pool = pd.read_csv(args.ranked_candidates, sep="\t")
    excluded: frozenset[str] = frozenset()
    if args.exclude:
        excluded = frozenset(pd.read_csv(args.exclude, sep="\t")["candidate_id"].astype(str))

    constraints = SelectionConstraints(
        max_per_core_sequence_cluster=args.max_per_cluster,
        max_per_bgc=args.max_per_bgc,
    )
    record = select_cohort(pool, constraints, cohort_size=args.cohort_size,
                           longlist_size=args.longlist_size, excluded_candidate_ids=excluded)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["selection_order", "candidate_id"])
        for order, candidate_id in enumerate(record.selected, start=1):
            writer.writerow([order, candidate_id])
    ledger = output.with_name(output.stem + "_rejections.tsv")
    with ledger.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["reason", "candidate_id"])
        for reason, ids in sorted(record.rejected.items()):
            for candidate_id in ids:
                writer.writerow([reason, candidate_id])
    print(f"selected {len(record.selected)} candidates -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
