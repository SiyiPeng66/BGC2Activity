#!/usr/bin/env python3
"""Check a feature-table header against the frozen feedback-update input contract."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse
import csv
from pathlib import Path

from bgc2activity.scoring_contract import REQUIRED_COLUMNS, ScoreInputError, validate_update_feature_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_table_tsv", type=Path)
    args = parser.parse_args()
    with args.feature_table_tsv.open(encoding="utf-8", newline="") as handle:
        header = set(csv.DictReader(handle, delimiter="\t").fieldnames or ())
    missing = REQUIRED_COLUMNS - header
    if missing:
        parser.error("missing required columns: " + ", ".join(sorted(missing)))
    try:
        selected = validate_update_feature_selection(("sbase", "length", "net_charge_pH7", "GRAVY", "aromatic_fraction", "nonstandard_monomer_fraction", "is_lipidated", "is_c_terminal_amidated", "molecular_weight", "structural_missingness_indicator"))
    except ScoreInputError as exc:
        parser.error(str(exc))
    print("approved activity update inputs: " + ", ".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
