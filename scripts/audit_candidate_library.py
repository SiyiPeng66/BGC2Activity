#!/usr/bin/env python3
"""Validate a normalized candidate-library TSV against Supplementary Method 13."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse
import json

from bgc2activity.candidates import CandidateLibraryError, audit_candidate_library


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_library_tsv", help="UTF-8, tab-delimited normalized candidate library")
    args = parser.parse_args()
    try:
        print(json.dumps(audit_candidate_library(args.candidate_library_tsv), indent=2, sort_keys=True))
    except CandidateLibraryError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
