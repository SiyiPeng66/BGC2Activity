#!/usr/bin/env python3
"""Validate the full NRP45 derivative library against the Fig. 4 contract."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse
import json

from bgc2activity.optimization import DerivativeLibraryError, audit_derivative_library


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derivative_library_tsv", help="UTF-8, tab-delimited NRP45 derivative library")
    args = parser.parse_args()
    try:
        print(json.dumps(audit_derivative_library(args.derivative_library_tsv), indent=2, sort_keys=True))
    except DerivativeLibraryError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
