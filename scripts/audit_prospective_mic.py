#!/usr/bin/env python3
"""Validate the normalized Round-1/Round-2 MIC table against the final endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse
import json

from bgc2activity.screening import MicAuditError, audit_prospective_mic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prospective_mic_tsv", help="UTF-8, tab-delimited prospective MIC table")
    args = parser.parse_args()
    try:
        print(json.dumps(audit_prospective_mic(args.prospective_mic_tsv), indent=2, sort_keys=True))
    except MicAuditError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
