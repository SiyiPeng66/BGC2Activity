#!/usr/bin/env python3
"""Write the ten-role prospective-screening freeze manifest required by Method 17."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import argparse
from pathlib import Path

from bgc2activity.release import FreezeManifestError, write_freeze_manifest
from bgc2activity.release.freeze import REQUIRED_ROLES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--study-version", required=True)
    for role in sorted(REQUIRED_ROLES):
        parser.add_argument(f"--{role.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    artifacts = {role: getattr(args, role) for role in REQUIRED_ROLES}
    try:
        manifest, metadata = write_freeze_manifest(
            artifacts,
            args.output_dir,
            code_commit=args.code_commit,
            study_version=args.study_version,
        )
    except FreezeManifestError as exc:
        parser.error(str(exc))
    print(manifest)
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
