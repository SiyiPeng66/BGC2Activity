"""Create a content-addressed prospective-screening freeze manifest.

Supplementary Method 17 requires the Round-2 selection package to record frozen
model/configuration/data/score/selection artifacts, UTC timestamps, code commits,
runtime information and SHA-256 checksums.  This module records metadata only;
it never changes the supplied scientific artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


REQUIRED_ROLES = frozenset(
    {
        "model_checkpoint",
        "configuration",
        "round1_labels",
        "candidate_features",
        "sbase_scores",
        "supd_scores",
        "calibrated_scores",  # calibrated predicted log10(MIC)
        "selection_rules",
        "round2_candidate_list",
        "runtime_environment",
    }
)


class FreezeManifestError(ValueError):
    """Raised when a prospective freeze package is incomplete or ambiguous."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_artifacts(artifacts: Mapping[str, str | Path]) -> dict[str, Path]:
    provided = set(artifacts)
    missing = REQUIRED_ROLES - provided
    unexpected = provided - REQUIRED_ROLES
    if missing or unexpected:
        fragments = []
        if missing:
            fragments.append("missing roles: " + ", ".join(sorted(missing)))
        if unexpected:
            fragments.append("unexpected roles: " + ", ".join(sorted(unexpected)))
        raise FreezeManifestError("; ".join(fragments))

    resolved = {role: Path(value).resolve() for role, value in artifacts.items()}
    absent = [f"{role}={path}" for role, path in resolved.items() if not path.is_file()]
    if absent:
        raise FreezeManifestError("artifact files not found: " + ", ".join(absent))
    if len(set(resolved.values())) != len(resolved):
        raise FreezeManifestError("each freeze role must refer to a distinct file")
    return resolved


def write_freeze_manifest(
    artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    code_commit: str,
    study_version: str,
    frozen_at_utc: datetime | None = None,
) -> tuple[Path, Path]:
    """Write a TSV manifest and JSON metadata for exactly ten frozen artifacts.

    The caller supplies a code commit and study version because this function has
    no authority to infer provenance from an arbitrary working tree.
    """

    if not code_commit.strip():
        raise FreezeManifestError("code_commit is required")
    if not study_version.strip():
        raise FreezeManifestError("study_version is required")
    files = _validate_artifacts(artifacts)
    timestamp = frozen_at_utc or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise FreezeManifestError("frozen_at_utc must include a timezone")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    manifest_path = target / "SHA256SUMS.tsv"
    metadata_path = target / "freeze_metadata.json"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("role", "source_path", "bytes", "sha256"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for role in sorted(files):
            path = files[role]
            writer.writerow(
                {
                    "role": role,
                    "source_path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    metadata = {
        "schema_version": "1.0",
        "study_version": study_version,
        "code_commit": code_commit,
        "frozen_at_utc": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifact_count": len(files),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, metadata_path
