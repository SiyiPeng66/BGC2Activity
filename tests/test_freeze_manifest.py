from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bgc2activity.release.freeze import FreezeManifestError, REQUIRED_ROLES, sha256_file, write_freeze_manifest


class FreezeManifestTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> dict[str, Path]:
        artifacts = {}
        for index, role in enumerate(sorted(REQUIRED_ROLES), start=1):
            path = root / f"{index:02d}_{role}.txt"
            path.write_text(f"{role}\n", encoding="utf-8")
            artifacts[role] = path
        return artifacts

    def test_writes_a_complete_content_addressed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, metadata = write_freeze_manifest(
                self._artifacts(root),
                root / "freeze",
                code_commit="abc123",
                study_version="v1.0.0",
                frozen_at_utc=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            with manifest.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual({row["role"] for row in rows}, REQUIRED_ROLES)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))
            self.assertEqual(json.loads(metadata.read_text(encoding="utf-8"))["manifest_sha256"], sha256_file(manifest))

    def test_rejects_an_incomplete_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self._artifacts(root)
            artifacts.pop("round2_candidate_list")
            with self.assertRaisesRegex(FreezeManifestError, "missing roles: round2_candidate_list"):
                write_freeze_manifest(artifacts, root / "freeze", code_commit="abc123", study_version="v1")

    def test_rejects_duplicate_role_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = self._artifacts(root)
            artifacts["calibrated_scores"] = artifacts["supd_scores"]
            with self.assertRaisesRegex(FreezeManifestError, "distinct file"):
                write_freeze_manifest(artifacts, root / "freeze", code_commit="abc123", study_version="v1")
