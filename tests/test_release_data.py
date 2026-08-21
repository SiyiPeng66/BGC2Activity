"""Run the dependency-free release-data audit inside the unittest suite.

These tests validate the committed release tables against the manuscript-fixed
invariants (row counts, uniqueness, funnel stages, hit counts, checksums).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_release_data as audit


class ReleaseDataAuditTests(unittest.TestCase):
    def test_genome_inventory(self):
        audit.audit_genomes()

    def test_bgc_indices(self):
        audit.audit_bgc_indices()

    def test_candidate_library(self):
        audit.audit_candidate_library()

    def test_library_funnel(self):
        audit.audit_funnel()

    def test_screening(self):
        audit.audit_screening()

    def test_checksums(self):
        audit.audit_checksums()


if __name__ == "__main__":
    unittest.main()
