from __future__ import annotations

import unittest

from bgc2activity.candidates.audit import CandidateLibraryError, LibraryExpectations, audit_rows


def row(candidate_id: str, parent_bgc_id: str, core: str, displayed: str, lipidated: str = "false"):
    return {
        "candidate_id": candidate_id,
        "parent_bgc_id": parent_bgc_id,
        "genome_accession": "GCF_000000001.1",
        "contig_id": "contig_1",
        "region_start": "1",
        "region_end": "5000",
        "a_domain_coordinates": "101..300;901..1100;1701..1900;2501..2700;3301..3500",
        "predicted_substrates": "ala;val;leu;orn;dab",
        "module_count": "5",
        "canonical_core_sequence": core,
        "displayed_sequence": displayed,
        "c_terminal": "OH",
        "variant_class": "direct_concatenation",
        "variant_detail": "base_sequence",
        "reconstruction_confidence": "0.80",
        "is_nonstandard_or_modified": "true",
        "is_lipidated": lipidated,
    }


class CandidateLibraryAuditTests(unittest.TestCase):
    def test_accepts_a_complete_custom_contract(self):
        rows = [
            row("BNP_000001", "GCF_000000001.1:contig_1:1-5000", "A-B-C-D-E", "H-A-B-C-D-E-OH"),
            row("BNP_000002", "GCF_000000002.1:contig_1:1-5000", "F-G-H-I-J", "H-F-G-H-I-J-NH2", "true"),
        ]
        expected = LibraryExpectations(
            records=2,
            parent_bgcs=2,
            core_sequences=2,
            displayed_sequences=2,
            nonstandard_or_modified=2,
            lipidated=1,
        )
        self.assertEqual(audit_rows(rows, expected=expected)["parent_bgcs"], 2)

    def test_rejects_abbreviated_parent_identifier(self):
        bad = row("BNP_000001", "GCF_000000001.1__c00001_GCF_000...region001", "A-B-C-D-E", "H-A-B-C-D-E-OH")
        with self.assertRaisesRegex(CandidateLibraryError, "abbreviated"):
            audit_rows([bad], expected=LibraryExpectations(records=1, parent_bgcs=1, core_sequences=1, displayed_sequences=1, nonstandard_or_modified=1, lipidated=0))

    def test_rejects_an_underlength_candidate(self):
        bad = row("BNP_000001", "GCF_000000001.1:contig_1:1-5000", "A-B-C-D", "H-A-B-C-D-OH")
        bad["module_count"] = "4"
        with self.assertRaisesRegex(CandidateLibraryError, "at least 5"):
            audit_rows([bad], expected=LibraryExpectations(records=1, parent_bgcs=1, core_sequences=1, displayed_sequences=1, nonstandard_or_modified=1, lipidated=0))

    def test_reports_contract_count_mismatch(self):
        with self.assertRaisesRegex(CandidateLibraryError, "parent_bgcs: observed 1, expected 2"):
            audit_rows(
                [row("BNP_000001", "GCF_000000001.1:contig_1:1-5000", "A-B-C-D-E", "H-A-B-C-D-E-OH")],
                expected=LibraryExpectations(records=1, parent_bgcs=2, core_sequences=1, displayed_sequences=1, nonstandard_or_modified=1, lipidated=0),
            )
