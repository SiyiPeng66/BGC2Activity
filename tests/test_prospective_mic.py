from __future__ import annotations

import unittest

from bgc2activity.screening.mic import MicAuditError, ScreeningExpectations, summarize_rows


ORGANISMS = ["S. aureus", "E. coli", "Salmonella sp.", "P. aeruginosa", "A. baumannii"]


def candidate_rows(candidate_id: str, library_id: str, screening_round: str, minimum: str):
    values = [minimum, ">256", ">256", ">256", ">256"]
    return [
        {
            "candidate_id": candidate_id,
            "candidate_library_id": library_id,
            "screening_round": screening_round,
            "sequence": f"H-{candidate_id}-OH",
            "test_organism": organism,
            "MIC_ug_mL": value,
            "active_at_MIC_le_256": "true" if value != ">256" else "false",
        }
        for organism, value in zip(ORGANISMS, values)
    ]


class ProspectiveMicTests(unittest.TestCase):
    def test_summarizes_custom_two_round_endpoints(self):
        rows = candidate_rows("R1_01", "BNP_000001", "Round 1", "64")
        rows += candidate_rows("R2_01", "BNP_000002", "Round 2", "8")
        expected = ScreeningExpectations(
            candidates_per_round=1,
            organisms_per_candidate=5,
            round1_screening_hits=1,
            round2_screening_hits=1,
            round1_broad_spectrum_hits=0,
            round2_broad_spectrum_hits=0,
            round1_potent_hits=1,
            round2_potent_hits=1,
        )
        self.assertEqual(summarize_rows(rows, expected=expected)["measurements"], 10)

    def test_requires_library_mapping(self):
        rows = candidate_rows("R1_01", "", "Round 1", "64") + candidate_rows("R2_01", "BNP_000002", "Round 2", "8")
        with self.assertRaisesRegex(MicAuditError, "missing candidate_library_id"):
            summarize_rows(
                rows,
                expected=ScreeningExpectations(1, 5, 1, 1, 0, 0, 1, 1),
            )

    def test_rejects_endpoint_flag_disagreement(self):
        rows = candidate_rows("R1_01", "BNP_000001", "Round 1", "64") + candidate_rows("R2_01", "BNP_000002", "Round 2", "8")
        rows[0]["active_at_MIC_le_256"] = "false"
        with self.assertRaisesRegex(MicAuditError, "disagrees"):
            summarize_rows(rows, expected=ScreeningExpectations(1, 5, 1, 1, 0, 0, 1, 1))

    def test_rejects_round_overlap(self):
        rows = candidate_rows("shared", "BNP_000001", "Round 1", "64") + candidate_rows("shared", "BNP_000001", "Round 2", "8")
        with self.assertRaisesRegex(MicAuditError, "share candidate IDs"):
            summarize_rows(rows, expected=ScreeningExpectations(1, 5, 1, 1, 0, 0, 1, 1))

    def test_broad_spectrum_requires_two_active_organisms(self):
        rows = candidate_rows("R1_01", "BNP_000001", "Round 1", "64")
        rows[1]["MIC_ug_mL"] = "128"  # second active organism
        rows[1]["active_at_MIC_le_256"] = "true"
        rows += candidate_rows("R2_01", "BNP_000002", "Round 2", ">256")
        expected = ScreeningExpectations(
            candidates_per_round=1,
            organisms_per_candidate=5,
            round1_screening_hits=1,
            round2_screening_hits=0,
            round1_broad_spectrum_hits=1,
            round2_broad_spectrum_hits=0,
            round1_potent_hits=1,
            round2_potent_hits=0,
        )
        summary = summarize_rows(rows, expected=expected)
        self.assertEqual(summary["round1_broad_spectrum_hits"], 1)
