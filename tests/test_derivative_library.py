from __future__ import annotations

import unittest

from bgc2activity.optimization.derivatives import (
    DerivativeLibraryError,
    DerivativeLibraryExpectations,
    summarize_rows,
)


def row(identifier: str, subtype: str, rank: int, mass: float, selected: str):
    return {
        "derivative_id": identifier,
        "parent_scaffold": "NRP45",
        "subtype": subtype,
        "candidate_structure": f"H-{identifier}-OH",
        "enumeration_rule": "frozen_rule",
        "molecular_weight_Da": str(mass),
        "feedback_updated_rank": str(rank),
        "selected_for_synthesis": selected,
    }


class DerivativeLibraryTests(unittest.TestCase):
    def setUp(self):
        self.expected = DerivativeLibraryExpectations(
            total_derivatives=4,
            subtype_counts={"T1": 1, "T2": 1, "T3": 1, "T4": 1},
            selected_for_synthesis=2,
            minimum_mass_da=574.7,
            maximum_mass_da=928.3,
        )
        self.rows = [
            row("NRP45-A01", "T1", 1, 574.7, "true"),
            row("NRP45-A02", "T2", 2, 650.0, "true"),
            row("NRP45-A03", "T3", 3, 700.0, "false"),
            row("NRP45-A04", "T4", 4, 928.3, "false"),
        ]

    def test_accepts_a_complete_custom_contract(self):
        self.assertEqual(summarize_rows(self.rows, expected=self.expected)["total_derivatives"], 4)

    def test_rejects_non_nrp45_parent(self):
        self.rows[0]["parent_scaffold"] = "other"
        with self.assertRaisesRegex(DerivativeLibraryError, "must be NRP45"):
            summarize_rows(self.rows, expected=self.expected)

    def test_rejects_a_duplicate_rank(self):
        self.rows[3]["feedback_updated_rank"] = "3"
        with self.assertRaisesRegex(DerivativeLibraryError, "rank values must be unique"):
            summarize_rows(self.rows, expected=self.expected)

    def test_rejects_wrong_subtype_composition(self):
        self.rows[3]["subtype"] = "T3"
        with self.assertRaisesRegex(DerivativeLibraryError, "subtype_counts"):
            summarize_rows(self.rows, expected=self.expected)
