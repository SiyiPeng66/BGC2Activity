"""Audit the manuscript-defined local NRP45 derivative library."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_COLUMNS = frozenset(
    {
        "derivative_id",
        "parent_scaffold",
        "subtype",
        "candidate_structure",
        "enumeration_rule",
        "molecular_weight_Da",
        "feedback_updated_rank",
        "selected_for_synthesis",
    }
)


@dataclass(frozen=True)
class DerivativeLibraryExpectations:
    """Counts and mass range reported for Fig. 4 local optimisation."""

    total_derivatives: int = 384
    subtype_counts: Mapping[str, int] = field(
        default_factory=lambda: {"T1": 54, "T2": 126, "T3": 168, "T4": 36}
    )
    selected_for_synthesis: int = 20
    minimum_mass_da: float = 574.7
    maximum_mass_da: float = 928.3


class DerivativeLibraryError(ValueError):
    """Raised when a derivative-library release table violates the contract."""


def _as_bool(value: str, *, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise DerivativeLibraryError(f"row {row_number}: selected_for_synthesis must be boolean")


def _require(value: str, *, row_number: int, column: str) -> str:
    value = value.strip()
    if not value:
        raise DerivativeLibraryError(f"row {row_number}: missing {column}")
    return value


def _validated_rows(rows: Iterable[Mapping[str, str]], *, allowed_subtypes: set[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        subtype = _require(row["subtype"], row_number=row_number, column="subtype")
        if subtype not in allowed_subtypes:
            raise DerivativeLibraryError(f"row {row_number}: unsupported subtype {subtype!r}")
        parent = _require(row["parent_scaffold"], row_number=row_number, column="parent_scaffold")
        if parent != "NRP45":
            raise DerivativeLibraryError(f"row {row_number}: parent_scaffold must be NRP45")
        try:
            molecular_weight = float(_require(row["molecular_weight_Da"], row_number=row_number, column="molecular_weight_Da"))
            rank = int(_require(row["feedback_updated_rank"], row_number=row_number, column="feedback_updated_rank"))
        except ValueError as exc:
            raise DerivativeLibraryError(f"row {row_number}: invalid molecular weight or feedback-updated rank") from exc
        if molecular_weight <= 0 or rank < 1:
            raise DerivativeLibraryError(f"row {row_number}: molecular weight and rank must be positive")
        output.append(
            {
                "derivative_id": _require(row["derivative_id"], row_number=row_number, column="derivative_id"),
                "subtype": subtype,
                "candidate_structure": _require(row["candidate_structure"], row_number=row_number, column="candidate_structure"),
                "enumeration_rule": _require(row["enumeration_rule"], row_number=row_number, column="enumeration_rule"),
                "molecular_weight_Da": molecular_weight,
                "feedback_updated_rank": rank,
                "selected_for_synthesis": _as_bool(row["selected_for_synthesis"], row_number=row_number),
            }
        )
    return output


def summarize_rows(
    rows: Iterable[Mapping[str, str]], *, expected: DerivativeLibraryExpectations
) -> dict[str, object]:
    """Validate arbitrary derivative rows against a formal or test contract."""

    validated = _validated_rows(rows, allowed_subtypes=set(expected.subtype_counts))
    identifiers = [str(row["derivative_id"]) for row in validated]
    ranks = [int(row["feedback_updated_rank"]) for row in validated]
    if len(identifiers) != len(set(identifiers)):
        raise DerivativeLibraryError("derivative_id values must be unique")
    if len(ranks) != len(set(ranks)):
        raise DerivativeLibraryError("feedback_updated_rank values must be unique")
    if ranks and set(ranks) != set(range(1, len(ranks) + 1)):
        raise DerivativeLibraryError("feedback_updated_rank must cover the consecutive range 1..N")

    masses = [float(row["molecular_weight_Da"]) for row in validated]
    summary: dict[str, object] = {
        "total_derivatives": len(validated),
        "selected_for_synthesis": sum(bool(row["selected_for_synthesis"]) for row in validated),
        "minimum_mass_da": round(min(masses), 1) if masses else None,
        "maximum_mass_da": round(max(masses), 1) if masses else None,
        "subtype_counts": dict(Counter(str(row["subtype"]) for row in validated)),
    }
    mismatches: list[str] = []
    for name in ("total_derivatives", "selected_for_synthesis", "minimum_mass_da", "maximum_mass_da"):
        expected_value = getattr(expected, name)
        if summary[name] != expected_value:
            mismatches.append(f"{name}: observed {summary[name]}, expected {expected_value}")
    if summary["subtype_counts"] != dict(expected.subtype_counts):
        mismatches.append(f"subtype_counts: observed {summary['subtype_counts']}, expected {dict(expected.subtype_counts)}")
    if mismatches:
        raise DerivativeLibraryError("derivative-library mismatch; " + "; ".join(mismatches))
    return summary


def audit_derivative_library(
    path: str | Path, *, expected: DerivativeLibraryExpectations = DerivativeLibraryExpectations()
) -> dict[str, object]:
    """Read and audit the UTF-8 TSV of all local NRP45 derivatives."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise DerivativeLibraryError(f"missing required columns: {', '.join(sorted(missing))}")
        return summarize_rows(reader, expected=expected)
