"""Audit a normalized NRPS-informed synthetic candidate library.

This module encodes the final supplementary-method contract.  It accepts only a
release schema, so legacy workbooks with abbreviated identifiers cannot be
mistaken for final release data.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_COLUMNS = frozenset(
    {
        "candidate_id",
        "parent_bgc_id",
        "genome_accession",
        "contig_id",
        "region_start",
        "region_end",
        "a_domain_coordinates",
        "predicted_substrates",
        "module_count",
        "canonical_core_sequence",
        "displayed_sequence",
        "c_terminal",
        "variant_class",
        "variant_detail",
        "reconstruction_confidence",
        "is_nonstandard_or_modified",
        "is_lipidated",
    }
)


@dataclass(frozen=True)
class LibraryExpectations:
    """Counts fixed by Supplementary Method 13."""

    records: int = 23_540
    parent_bgcs: int = 12_266
    core_sequences: int = 3_713
    displayed_sequences: int = 5_303
    nonstandard_or_modified: int = 20_579
    lipidated: int = 1_625


class CandidateLibraryError(ValueError):
    """Raised when a candidate-library release file violates the contract."""


def _as_bool(value: str, *, row_number: int, column: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise CandidateLibraryError(
        f"row {row_number}: {column} must be one of 1/0, true/false or yes/no; got {value!r}"
    )


def _require(value: str, *, row_number: int, column: str) -> str:
    value = value.strip()
    if not value:
        raise CandidateLibraryError(f"row {row_number}: missing {column}")
    return value


def _validate_row(row: Mapping[str, str], *, row_number: int) -> dict[str, object]:
    candidate_id = _require(row["candidate_id"], row_number=row_number, column="candidate_id")
    parent_bgc_id = _require(row["parent_bgc_id"], row_number=row_number, column="parent_bgc_id")
    if "..." in parent_bgc_id:
        raise CandidateLibraryError(
            f"row {row_number}: parent_bgc_id is abbreviated; release data require a full identifier"
        )

    try:
        module_count = int(_require(row["module_count"], row_number=row_number, column="module_count"))
        region_start = int(_require(row["region_start"], row_number=row_number, column="region_start"))
        region_end = int(_require(row["region_end"], row_number=row_number, column="region_end"))
        reconstruction_confidence = float(
            _require(row["reconstruction_confidence"], row_number=row_number, column="reconstruction_confidence")
        )
    except ValueError as exc:
        raise CandidateLibraryError(f"row {row_number}: invalid numeric value") from exc

    if module_count < 5:
        raise CandidateLibraryError(f"row {row_number}: module_count must be at least 5")
    if region_start < 1 or region_end < region_start:
        raise CandidateLibraryError(f"row {row_number}: invalid BGC genomic coordinates")
    if not 0.0 <= reconstruction_confidence <= 1.0:
        raise CandidateLibraryError(f"row {row_number}: reconstruction_confidence must be in [0, 1]")

    required_text = (
        "genome_accession",
        "contig_id",
        "a_domain_coordinates",
        "predicted_substrates",
        "canonical_core_sequence",
        "displayed_sequence",
        "c_terminal",
        "variant_class",
        "variant_detail",
    )
    values: dict[str, object] = {
        "candidate_id": candidate_id,
        "parent_bgc_id": parent_bgc_id,
        "module_count": module_count,
        "region_start": region_start,
        "region_end": region_end,
        "reconstruction_confidence": reconstruction_confidence,
    }
    for column in required_text:
        values[column] = _require(row[column], row_number=row_number, column=column)
    values["is_nonstandard_or_modified"] = _as_bool(
        row["is_nonstandard_or_modified"], row_number=row_number, column="is_nonstandard_or_modified"
    )
    values["is_lipidated"] = _as_bool(row["is_lipidated"], row_number=row_number, column="is_lipidated")
    return values


def read_candidate_library(path: str | Path) -> list[dict[str, object]]:
    """Read and validate a UTF-8, tab-delimited normalized candidate library."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        headers = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise CandidateLibraryError(f"missing required columns: {', '.join(sorted(missing))}")
        records = [_validate_row(row, row_number=index) for index, row in enumerate(reader, start=2)]

    ids = [str(record["candidate_id"]) for record in records]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise CandidateLibraryError(f"duplicate candidate_id values: {', '.join(sorted(duplicate_ids)[:5])}")
    return records


def audit_candidate_library(
    path: str | Path,
    *,
    expected: LibraryExpectations = LibraryExpectations(),
) -> dict[str, int]:
    """Return summary counts or raise when the final-library contract is unmet."""

    records = read_candidate_library(path)
    summary = {
        "records": len(records),
        "parent_bgcs": len({record["parent_bgc_id"] for record in records}),
        "core_sequences": len({record["canonical_core_sequence"] for record in records}),
        "displayed_sequences": len({record["displayed_sequence"] for record in records}),
        "nonstandard_or_modified": sum(bool(record["is_nonstandard_or_modified"]) for record in records),
        "lipidated": sum(bool(record["is_lipidated"]) for record in records),
    }
    expected_counts = asdict(expected)
    mismatches = [
        f"{name}: observed {summary[name]}, expected {expected_count}"
        for name, expected_count in expected_counts.items()
        if summary[name] != expected_count
    ]
    if mismatches:
        raise CandidateLibraryError("candidate-library count mismatch; " + "; ".join(mismatches))
    return summary


def audit_rows(
    rows: Iterable[Mapping[str, str]], *, expected: LibraryExpectations
) -> dict[str, int]:
    """Test helper for validated in-memory rows with custom expected counts."""

    records = [_validate_row(row, row_number=index) for index, row in enumerate(rows, start=2)]
    summary = {
        "records": len(records),
        "parent_bgcs": len({record["parent_bgc_id"] for record in records}),
        "core_sequences": len({record["canonical_core_sequence"] for record in records}),
        "displayed_sequences": len({record["displayed_sequence"] for record in records}),
        "nonstandard_or_modified": sum(bool(record["is_nonstandard_or_modified"]) for record in records),
        "lipidated": sum(bool(record["is_lipidated"]) for record in records),
    }
    mismatches = [
        f"{name}: observed {summary[name]}, expected {expected_count}"
        for name, expected_count in asdict(expected).items()
        if summary[name] != expected_count
    ]
    if mismatches:
        raise CandidateLibraryError("candidate-library count mismatch; " + "; ".join(mismatches))
    return summary
