"""Audit complete prospective MIC data for the two fixed-budget cohorts.

Hit criteria follow Supplementary Method 18: MIC <= 256 ug mL^-1 against any
test organism defines a screening hit; reaching this threshold against >= 2
organisms defines a broad-spectrum hit; MIC <= 64 ug mL^-1 against any
organism defines a potent hit; the minimum MIC is the smallest value across
the five test organisms, and candidates above 256 ug mL^-1 against all five
organisms are recorded as > 256 without substitution by a continuous value.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_COLUMNS = frozenset(
    {
        "candidate_id",
        "candidate_library_id",
        "screening_round",
        "sequence",
        "test_organism",
        "MIC_ug_mL",
        "active_at_MIC_le_256",
    }
)


@dataclass(frozen=True)
class ScreeningExpectations:
    """Primary two-round screening endpoints reported in the final manuscript."""

    candidates_per_round: int = 40
    organisms_per_candidate: int = 5
    round1_screening_hits: int = 8
    round2_screening_hits: int = 18
    round1_broad_spectrum_hits: int = 0
    round2_broad_spectrum_hits: int = 5
    round1_potent_hits: int = 3
    round2_potent_hits: int = 12


class MicAuditError(ValueError):
    """Raised when a prospective MIC table violates its study contract."""


def _as_bool(value: str, *, row_number: int, column: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise MicAuditError(f"row {row_number}: {column} must be boolean; got {value!r}")


def parse_mic_ug_ml(value: str, *, row_number: int) -> float:
    """Parse a numeric MIC or the final screening assay's right-censored >256."""

    normalized = value.strip().replace(" ", "")
    if normalized == ">256":
        return math.inf
    try:
        mic = float(normalized)
    except ValueError as exc:
        raise MicAuditError(f"row {row_number}: invalid MIC_ug_mL {value!r}") from exc
    if not 0 < mic <= 256:
        raise MicAuditError(f"row {row_number}: finite MIC_ug_mL must be in (0, 256]")
    return mic


def _require(value: str, *, row_number: int, column: str) -> str:
    output = value.strip()
    if not output:
        raise MicAuditError(f"row {row_number}: missing {column}")
    return output


def _validated_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
    validated: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        screening_round = _require(row["screening_round"], row_number=row_number, column="screening_round")
        if screening_round not in {"Round 1", "Round 2"}:
            raise MicAuditError(f"row {row_number}: screening_round must be Round 1 or Round 2")
        mic = parse_mic_ug_ml(row["MIC_ug_mL"], row_number=row_number)
        active = _as_bool(row["active_at_MIC_le_256"], row_number=row_number, column="active_at_MIC_le_256")
        if active != math.isfinite(mic):
            raise MicAuditError(
                f"row {row_number}: active_at_MIC_le_256 disagrees with MIC_ug_mL {row['MIC_ug_mL']!r}"
            )
        validated.append(
            {
                "candidate_id": _require(row["candidate_id"], row_number=row_number, column="candidate_id"),
                "candidate_library_id": _require(
                    row["candidate_library_id"], row_number=row_number, column="candidate_library_id"
                ),
                "screening_round": screening_round,
                "sequence": _require(row["sequence"], row_number=row_number, column="sequence"),
                "test_organism": _require(row["test_organism"], row_number=row_number, column="test_organism"),
                "mic": mic,
            }
        )
    return validated


def summarize_rows(
    rows: Iterable[Mapping[str, str]], *, expected: ScreeningExpectations
) -> dict[str, int]:
    """Validate rows and return endpoint counts for custom or formal expectations."""

    validated = _validated_rows(rows)
    by_candidate: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    library_links: dict[str, str] = {}
    for row in validated:
        candidate_id = str(row["candidate_id"])
        linked_library_id = str(row["candidate_library_id"])
        previous = library_links.setdefault(candidate_id, linked_library_id)
        if previous != linked_library_id:
            raise MicAuditError(f"candidate {candidate_id}: multiple candidate_library_id values")
        by_candidate[(str(row["screening_round"]), candidate_id)].append(row)

    candidates_by_round: dict[str, set[str]] = {"Round 1": set(), "Round 2": set()}
    screening_hits = {"Round 1": 0, "Round 2": 0}
    broad_spectrum_hits = {"Round 1": 0, "Round 2": 0}
    potent_hits = {"Round 1": 0, "Round 2": 0}
    for (screening_round, candidate_id), measurements in by_candidate.items():
        candidates_by_round[screening_round].add(candidate_id)
        organisms = [str(record["test_organism"]) for record in measurements]
        if len(measurements) != expected.organisms_per_candidate or len(set(organisms)) != expected.organisms_per_candidate:
            raise MicAuditError(
                f"{screening_round} candidate {candidate_id}: expected {expected.organisms_per_candidate} unique organisms"
            )
        best_mic = min(float(record["mic"]) for record in measurements)
        active_organisms = sum(1 for record in measurements if float(record["mic"]) <= 256)
        screening_hits[screening_round] += int(best_mic <= 256)
        broad_spectrum_hits[screening_round] += int(active_organisms >= 2)
        potent_hits[screening_round] += int(best_mic <= 64)

    overlap = candidates_by_round["Round 1"] & candidates_by_round["Round 2"]
    if overlap:
        raise MicAuditError("Round 1 and Round 2 share candidate IDs: " + ", ".join(sorted(overlap)[:5]))
    summary = {
        "round1_candidates": len(candidates_by_round["Round 1"]),
        "round2_candidates": len(candidates_by_round["Round 2"]),
        "round1_screening_hits": screening_hits["Round 1"],
        "round2_screening_hits": screening_hits["Round 2"],
        "round1_broad_spectrum_hits": broad_spectrum_hits["Round 1"],
        "round2_broad_spectrum_hits": broad_spectrum_hits["Round 2"],
        "round1_potent_hits": potent_hits["Round 1"],
        "round2_potent_hits": potent_hits["Round 2"],
        "measurements": len(validated),
    }
    expected_values = {
        "round1_candidates": expected.candidates_per_round,
        "round2_candidates": expected.candidates_per_round,
        "round1_screening_hits": expected.round1_screening_hits,
        "round2_screening_hits": expected.round2_screening_hits,
        "round1_broad_spectrum_hits": expected.round1_broad_spectrum_hits,
        "round2_broad_spectrum_hits": expected.round2_broad_spectrum_hits,
        "round1_potent_hits": expected.round1_potent_hits,
        "round2_potent_hits": expected.round2_potent_hits,
    }
    mismatches = [
        f"{field}: observed {summary[field]}, expected {expected_value}"
        for field, expected_value in expected_values.items()
        if summary[field] != expected_value
    ]
    if mismatches:
        raise MicAuditError("prospective-screening mismatch; " + "; ".join(mismatches))
    return summary


def audit_prospective_mic(
    path: str | Path, *, expected: ScreeningExpectations = ScreeningExpectations()
) -> dict[str, int]:
    """Read a normalized TSV and audit its two-round MIC endpoints."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise MicAuditError(f"missing required columns: {', '.join(sorted(missing))}")
        return summarize_rows(reader, expected=expected)
