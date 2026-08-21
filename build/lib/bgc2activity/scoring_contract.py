"""Input boundary for the manuscript-defined feedback score update.

The update may use the frozen Base Activity Model logit and exactly nine
pre-specified chemical features. BGC detection/reconstruction/novelty metadata
remain provenance fields and are rejected when requested as activity inputs.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


BASE_SCORE_COLUMN = "sbase"
UPDATE_FEATURE_COLUMNS = (
    "length",
    "net_charge_pH7",
    "GRAVY",
    "aromatic_fraction",
    "nonstandard_monomer_fraction",
    "is_lipidated",
    "is_c_terminal_amidated",
    "molecular_weight",
    "structural_missingness_indicator",
)
REQUIRED_COLUMNS = frozenset({"candidate_library_id", BASE_SCORE_COLUMN, *UPDATE_FEATURE_COLUMNS})
FORBIDDEN_ACTIVITY_INPUTS = frozenset(
    {
        "bgc_score",
        "BGC_score",
        "bgc_evidence",
        "product_confidence",
        "reconstruction_confidence",
        "novelty_score",
        "evolutionary_novelty",
        "parent_bgc_id",
    }
)


class ScoreInputError(ValueError):
    """Raised when an activity-scoring data selection crosses the method boundary."""


def validate_update_feature_selection(columns: Iterable[str]) -> tuple[str, ...]:
    """Require precisely the frozen score-plus-nine-feature update design."""

    selected = tuple(columns)
    forbidden = set(selected) & FORBIDDEN_ACTIVITY_INPUTS
    if forbidden:
        raise ScoreInputError(
            "BGC metadata cannot be an activity-model input: " + ", ".join(sorted(forbidden))
        )
    if len(selected) != len(set(selected)):
        raise ScoreInputError("activity input columns must not be duplicated")
    expected = (BASE_SCORE_COLUMN, *UPDATE_FEATURE_COLUMNS)
    if selected != expected:
        raise ScoreInputError(
            "activity update must use the frozen Sbase-plus-nine-feature order: " + ", ".join(expected)
        )
    return selected


def _optional_float(value: object, *, column: str) -> float:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "na", "nan", "null"}):
        return math.nan
    if isinstance(value, bool):
        value = float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ScoreInputError(f"{column} must be numeric or explicitly missing") from exc


def select_update_inputs(record: Mapping[str, object]) -> tuple[float, tuple[float, ...]]:
    """Extract only approved numerical inputs from a record.

    Extra provenance columns may exist in the record but are never returned.
    Missing values are represented as ``NaN`` and must be imputed only inside
    the training fold. This function never substitutes a global value.
    """

    missing = REQUIRED_COLUMNS - set(record)
    if missing:
        raise ScoreInputError("missing required score inputs: " + ", ".join(sorted(missing)))
    values: list[float] = []
    for column in (BASE_SCORE_COLUMN, *UPDATE_FEATURE_COLUMNS):
        values.append(_optional_float(record[column], column=column))
    return values[0], tuple(values[1:])


def require_finite_after_fold_preprocessing(values: Iterable[float]) -> tuple[float, ...]:
    """Assert that nested fold-specific preprocessing has resolved missingness."""

    output = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in output):
        raise ScoreInputError("all update inputs must be finite after fold-specific preprocessing")
    return output
