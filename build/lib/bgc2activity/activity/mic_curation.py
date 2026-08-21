"""Curation of public MIC data (Supplementary Method 11).

The final main text trains the Base MIC Regression Model on curated
quantitative public MIC measurements with log10(MIC) as the continuous
target and a scalar regression head. This module implements record
harmonisation, quantitative target derivation and entity-level aggregation.
The strict binary labelling scheme (MIC <= 8 positive, MIC > 64 negative) is
retained *only* as a descriptive endpoint summary: the main text reports the
prospective MIC distribution at thresholds <= 8 ... <= 256 and Supplementary
Table S2c reports the strict-label funnel; neither enters model training
("prespecified MIC thresholds were used only to define experimental
screening and potent-hit endpoints and were not used as binary labels for
training the Base Model or feedback updater").

Quantitative target rules (reconstruction decisions RD-7/RD-8):

* point estimate (= v)     -> log10(v), uncensored;
* upper bound (< v, <= v)  -> log10(v), left-censored (true value <= anchor);
* lower bound (> v, >= v)  -> log10(v), right-censored (true value >= anchor);
* interval [value, upper]  -> log10 of the geometric mean;
* records without a numeric anchor or a known unit bucket are excluded;
* micromolar and microgram-per-millilitre values remain separate unit
  buckets and are never converted (Method 11); when a canonical sequence
  carries records in both buckets the ug_mL bucket is preferred for the
  entity-level target (the mass bucket matches the project's broth
  microdilution assays);
* the entity-level target is the minimum log10(MIC) over targets and sources
  within the preferred bucket - the quantitative analogue of the "positive
  if any target is explicitly positive" merge rule, preserving evidence of
  activity against at least one tested target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

POSITIVE_THRESHOLD = 8.0
NEGATIVE_THRESHOLD = 64.0
UNIT_BUCKETS = ("uM", "ug_mL")
PREFERRED_UNIT_BUCKET = "ug_mL"


@dataclass(frozen=True)
class MicMeasurement:
    source_database: str
    source_record_id: str
    canonical_sequence: str
    target_organism: str
    unit_bucket: str
    relation: str  # one of =, <, >, <=, >=
    value: float
    upper_bound: float | None = None  # interval records: MIC in [value, upper_bound]


@dataclass(frozen=True)
class QuantitativeTarget:
    log10_mic: float
    censoring: str  # "none", "left" or "right"


def log10_mic_target(measurement: MicMeasurement) -> QuantitativeTarget | None:
    """Derive the continuous log10(MIC) training target of one record."""
    if measurement.unit_bucket not in UNIT_BUCKETS:
        return None
    if measurement.value <= 0:
        return None
    if measurement.upper_bound is not None:
        if measurement.upper_bound <= 0 or measurement.upper_bound < measurement.value:
            return None
        geometric_mean = math.sqrt(measurement.value * measurement.upper_bound)
        return QuantitativeTarget(math.log10(geometric_mean), "none")
    if measurement.relation == "=":
        return QuantitativeTarget(math.log10(measurement.value), "none")
    if measurement.relation in ("<", "<="):
        return QuantitativeTarget(math.log10(measurement.value), "left")
    if measurement.relation in (">", ">="):
        return QuantitativeTarget(math.log10(measurement.value), "right")
    return None


def aggregate_sequence_targets(measurements: list[MicMeasurement]) -> dict[str, float]:
    """Entity-level quantitative target per canonical sequence (RD-8).

    Prefers the ug_mL bucket and takes the minimum log10(MIC) within it
    (most potent evidence); falls back to the uM bucket only when a sequence
    has no usable ug_mL record.
    """
    best: dict[str, tuple[int, float]] = {}
    for measurement in measurements:
        target = log10_mic_target(measurement)
        if target is None:
            continue
        bucket_rank = 0 if measurement.unit_bucket == PREFERRED_UNIT_BUCKET else 1
        current = best.get(measurement.canonical_sequence)
        if current is None or (bucket_rank, target.log10_mic) < current:
            best[measurement.canonical_sequence] = (bucket_rank, target.log10_mic)
    return {sequence: value for sequence, (_rank, value) in best.items()}


def strict_label_for_measurement(measurement: MicMeasurement) -> int | None:
    """Descriptive strict label: 1 (positive), 0 (negative) or None (excluded).

    Used only for endpoint summaries and the Supplementary Table S2c funnel;
    never as a training label under the regression formulation.
    """
    if measurement.unit_bucket not in UNIT_BUCKETS:
        return None
    if measurement.upper_bound is not None:
        # interval record: labelled only when the whole interval falls on one
        # side of a threshold; intervals crossing or between are excluded
        if measurement.upper_bound <= POSITIVE_THRESHOLD:
            return 1
        if measurement.value > NEGATIVE_THRESHOLD:
            return 0
        return None
    if measurement.relation in ("<", "<=", "="):
        if measurement.value <= POSITIVE_THRESHOLD:
            return 1
        if measurement.value > NEGATIVE_THRESHOLD:
            return 0
        return None
    if measurement.relation in (">", ">="):
        if measurement.relation == ">" and measurement.value >= NEGATIVE_THRESHOLD:
            return 0
        if measurement.relation == ">=" and measurement.value > NEGATIVE_THRESHOLD:
            return 0
        return None
    return None


def aggregate_sequence_labels(measurements: list[MicMeasurement]) -> dict[str, int]:
    """Descriptive merge per canonical sequence: positive if any target is positive."""
    labels: dict[str, int] = {}
    for measurement in measurements:
        label = strict_label_for_measurement(measurement)
        if label is None:
            continue
        current = labels.get(measurement.canonical_sequence)
        if current is None:
            labels[measurement.canonical_sequence] = label
        elif label == 1:
            labels[measurement.canonical_sequence] = 1
    return labels


def replicate_consistency_groups(measurements: list[MicMeasurement]) -> dict[tuple[str, str, str], list[float]]:
    """Group point estimates for replicate-consistency assessment (Method 11).

    Consistency is assessed only between point estimates (relation '=') that
    share identical source identifiers, target organisms and units. Returns
    ``(source_record_id, target_organism, unit_bucket) -> values``; groups of
    size >= 2 are the comparable replicate sets.
    """
    groups: dict[tuple[str, str, str], list[float]] = {}
    for measurement in measurements:
        if measurement.relation != "=" or measurement.upper_bound is not None:
            continue
        key = (measurement.source_record_id, measurement.target_organism, measurement.unit_bucket)
        groups.setdefault(key, []).append(measurement.value)
    return groups
