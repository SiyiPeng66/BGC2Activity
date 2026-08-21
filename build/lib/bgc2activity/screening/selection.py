"""Pre-specified, activity-independent candidate selection (Methods 14, 17).

The Round-1 cohort is established before any within-project activity data are
available: candidates are ranked in ascending order of the frozen Base MIC
Regression Model prediction (lower predicted log10(MIC) = higher priority),
the top 100 are taken, and only the prespecified activity-independent
constraints reduce the list to the final 40:

1. chemical-entity deduplication;
2. core-sequence / structural-cluster diversity caps;
3. per-BGC caps;
4. synthesis-feasibility checks.

The same constraint stack is applied to the Round-2 selection from candidates
that did not enter Round 1, using the frozen Calibrated Model scores
(calibrated predicted log10(MIC)). Any departure from these rules must be
reported explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

RANK_LONGLIST_SIZE = 100
FINAL_COHORT_SIZE = 40


@dataclass(frozen=True)
class SelectionConstraints:
    max_per_chemical_entity: int = 1
    max_per_core_sequence_cluster: int = 2
    max_per_bgc: int = 1
    synthesis_feasibility_column: str = "synthesis_feasible"


@dataclass
class SelectionRecord:
    selected: list[str] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)


def select_cohort(
    ranked_candidates: pd.DataFrame,
    constraints: SelectionConstraints,
    cohort_size: int = FINAL_COHORT_SIZE,
    longlist_size: int = RANK_LONGLIST_SIZE,
    excluded_candidate_ids: frozenset[str] = frozenset(),
    score_lower_is_better: bool = True,
) -> SelectionRecord:
    """Apply the frozen selection stack to a score-ranked candidate table.

    Required columns: candidate_id, score (predicted log10(MIC); lower is
    higher priority when ``score_lower_is_better``), chemical_entity_id,
    core_sequence_cluster_id, parent_bgc_id, synthesis_feasibility_column.
    """
    required = {"candidate_id", "score", "chemical_entity_id", "core_sequence_cluster_id",
                "parent_bgc_id", constraints.synthesis_feasibility_column}
    missing = required - set(ranked_candidates.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    pool = ranked_candidates[~ranked_candidates["candidate_id"].isin(excluded_candidate_ids)]
    pool = pool.sort_values("score", ascending=score_lower_is_better, kind="mergesort").head(longlist_size)

    record = SelectionRecord()
    entity_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    bgc_counts: dict[str, int] = {}

    for _index, row in pool.iterrows():
        if len(record.selected) >= cohort_size:
            break
        candidate_id = str(row["candidate_id"])
        if not bool(row[constraints.synthesis_feasibility_column]):
            record.rejected.setdefault("synthesis_infeasible", []).append(candidate_id)
            continue
        entity = str(row["chemical_entity_id"])
        if entity_counts.get(entity, 0) >= constraints.max_per_chemical_entity:
            record.rejected.setdefault("chemical_entity_cap", []).append(candidate_id)
            continue
        cluster = str(row["core_sequence_cluster_id"])
        if cluster_counts.get(cluster, 0) >= constraints.max_per_core_sequence_cluster:
            record.rejected.setdefault("core_sequence_cluster_cap", []).append(candidate_id)
            continue
        bgc = str(row["parent_bgc_id"])
        if bgc_counts.get(bgc, 0) >= constraints.max_per_bgc:
            record.rejected.setdefault("per_bgc_cap", []).append(candidate_id)
            continue
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        bgc_counts[bgc] = bgc_counts.get(bgc, 0) + 1
        record.selected.append(candidate_id)

    return record
