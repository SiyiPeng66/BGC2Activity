"""Endpoint formulas of the wet-lab assays (Supplementary Methods 20-23).

All formulas are implemented exactly as written in the supplementary methods:

* haemolysis rate (%) and MHC5 (Method 21.1);
* CCK-8 relative cell viability (%) (Method 21.2);
* biofilm formation inhibition (%) and mature-biofilm biomass reduction (%)
  (Methods 22.1-22.2);
* time-kill viable burden log10(CFU mL^-1 + 1) (Method 20.1);
* resistance change MICn/MIC0 by serial passage (Method 20.2);
* implant bacterial burden log10 CFU per implant with the 10 CFU detection
  limit, and semi-quantitative histopathology scores (Methods 23.4, 23.7).

Technical replicates are averaged within each experiment before entering
between-group statistics (Supplementary Method 25); that aggregation is
provided by ``analysis.statistics.average_technical_replicates``.
"""

from __future__ import annotations

HAEMOLYSIS_LIMIT_PERCENT = 5.0
IMPLANT_DETECTION_LIMIT_LOG10 = 1.0  # 100 uL undiluted -> 10 CFU per implant
OVERALL_HISTOPATHOLOGY_ITEMS = 4  # 0-12 total
IMPLANT_SITE_HISTOPATHOLOGY_ITEMS = 3  # 0-9 total
MAX_ITEM_SCORE = 3


def haemolysis_rate(abs_sample: float, abs_negative: float, abs_positive: float) -> float:
    """Haemolysis rate (%) = (A_sample - A_neg) / (A_pos - A_neg) * 100
    (Method 21.1; saline 0% and 0.1% Triton X-100 100% controls)."""
    denominator = abs_positive - abs_negative
    if denominator == 0:
        raise ValueError("positive and negative controls must differ")
    return (abs_sample - abs_negative) / denominator * 100.0


def mhc5(concentrations: list[float], haemolysis_rates: list[float],
         limit_percent: float = HAEMOLYSIS_LIMIT_PERCENT) -> float | None:
    """MHC5: the highest peptide concentration causing no more than 5%
    haemolysis (Method 21.1); None when even the lowest concentration exceeds
    the limit."""
    if len(concentrations) != len(haemolysis_rates):
        raise ValueError("concentrations and rates must have the same length")
    passing = [concentration for concentration, rate in zip(concentrations, haemolysis_rates) if rate <= limit_percent]
    return max(passing) if passing else None


def cell_viability(abs_sample: float, abs_blank: float, abs_control: float) -> float:
    """Relative cell viability (%) = (A_sample - A_blank) / (A_control -
    A_blank) * 100 (Method 21.2, CCK-8; untreated cells 100%)."""
    denominator = abs_control - abs_blank
    if denominator == 0:
        raise ValueError("control and blank must differ")
    return (abs_sample - abs_blank) / denominator * 100.0


def biofilm_inhibition_rate(od_treated: float, od_blank: float, od_growth_control: float) -> float:
    """Biofilm formation inhibition (%) = [1 - (OD_treated - OD_blank) /
    (OD_growth_control - OD_blank)] * 100 (Method 22.1)."""
    denominator = od_growth_control - od_blank
    if denominator == 0:
        raise ValueError("growth control and blank must differ")
    return (1.0 - (od_treated - od_blank) / denominator) * 100.0


def mature_biofilm_reduction_rate(od_treated: float, od_blank: float, od_untreated_mature: float) -> float:
    """Mature-biofilm biomass reduction (%) = [1 - (OD_treated - OD_blank) /
    (OD_untreated_mature - OD_blank)] * 100 (Method 22.2)."""
    denominator = od_untreated_mature - od_blank
    if denominator == 0:
        raise ValueError("untreated mature biofilm control and blank must differ")
    return (1.0 - (od_treated - od_blank) / denominator) * 100.0


def viable_burden_log10(cfu_per_ml: float) -> float:
    """Time-kill viable burden reported as log10(CFU mL^-1 + 1) (Method 20.1)."""
    import math

    if cfu_per_ml < 0:
        raise ValueError("CFU mL^-1 must be non-negative")
    return math.log10(cfu_per_ml + 1.0)


def resistance_change(mic_day_n: float, mic_baseline: float) -> float:
    """Resistance change expressed as MICn/MIC0 (Method 20.2)."""
    if mic_baseline <= 0:
        raise ValueError("baseline MIC must be positive")
    return mic_day_n / mic_baseline


def implant_burden_log10(colony_count: int, dilution_factor: float, plated_volume_ml: float = 0.1) -> float:
    """Implant bacterial burden in log10 CFU per implant (Method 23.4).

    The 1 mL sonication suspension is converted by dilution factor and plating
    volume; the theoretical detection limit is 10 CFU per implant
    (100 uL undiluted), below which the value is reported at the limit.
    """
    import math

    if colony_count < 0 or dilution_factor <= 0 or plated_volume_ml <= 0:
        raise ValueError("invalid plating parameters")
    cfu_per_implant = colony_count * dilution_factor / plated_volume_ml
    if cfu_per_implant < 10.0:
        return IMPLANT_DETECTION_LIMIT_LOG10
    return math.log10(cfu_per_implant)


def histopathology_score(item_scores: list[int], expected_items: int) -> int:
    """Semi-quantitative histopathology total (Method 23.7).

    Each item is scored 0-3 (0 none; 1 mild/focal <25%; 2 moderate/multifocal
    25-50%; 3 severe/diffuse >50%). ``expected_items`` is 4 for the overall
    score (0-12) and 3 for the implant-site score (0-9). Fields are averaged
    across sections/non-overlapping fields per animal before this call.
    """
    if len(item_scores) != expected_items:
        raise ValueError(f"expected {expected_items} scored items")
    total = 0
    for score in item_scores:
        if not 0 <= score <= MAX_ITEM_SCORE:
            raise ValueError("item scores must be in 0-3")
        total += score
    return total
