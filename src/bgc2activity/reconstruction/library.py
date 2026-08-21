"""NRPS-informed synthetic candidate library construction and variant
generation (Supplementary Method 13).

From the 12,266 system-level parent candidates (confidence >= 0.8, valid
module count >= 5, synthesis-admissibility), a bounded set of variants is
generated under pre-frozen rules - C-terminal free acid/amidation,
D-configuration of Orn, and lipidated side chains - without Cartesian
combinatorial expansion, giving 23,540 systematic candidate records of which
29 are project-defined constrained variants derived from 19 parent BGCs. The
final library covers 12,266 BGCs, 3,713 core sequences and 5,303 displayed
sequences; 20,579 records contain nonstandard or modified monomers and 1,625
carry explicit lipidation.

Released composition (Table S1c / data/candidate_library/
constrained_variant_rules.tsv): 12,035 direct-concatenation records (OH) and
9,076 C-terminal amidation records (NH2) - terminal forms are pre-frozen per
parent rather than a full Cartesian product; 783 Orn-configuration records
(first Orn to D-Orn, OH); seven lipid rules with 231 parents each (Myr
reference OH, Lau OH, C13:0 OH, iso-C13:0 branched OH, MyrO unsaturated OH,
3R-OH-Myr beta-hydroxy OH, Myr-NH2 fatty-acylated plus amidation NH2); and 29
project-defined constrained variants.

Physicochemical annotation follows the manuscript: molecular weight is the sum
of monomer masses minus condensation water with terminal/lipidation
corrections; net charge at pH 7 uses Henderson-Hasselbalch with the
lipidated-Lys side chain neutralized; the isoelectric point is the pH at
which the net charge equals zero (missing when no crossing exists between
pH 0 and 14); GRAVY excludes fatty acyl groups. Masses are average isotopic
residue masses; ionization constants are standard aqueous approximations.
The generated entities are synthetic candidates, not natural products.
"""

from __future__ import annotations

from dataclasses import dataclass

WATER_MASS = 18.015
HYDROGEN_MASS = 1.008

VARIANT_CLASS_DIRECT = "direct_concatenation"
VARIANT_CLASS_C_TERMINAL_AMIDATION = "c_terminal_amidation"
VARIANT_CLASS_ORN_CONFIGURATION = "orn_configuration"
VARIANT_CLASS_CONSTRAINED = "constrained_variant"
VARIANT_CLASS_FATTY_ACYL_REFERENCE = "fatty_acyl_reference"
VARIANT_CLASS_FATTY_ACYL_CHAIN_LENGTH = "fatty_acyl_chain_length"
VARIANT_CLASS_BRANCHED_CHAIN_FATTY_ACYL = "branched_chain_fatty_acyl"
VARIANT_CLASS_UNSATURATED_FATTY_ACYL = "unsaturated_fatty_acyl"
VARIANT_CLASS_BETA_HYDROXY_FATTY_ACYL = "beta_hydroxy_fatty_acyl"
VARIANT_CLASS_FATTY_ACYLATED_PLUS_AMIDATION = "fatty_acylated_plus_c_terminal_amidation"

SYSTEMATIC_LIBRARY_INVARIANTS = {
    "parent_bgc_count": 12_266,
    "systematic_candidate_records": 23_540,
    "constrained_variant_records": 29,
    "constrained_variant_parents": 19,
    "unique_core_sequences": 3_713,
    "unique_displayed_sequences": 5_303,
    "nonstandard_or_modified_records": 20_579,
    "explicit_lipidation_records": 1_625,
    "direct_concatenation_records": 12_035,
    "c_terminal_amidation_records": 9_076,
    "orn_configuration_records": 783,
    "lipid_rules": 7,
    "parents_per_lipid_rule": 231,
}

# Frozen lipid-variant rule table mirroring Table S1c /
# constrained_variant_rules.tsv: lipid key -> (variant class, detail,
# C-terminal state, acyl token used for display).
LIPID_VARIANT_RULES = {
    "Myr": (VARIANT_CLASS_FATTY_ACYL_REFERENCE, "Myr", "OH", "Myr"),
    "Lau": (VARIANT_CLASS_FATTY_ACYL_CHAIN_LENGTH, "Lau", "OH", "Lau"),
    "C13:0": (VARIANT_CLASS_FATTY_ACYL_CHAIN_LENGTH, "C13:0", "OH", "C13:0"),
    "iso-C13:0": (VARIANT_CLASS_BRANCHED_CHAIN_FATTY_ACYL, "iso-C13:0", "OH", "iso-C13:0"),
    "MyrO": (VARIANT_CLASS_UNSATURATED_FATTY_ACYL, "MyrO", "OH", "MyrO"),
    "3R-OH-Myr": (VARIANT_CLASS_BETA_HYDROXY_FATTY_ACYL, "3R-OH-Myr", "OH", "3R-OH-Myr"),
    "Myr-NH2": (VARIANT_CLASS_FATTY_ACYLATED_PLUS_AMIDATION, "Myr-NH2", "NH2", "Myr"),
}

STANDARD_MONOMER_RESIDUE_MASSES = {
    "Gly": 57.05, "Ala": 71.08, "Ser": 87.08, "Pro": 97.12, "Val": 99.13,
    "Thr": 101.15, "Cys": 103.14, "Leu": 113.16, "Ile": 113.16, "Asn": 114.10,
    "Asp": 115.09, "Gln": 128.13, "Lys": 128.17, "Glu": 129.12, "Met": 131.19,
    "His": 137.14, "Phe": 147.18, "Arg": 156.19, "Tyr": 163.18, "Trp": 186.21,
}

NONSTANDARD_MONOMER_RESIDUE_MASSES = {
    "Orn": 114.15, "Dab": 100.12, "Dap": 86.09, "HxOrn": 130.13, "HOrn": 130.13,
    "Hpg": 149.15, "Dpg": 165.15, "Pip": 111.14, "Hse": 101.10, "Hty": 179.17,
    "Sal": 120.10, "DHBA": 136.10, "β-Ala": 71.08, "β-Asp": 115.09, "β-Lys": 128.17,
}

FATTY_ACYL_MASSES = {
    "Myr": 210.36, "Lau": 183.30, "C13:0": 197.33, "iso-C13:0": 197.33,
    "MyrO": 208.34, "3R-OH-Myr": 226.35,
}

AROMATIC_MONOMERS = frozenset({"Phe", "Tyr", "Trp", "His", "Hpg", "Dpg", "Hty", "Sal", "DHBA"})

KYTE_DOOLITTLE_HYDROPATHY = {
    "Ile": 4.5, "Val": 4.2, "Leu": 3.8, "Phe": 2.8, "Cys": 2.5, "Met": 1.9,
    "Ala": 1.8, "Gly": -0.4, "Thr": -0.7, "Ser": -0.8, "Trp": -0.9, "Tyr": -1.3,
    "Pro": -1.6, "His": -3.2, "Glu": -3.5, "Gln": -3.5, "Asp": -3.5, "Asn": -3.5,
    "Lys": -3.9, "Arg": -4.5,
}

_PKAS_ACIDIC = {"Asp": 3.9, "Glu": 4.3, "Cys": 8.3, "Tyr": 10.1, "β-Asp": 3.9, "DHBA": 7.0, "Sal": 6.5}
_PKAS_BASIC = {"His": 6.0, "Lys": 10.5, "Arg": 12.5, "Orn": 10.8, "Dab": 10.6, "Dap": 9.9, "β-Lys": 10.5}
N_TERMINAL_PKA = 9.0
C_TERMINAL_PKA = 2.0
LIPIDATED_LYSINE_TOKEN_PREFIX = "Lys(ε-"


@dataclass(frozen=True)
class ParentCandidate:
    """A system-level NRPS-inferred parent candidate passing all Method 8 gates.

    ``terminal_forms`` lists the C-terminal states admitted for this parent by
    the pre-frozen terminal-form rules (Table S1); the released library is not
    a Cartesian product of both forms for every parent (12,035 OH + 9,076 NH2
    base records over 12,266 parents). ``lipid_rule`` names one key of
    ``LIPID_VARIANT_RULES`` when the parent is lipidation-eligible.
    """

    bgc_id: str
    genome_accession: str
    core_monomers: tuple[str, ...]
    terminal_forms: tuple[str, ...] = ("OH",)
    has_ornithine: bool = False
    lipid_rule: str | None = None


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    parent_bgc_id: str
    genome_accession: str
    core_sequence: str
    displayed_sequence: str
    c_terminal: str  # "OH" or "NH2"
    variant_class: str
    variant_detail: str
    is_constrained_variant: bool
    is_nonstandard_or_modified: bool
    is_lipidated: bool


def display_sequence(core_monomers: tuple[str, ...], c_terminal: str, lipid_acyl: str | None = None) -> str:
    """Render the displayed sequence with H- N terminus and OH/NH2 C terminus.

    Lipidated Lys residues are displayed as ``Lys(ε-<acyl>)`` tokens.
    """
    tokens = []
    for monomer in core_monomers:
        if lipid_acyl is not None and monomer == "Lys":
            tokens.append(f"{LIPIDATED_LYSINE_TOKEN_PREFIX}{lipid_acyl})")
        else:
            tokens.append(monomer)
    return "H-" + "-".join(tokens) + ("-OH" if c_terminal == "OH" else "-NH2")


def enumerate_systematic_variants(parent: ParentCandidate) -> list[dict]:
    """Enumerate the bounded variant set for one parent under pre-frozen rules
    (no Cartesian combinatorial expansion, Supplementary Method 13).

    Each admitted terminal form yields one base record: OH records are
    direct-concatenation variants, NH2 records are C-terminal amidation
    variants. Ornithine parents additionally yield the Orn-configuration
    record (first Orn to D-Orn, OH terminal), and lipidation-eligible parents
    yield the single lipid record fixed by their ``lipid_rule`` (class,
    detail and terminal state follow the frozen rule table).
    """
    variants = []
    for terminal in parent.terminal_forms:
        if terminal == "OH":
            variants.append({"c_terminal": "OH", "variant_class": VARIANT_CLASS_DIRECT,
                             "variant_detail": "base_sequence", "lipid_acyl": None})
        elif terminal == "NH2":
            variants.append({"c_terminal": "NH2", "variant_class": VARIANT_CLASS_C_TERMINAL_AMIDATION,
                             "variant_detail": "base_sequence-NH2", "lipid_acyl": None})
        else:
            raise ValueError(f"unknown terminal form: {terminal}")
    if parent.has_ornithine:
        variants.append(
            {"c_terminal": "OH", "variant_class": VARIANT_CLASS_ORN_CONFIGURATION,
             "variant_detail": "first_Orn_to_D-Orn", "lipid_acyl": None}
        )
    if parent.lipid_rule is not None:
        if parent.lipid_rule not in LIPID_VARIANT_RULES:
            raise ValueError(f"unknown lipid rule: {parent.lipid_rule}")
        variant_class, detail, terminal, acyl = LIPID_VARIANT_RULES[parent.lipid_rule]
        variants.append(
            {"c_terminal": terminal, "variant_class": variant_class,
             "variant_detail": detail, "lipid_acyl": acyl}
        )
    return variants


def build_candidate_records(
    parents: list[ParentCandidate],
    constrained_variants: list[dict] | None = None,
) -> list[CandidateRecord]:
    """Assemble the full candidate library with sequential ``BNP_%06d`` IDs.

    Systematic variants are enumerated per parent in input order; constrained
    variants (29 project-defined records from 19 parent BGCs) are supplied
    verbatim because they are project-defined rather than rule-derived.
    """
    records: list[CandidateRecord] = []
    for parent in parents:
        for variant in enumerate_systematic_variants(parent):
            displayed = display_sequence(parent.core_monomers, variant["c_terminal"], variant["lipid_acyl"])
            records.append(
                CandidateRecord(
                    candidate_id="",
                    parent_bgc_id=parent.bgc_id,
                    genome_accession=parent.genome_accession,
                    core_sequence="-".join(parent.core_monomers),
                    displayed_sequence=displayed,
                    c_terminal=variant["c_terminal"],
                    variant_class=variant["variant_class"],
                    variant_detail=variant["variant_detail"],
                    is_constrained_variant=False,
                    is_nonstandard_or_modified=_carries_nonstandard(parent.core_monomers, variant["lipid_acyl"], variant["c_terminal"]),
                    is_lipidated=variant["lipid_acyl"] is not None,
                )
            )
    for constrained in constrained_variants or []:
        records.append(
            CandidateRecord(
                candidate_id="",
                parent_bgc_id=constrained["bgc_id"],
                genome_accession=constrained["genome_accession"],
                core_sequence=constrained["core_sequence"],
                displayed_sequence=constrained["displayed_sequence"],
                c_terminal=constrained.get("c_terminal", "OH"),
                variant_class=VARIANT_CLASS_CONSTRAINED,
                variant_detail="project_defined_sequence_variant",
                is_constrained_variant=True,
                is_nonstandard_or_modified=constrained.get("is_nonstandard_or_modified", True),
                is_lipidated=constrained.get("is_lipidated", False),
            )
        )
    numbered = []
    for index, record in enumerate(records, start=1):
        numbered.append(
            CandidateRecord(
                candidate_id=f"BNP_{index:06d}",
                parent_bgc_id=record.parent_bgc_id,
                genome_accession=record.genome_accession,
                core_sequence=record.core_sequence,
                displayed_sequence=record.displayed_sequence,
                c_terminal=record.c_terminal,
                variant_class=record.variant_class,
                variant_detail=record.variant_detail,
                is_constrained_variant=record.is_constrained_variant,
                is_nonstandard_or_modified=record.is_nonstandard_or_modified,
                is_lipidated=record.is_lipidated,
            )
        )
    return numbered


def _carries_nonstandard(core_monomers: tuple[str, ...], lipid_acyl: str | None, c_terminal: str) -> bool:
    if lipid_acyl is not None or c_terminal == "NH2":
        return True
    return any(monomer in NONSTANDARD_MONOMER_RESIDUE_MASSES for monomer in core_monomers)


def molecular_weight(
    core_monomers: tuple[str, ...],
    c_terminal: str = "OH",
    lipid_acyl: str | None = None,
    n_lipidated_lysines: int | None = None,
) -> float:
    """Molecular weight: sum of residue masses minus condensation water, with
    terminal and lipidation corrections (Supplementary Method 13).

    Peptide backbone mass = sum(residue masses) + water. A C-terminal amide
    replaces the terminal OH by NH2 (-0.988 Da net). Each fatty acyl group
    condenses with a Lys side chain, contributing acyl mass minus one hydrogen.
    """
    residue_mass = 0.0
    for monomer in core_monomers:
        if monomer in STANDARD_MONOMER_RESIDUE_MASSES:
            residue_mass += STANDARD_MONOMER_RESIDUE_MASSES[monomer]
        elif monomer in NONSTANDARD_MONOMER_RESIDUE_MASSES:
            residue_mass += NONSTANDARD_MONOMER_RESIDUE_MASSES[monomer]
        else:
            raise KeyError(f"monomer without a frozen residue mass: {monomer}")
    mass = residue_mass + WATER_MASS
    if c_terminal == "NH2":
        mass -= 17.007 - 16.019  # OH -> NH2 replacement
    if lipid_acyl is not None:
        acyl_mass = FATTY_ACYL_MASSES[lipid_acyl]
        n_sites = n_lipidated_lysines if n_lipidated_lysines is not None else sum(
            1 for monomer in core_monomers if monomer == "Lys"
        )
        mass += n_sites * (acyl_mass - HYDROGEN_MASS)
    return mass


def _acidic_charge(pka: float, ph: float) -> float:
    return -1.0 / (1.0 + 10.0 ** (pka - ph))


def _basic_charge(pka: float, ph: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def net_charge_ph7(
    core_monomers: tuple[str, ...],
    c_terminal: str = "OH",
    lipidated_lysines: frozenset[int] = frozenset(),
    n_terminus_free: bool = True,
    ph: float = 7.0,
) -> float:
    """Henderson-Hasselbalch net charge estimate (Supplementary Method 13).

    The side-chain charge of lipidated Lys residues is neutralized; an
    amidated C terminus carries no terminal carboxyl charge.
    """
    charge = 0.0
    if n_terminus_free:
        charge += _basic_charge(N_TERMINAL_PKA, ph)
    if c_terminal == "OH":
        charge += _acidic_charge(C_TERMINAL_PKA, ph)
    for position, monomer in enumerate(core_monomers):
        if monomer in _PKAS_ACIDIC:
            charge += _acidic_charge(_PKAS_ACIDIC[monomer], ph)
        if monomer in _PKAS_BASIC:
            if monomer == "Lys" and position in lipidated_lysines:
                continue
            charge += _basic_charge(_PKAS_BASIC[monomer], ph)
    return charge


def isoelectric_point(
    core_monomers: tuple[str, ...],
    c_terminal: str = "OH",
    lipidated_lysines: frozenset[int] = frozenset(),
    ph_range: tuple[float, float] = (0.0, 14.0),
    tolerance: float = 1e-3,
) -> float | None:
    """pH at which the net charge crosses zero (recorded as missing when no
    crossing exists within pH 0-14, Supplementary Method 13)."""
    low, high = ph_range
    charge_low = net_charge_ph7(core_monomers, c_terminal, lipidated_lysines, ph=low)
    charge_high = net_charge_ph7(core_monomers, c_terminal, lipidated_lysines, ph=high)
    if charge_low * charge_high > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2.0
        charge_mid = net_charge_ph7(core_monomers, c_terminal, lipidated_lysines, ph=mid)
        if abs(charge_mid) < tolerance:
            return mid
        if charge_low * charge_mid <= 0:
            high = mid
        else:
            low = mid
            charge_low = charge_mid
    return (low + high) / 2.0


def gravy(core_monomers: tuple[str, ...]) -> float | None:
    """GRAVY computed from Kyte-Doolittle hydropathy, excluding fatty acyl
    groups and any monomer without a tabulated value (Supplementary Method 13)."""
    values = [KYTE_DOOLITTLE_HYDROPATHY[monomer] for monomer in core_monomers if monomer in KYTE_DOOLITTLE_HYDROPATHY]
    if not values:
        return None
    return sum(values) / len(values)


def aromatic_fraction(core_monomers: tuple[str, ...]) -> float:
    if not core_monomers:
        return 0.0
    return sum(1 for monomer in core_monomers if monomer in AROMATIC_MONOMERS) / len(core_monomers)


def annotate_physicochemical(record: CandidateRecord, core_monomers: tuple[str, ...]) -> dict[str, float | None]:
    """Physicochemical annotation block reported for every library record."""
    lipid_acyl = None
    if record.is_lipidated and record.variant_detail in {rule[1] for rule in LIPID_VARIANT_RULES.values()}:
        lipid_acyl = LIPID_VARIANT_RULES[record.variant_detail][3]
    return {
        "length": len(core_monomers),
        "molecular_weight": molecular_weight(core_monomers, record.c_terminal, lipid_acyl),
        "net_charge_pH7": net_charge_ph7(core_monomers, record.c_terminal),
        "isoelectric_point": isoelectric_point(core_monomers, record.c_terminal),
        "gravy": gravy(core_monomers),
        "aromatic_fraction": aromatic_fraction(core_monomers),
    }
