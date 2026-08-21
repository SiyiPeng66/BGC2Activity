"""Public peptide corpus curation and pretraining data partitioning
(Supplementary Methods 9 and 10.1).

Peptide representation-learning data were obtained from DBAASP, DRAMP, APD,
CAMPR4, dbAMP and SATPdb (frozen snapshot 2026-07-21). Of 118,897 source
records, 118,630 were parsed into canonical sequences (267 excluded as
unparseable) and 21,814 contained at least one nonstandard monomer or
explicit modification signal.

Sequence standardization comprises lexical parsing, alias normalization (for
example 24dab -> dab, asp-gamma -> beta-asp; D-configurations are expanded
only when explicitly specified by the source) and chemical-identity review.
Lipidated Lys residues are encoded as distinct tokens according to the
specific acyl group. C-terminal free acid/amidation is a candidate-level
field and is not counted as a monomer; terminal amidation, cyclization and
lipid conjugation are counted in a non-mutually-exclusive manner. Labels with
unknown structure or configuration retain their original label and review
status.

For pretraining, sequences with 4-64 core monomers are merged into chemical
entities on (canonical sequence, nonstandard-monomer fields), giving 64,127
entity candidates (17,618 with signals). The formal corpus excludes MIC
validation clusters, MIC test clusters and their structural neighbours
(similarity >= 0.80); MIC training clusters may remain but their labels and
activity fields never enter pretraining. The corpus is split 80/10/10 with
seed 20260802.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SOURCE_DATABASES = ("DBAASP", "DRAMP", "APD", "CAMPR4", "dbAMP", "SATPdb")
SNAPSHOT_FREEZE_DATE = "2026-07-21"
CURATION_INVARIANTS = {
    "source_records": 118_897,
    "parsed_canonical_sequences": 118_630,
    "unparseable_records": 267,
    "records_with_nonstandard_or_modification_signal": 21_814,
    "sequences_with_4_to_64_monomers": 105_353,
    "deduplicated_entity_candidates": 64_127,
    "entity_candidates_with_signal": 17_618,
}
MIN_CORE_MONOMERS = 4
MAX_CORE_MONOMERS = 64
NEIGHBOUR_EXCLUSION_SIMILARITY = 0.80
CORPUS_SPLIT_SEED = 20260802
CORPUS_SPLIT_FRACTIONS = (0.80, 0.10, 0.10)

CONFIGURATION_LEVELS = ("L", "D", "unknown")

# Frozen alias table (Supplementary Method 9; extended by Table S1 review).
MONOMER_ALIAS_TABLE = {
    "24dab": "dab",
    "asp-gamma": "beta-asp",
    "2,3-dab": "dap",
}

STANDARD_AMINO_ACIDS = (
    "ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his", "ile",
    "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val",
)

# Nonstandard monomers confirmed in the released candidate library
# (Table S4); the manuscript vocabulary counts 24 nonstandard tokens, the
# remainder being observed only in the public corpus and fixed by
# Supplementary Table S1 (see docs/MANUSCRIPT_RELEASE_CONTRACT.md).
LIBRARY_NONSTANDARD_MONOMERS = (
    "orn", "dab", "dap", "hxorn", "horn", "hpg", "dpg", "pip", "hse", "hty",
    "sal", "dhba", "beta-ala", "beta-asp", "beta-lys",
)
CORPUS_ONLY_NONSTANDARD_MONOMERS: tuple[str, ...] = ()

SPECIAL_TOKENS = ("<pad>", "<unk>", "<mask>")
MANUSCRIPT_VOCABULARY_SIZE = 47

LIPIDATED_LYSINE_TEMPLATE = "lys(ε-{acyl})"


def lipidated_lys_token(acyl_group: str) -> str:
    """Lipidated Lys residues are distinct tokens per acyl group (Method 9)."""
    return LIPIDATED_LYSINE_TEMPLATE.format(acyl=acyl_group)


def normalize_monomer(raw_label: str) -> tuple[str, str]:
    """Normalize one monomer label to (identity, configuration).

    Alias normalization follows the frozen table; a D-configuration is
    expanded only when explicitly specified by the source; an explicit L-
    prefix is recorded as L, and anything unspecified stays ``unknown``.
    """
    label = raw_label.strip().lower()
    configuration = "unknown"
    if label.startswith("d-"):
        configuration = "D"
        label = label[2:]
    elif label.startswith("l-"):
        configuration = "L"
        label = label[2:]
    identity = MONOMER_ALIAS_TABLE.get(label, label)
    return identity, configuration


def modification_flags(
    c_terminal: str = "OH", cyclized: bool = False, lipidated: bool = False,
) -> frozenset[str]:
    """Non-mutually-exclusive modification accounting (Method 9)."""
    flags = set()
    if c_terminal == "NH2":
        flags.add("terminal_amidation")
    if cyclized:
        flags.add("cyclization")
    if lipidated:
        flags.add("lipid_conjugation")
    return frozenset(flags)


@dataclass(frozen=True)
class PeptideRecord:
    source_database: str
    canonical_sequence: tuple[str, ...]
    configurations: tuple[str, ...]
    nonstandard_monomers: tuple[str, ...] = ()
    c_terminal: str = "OH"
    cyclized: bool = False
    lipidated: bool = False
    original_label: str | None = None  # retained verbatim when structure unknown
    review_status: str = "reviewed"


@dataclass(frozen=True)
class ChemicalEntity:
    canonical_sequence: tuple[str, ...]
    nonstandard_monomers: tuple[str, ...]
    source_count: int
    c_terminal: str = "OH"


def passes_length_filter(record: PeptideRecord) -> bool:
    """Pretraining corpus membership: 4-64 core monomers (Method 10.1)."""
    return MIN_CORE_MONOMERS <= len(record.canonical_sequence) <= MAX_CORE_MONOMERS


def merge_chemical_entities(records: list[PeptideRecord]) -> list[ChemicalEntity]:
    """Merge records sharing canonical sequence and nonstandard-monomer fields
    into chemical entities (Method 10.1)."""
    merged: dict[tuple, dict[str, object]] = {}
    for record in records:
        if not passes_length_filter(record):
            continue
        key = (record.canonical_sequence, record.nonstandard_monomers)
        entry = merged.get(key)
        if entry is None:
            merged[key] = {
                "source_count": 1,
                "c_terminal": record.c_terminal,
            }
        else:
            entry["source_count"] += 1
    entities = []
    for (sequence, nonstandard), entry in merged.items():
        entities.append(
            ChemicalEntity(
                canonical_sequence=sequence,
                nonstandard_monomers=nonstandard,
                source_count=int(entry["source_count"]),
                c_terminal=str(entry["c_terminal"]),
            )
        )
    return entities


def assign_corpus_clusters(
    entities: list[ChemicalEntity],
    fingerprints: list[frozenset[int]] | None = None,
) -> list[int]:
    """Structural clustering used for corpus exclusion (Methods 10.1/12 rule):
    identical sequences, terminal variants sharing a core, and pairs with
    global similarity >= 0.80 and length ratio >= 0.80 (or Morgan Tanimoto
    >= 0.60 when fingerprints are supplied) form one cluster."""
    from bgc2activity.activity.clustering import assign_structural_clusters

    sequences = ["-".join(entity.canonical_sequence) for entity in entities]
    return assign_structural_clusters(sequences, fingerprints=fingerprints)


def exclude_mic_heldout_neighbours(
    entities: list[ChemicalEntity],
    cluster_labels: list[int],
    heldout_cluster_ids: set[int],
) -> list[bool]:
    """Mask of corpus members retained after excluding MIC validation/test
    clusters and their structural neighbours (similarity >= 0.80; Method 10.1).

    ``heldout_cluster_ids`` must already include the neighbour clusters of the
    MIC validation and test clusters; MIC training clusters are not excluded,
    and their labels/activity fields never enter pretraining.
    """
    if len(entities) != len(cluster_labels):
        raise ValueError("entities and cluster_labels must have the same length")
    return [label not in heldout_cluster_ids for label in cluster_labels]


def split_corpus(
    entities: list[ChemicalEntity],
    seed: int = CORPUS_SPLIT_SEED,
    fractions: tuple[float, float, float] = CORPUS_SPLIT_FRACTIONS,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic 80/10/10 corpus split (seed 20260802, Method 10.1)."""
    import random

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    order = list(range(len(entities)))
    rng = random.Random(seed)
    rng.shuffle(order)
    n_train = int(len(order) * fractions[0])
    n_val = int(len(order) * fractions[1])
    train = sorted(order[:n_train])
    validation = sorted(order[n_train:n_train + n_val])
    test = sorted(order[n_train + n_val:])
    return train, validation, test


def build_vocabulary() -> dict[str, int]:
    """Monomer vocabulary: 20 standard amino acids, nonstandard monomers and
    special markers (Method 10.1).

    The manuscript fixes the vocabulary at 47 tokens (20 standard + 24
    nonstandard + special markers). Tokens confirmed in the released library
    are listed here; the remaining corpus-only nonstandard monomers fixed by
    Supplementary Table S1 are appended to CORPUS_ONLY_NONSTANDARD_MONOMERS
    when that table is extracted (declared gap, release contract).
    """
    tokens = list(STANDARD_AMINO_ACIDS)
    tokens.extend(LIBRARY_NONSTANDARD_MONOMERS)
    tokens.extend(CORPUS_ONLY_NONSTANDARD_MONOMERS)
    tokens.extend(SPECIAL_TOKENS)
    if len(set(tokens)) != len(tokens):
        raise ValueError("vocabulary tokens must be unique")
    return {token: index for index, token in enumerate(tokens)}


def vocabulary_status() -> dict[str, int]:
    """Reported vocabulary census against the manuscript-fixed size."""
    vocabulary = build_vocabulary()
    return {
        "current_size": len(vocabulary),
        "manuscript_size": MANUSCRIPT_VOCABULARY_SIZE,
        "standard": len(STANDARD_AMINO_ACIDS),
        "nonstandard_confirmed": len(LIBRARY_NONSTANDARD_MONOMERS) + len(CORPUS_ONLY_NONSTANDARD_MONOMERS),
        "special": len(SPECIAL_TOKENS),
    }
