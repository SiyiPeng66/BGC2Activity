"""Contract tests for public peptide corpus curation (Supplementary Methods 9
and 10.1)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgc2activity.representation.corpus import (
    CONFIGURATION_LEVELS,
    CORPUS_SPLIT_FRACTIONS,
    CORPUS_SPLIT_SEED,
    CURATION_INVARIANTS,
    MAX_CORE_MONOMERS,
    MIN_CORE_MONOMERS,
    NEIGHBOUR_EXCLUSION_SIMILARITY,
    SNAPSHOT_FREEZE_DATE,
    SOURCE_DATABASES,
    ChemicalEntity,
    PeptideRecord,
    build_vocabulary,
    exclude_mic_heldout_neighbours,
    lipidated_lys_token,
    merge_chemical_entities,
    modification_flags,
    normalize_monomer,
    passes_length_filter,
    split_corpus,
    vocabulary_status,
)


def make_record(sequence, c_terminal="OH", nonstandard=(), source="DBAASP"):
    return PeptideRecord(
        source_database=source,
        canonical_sequence=tuple(sequence),
        configurations=tuple("unknown" for _ in sequence),
        nonstandard_monomers=tuple(nonstandard),
        c_terminal=c_terminal,
    )


class TestCurationContract(unittest.TestCase):
    def test_sources_and_freeze_date(self):
        self.assertEqual(SOURCE_DATABASES, ("DBAASP", "DRAMP", "APD", "CAMPR4", "dbAMP", "SATPdb"))
        self.assertEqual(SNAPSHOT_FREEZE_DATE, "2026-07-21")

    def test_curation_counts(self):
        self.assertEqual(CURATION_INVARIANTS["source_records"], 118_897)
        self.assertEqual(CURATION_INVARIANTS["parsed_canonical_sequences"], 118_630)
        self.assertEqual(CURATION_INVARIANTS["unparseable_records"], 267)
        self.assertEqual(CURATION_INVARIANTS["records_with_nonstandard_or_modification_signal"], 21_814)
        self.assertEqual(CURATION_INVARIANTS["sequences_with_4_to_64_monomers"], 105_353)
        self.assertEqual(CURATION_INVARIANTS["deduplicated_entity_candidates"], 64_127)
        self.assertEqual(CURATION_INVARIANTS["entity_candidates_with_signal"], 17_618)


class TestStandardization(unittest.TestCase):
    def test_alias_normalization(self):
        self.assertEqual(normalize_monomer("24dab"), ("dab", "unknown"))
        self.assertEqual(normalize_monomer("asp-gamma"), ("beta-asp", "unknown"))

    def test_d_configuration_only_when_explicit(self):
        self.assertEqual(normalize_monomer("D-Orn"), ("orn", "D"))
        self.assertEqual(normalize_monomer("L-Lys"), ("lys", "L"))
        self.assertEqual(normalize_monomer("Orn"), ("orn", "unknown"))
        self.assertEqual(CONFIGURATION_LEVELS, ("L", "D", "unknown"))

    def test_lipidated_lys_tokens_per_acyl(self):
        self.assertEqual(lipidated_lys_token("Myr"), "lys(ε-Myr)")
        self.assertNotEqual(lipidated_lys_token("Myr"), lipidated_lys_token("Lau"))

    def test_modification_flags_non_exclusive(self):
        flags = modification_flags(c_terminal="NH2", cyclized=True, lipidated=True)
        self.assertEqual(flags, frozenset({"terminal_amidation", "cyclization", "lipid_conjugation"}))
        self.assertEqual(modification_flags(), frozenset())

    def test_length_filter_4_to_64(self):
        self.assertEqual((MIN_CORE_MONOMERS, MAX_CORE_MONOMERS), (4, 64))
        self.assertFalse(passes_length_filter(make_record(["ala"] * 3)))
        self.assertTrue(passes_length_filter(make_record(["ala"] * 4)))
        self.assertTrue(passes_length_filter(make_record(["ala"] * 64)))
        self.assertFalse(passes_length_filter(make_record(["ala"] * 65)))


class TestEntityMergingAndSplit(unittest.TestCase):
    def test_entity_merge_on_sequence_and_nonstandard_fields(self):
        records = [
            make_record(["ala", "lys", "glu", "ser"]),
            make_record(["ala", "lys", "glu", "ser"], source="DRAMP"),
            make_record(["ala", "lys", "glu", "ser"], nonstandard=("orn",)),
            make_record(["thr", "val", "gly", "asp"]),
        ]
        entities = merge_chemical_entities(records)
        self.assertEqual(len(entities), 3)
        counts = sorted(entity.source_count for entity in entities)
        self.assertEqual(counts, [1, 1, 2])

    def test_split_deterministic_80_10_10_seed_20260802(self):
        self.assertEqual(CORPUS_SPLIT_SEED, 20260802)
        self.assertEqual(CORPUS_SPLIT_FRACTIONS, (0.80, 0.10, 0.10))
        entities = [
            ChemicalEntity(tuple([f"m{index}"] * 4), (), 1) for index in range(100)
        ]
        train_a, val_a, test_a = split_corpus(entities)
        train_b, val_b, test_b = split_corpus(entities)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(test_a, test_b)
        self.assertEqual(len(train_a), 80)
        self.assertEqual(len(val_a), 10)
        self.assertEqual(len(test_a), 10)
        self.assertEqual(set(train_a) | set(val_a) | set(test_a), set(range(100)))
        self.assertFalse(set(train_a) & set(val_a))
        self.assertFalse(set(train_a) & set(test_a))

    def test_mic_heldout_exclusion_mask(self):
        self.assertEqual(NEIGHBOUR_EXCLUSION_SIMILARITY, 0.80)
        entities = [ChemicalEntity(tuple([f"m{index}"] * 4), (), 1) for index in range(4)]
        clusters = [0, 0, 1, 2]
        kept = exclude_mic_heldout_neighbours(entities, clusters, heldout_cluster_ids={0, 2})
        self.assertEqual(kept, [False, False, True, False])


class TestVocabulary(unittest.TestCase):
    def test_vocabulary_structure(self):
        vocabulary = build_vocabulary()
        self.assertEqual(len(vocabulary), len(set(vocabulary)))
        for token in ("<pad>", "<unk>", "<mask>"):
            self.assertIn(token, vocabulary)
        self.assertEqual(len(vocabulary), vocabulary_status()["current_size"])

    def test_manuscript_size_declared(self):
        status = vocabulary_status()
        self.assertEqual(status["manuscript_size"], 47)
        self.assertEqual(status["standard"], 20)
        self.assertEqual(status["special"], 3)
        self.assertEqual(status["nonstandard_confirmed"], 15)


if __name__ == "__main__":
    unittest.main()
