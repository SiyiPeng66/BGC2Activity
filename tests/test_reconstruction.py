"""Contract tests for candidate reconstruction and library construction
(Supplementary Methods 8 and 13)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgc2activity.reconstruction.screening import (
    ADOMAIN_CONFIDENCE_THRESHOLD,
    MINIMUM_CORE_MONOMERS,
    ADomainPrediction,
    accept_adomain_prediction,
    is_chemically_resolvable,
    is_mibig_high_similarity,
    reconstruct_core_sequence,
)
from bgc2activity.reconstruction.library import (
    LIPID_VARIANT_RULES,
    SYSTEMATIC_LIBRARY_INVARIANTS,
    VARIANT_CLASS_BETA_HYDROXY_FATTY_ACYL,
    VARIANT_CLASS_FATTY_ACYLATED_PLUS_AMIDATION,
    VARIANT_CLASS_FATTY_ACYL_CHAIN_LENGTH,
    VARIANT_CLASS_FATTY_ACYL_REFERENCE,
    ParentCandidate,
    annotate_physicochemical,
    aromatic_fraction,
    build_candidate_records,
    display_sequence,
    enumerate_systematic_variants,
    gravy,
    isoelectric_point,
    molecular_weight,
    net_charge_ph7,
)

SUBSTRATE_MAPPING = {"thr": "Thr", "lys": "Lys", "orn": "Orn", "glu": "Glu", "hxorn": "HxOrn"}


class TestMibigFiltering(unittest.TestCase):
    def test_similarity_is_strictly_greater_than_80_percent(self):
        # exactly 80% identity -> retained; 80.1% -> removed
        self.assertFalse(is_mibig_high_similarity(80, 100, 100, 100))
        self.assertTrue(is_mibig_high_similarity(81, 100, 100, 100))

    def test_coverage_required_on_both_sequences(self):
        self.assertTrue(is_mibig_high_similarity(90, 100, 125, 125))  # 80% both
        self.assertFalse(is_mibig_high_similarity(90, 100, 126, 125))  # query coverage < 80%
        self.assertFalse(is_mibig_high_similarity(90, 100, 125, 126))  # reference coverage < 80%

    def test_empty_alignment_never_removed(self):
        self.assertFalse(is_mibig_high_similarity(0, 0, 100, 100))


class TestADomainAndResolvability(unittest.TestCase):
    def test_confidence_threshold_is_0_8(self):
        self.assertEqual(ADOMAIN_CONFIDENCE_THRESHOLD, 0.8)
        self.assertTrue(accept_adomain_prediction(0.8))
        self.assertFalse(accept_adomain_prediction(0.79))

    def test_core_sequence_ordering_and_unresolved(self):
        predictions = [
            ADomainPrediction(0, 1, 0, "lys", 0.95),
            ADomainPrediction(0, 0, 0, "thr", 0.99),
            ADomainPrediction(0, 0, 1, "GLU", 0.90),
            ADomainPrediction(0, 2, 0, "orn", 0.50),  # below threshold
            ADomainPrediction(0, 3, 0, "unknown-substrate", 0.99),  # unmappable
        ]
        monomers, unresolved = reconstruct_core_sequence(predictions, SUBSTRATE_MAPPING)
        self.assertEqual(monomers, ["Thr", "Glu", "Lys", "?", "?"])
        self.assertEqual(unresolved, [3, 4])

    def test_chemical_resolvability_gate(self):
        self.assertEqual(MINIMUM_CORE_MONOMERS, 5)
        self.assertTrue(is_chemically_resolvable(["Thr"] * 5, []))
        self.assertFalse(is_chemically_resolvable(["Thr"] * 4, []))
        self.assertFalse(is_chemically_resolvable(["Thr"] * 5, [2]))

    def test_display_sequence_terminal_and_lipid_rendering(self):
        self.assertEqual(display_sequence(("Thr", "Lys"), "OH"), "H-Thr-Lys-OH")
        self.assertEqual(display_sequence(("Thr", "Lys"), "NH2"), "H-Thr-Lys-NH2")
        self.assertEqual(
            display_sequence(("Thr", "Lys"), "OH", "Myr"), "H-Thr-Lys(ε-Myr)-OH",
        )


class TestVariantEnumeration(unittest.TestCase):
    def test_terminal_forms_follow_per_parent_admission(self):
        # released library: 12,035 OH + 9,076 NH2 base records over 12,266
        # parents, so terminal forms are pre-frozen per parent
        oh_only = ParentCandidate("bgc1", "GCF_1", ("Thr", "Lys", "Glu", "Ser", "Ala"), terminal_forms=("OH",))
        variants = enumerate_systematic_variants(oh_only)
        self.assertEqual([variant["c_terminal"] for variant in variants], ["OH"])
        self.assertEqual(variants[0]["variant_class"], "direct_concatenation")
        both = ParentCandidate("bgc2", "GCF_1", ("Thr", "Lys", "Glu", "Ser", "Ala"), terminal_forms=("OH", "NH2"))
        variants = enumerate_systematic_variants(both)
        self.assertEqual([variant["c_terminal"] for variant in variants], ["OH", "NH2"])
        self.assertEqual(variants[1]["variant_class"], "c_terminal_amidation")

    def test_ornithine_parent_yields_configuration_variant(self):
        parent = ParentCandidate("bgc2", "GCF_1", ("Thr", "Orn", "Glu", "Ser", "Ala"),
                                 terminal_forms=("OH",), has_ornithine=True)
        variants = enumerate_systematic_variants(parent)
        self.assertEqual(len(variants), 2)
        self.assertEqual(variants[1]["variant_class"], "orn_configuration")
        self.assertEqual(variants[1]["variant_detail"], "first_Orn_to_D-Orn")

    def test_lipid_rules_follow_frozen_table(self):
        self.assertEqual(len(LIPID_VARIANT_RULES), 7)
        parent = ParentCandidate("bgc3", "GCF_1", ("Thr", "Lys", "Glu", "Ser", "Ala"),
                                 terminal_forms=("OH",), lipid_rule="3R-OH-Myr")
        variants = enumerate_systematic_variants(parent)
        self.assertEqual(variants[-1]["variant_class"], VARIANT_CLASS_BETA_HYDROXY_FATTY_ACYL)
        self.assertEqual(variants[-1]["c_terminal"], "OH")
        self.assertEqual(variants[-1]["lipid_acyl"], "3R-OH-Myr")
        amidated = ParentCandidate("bgc4", "GCF_1", ("Thr", "Lys", "Glu", "Ser", "Ala"),
                                   terminal_forms=("OH",), lipid_rule="Myr-NH2")
        variants = enumerate_systematic_variants(amidated)
        self.assertEqual(variants[-1]["variant_class"], VARIANT_CLASS_FATTY_ACYLATED_PLUS_AMIDATION)
        self.assertEqual(variants[-1]["c_terminal"], "NH2")
        self.assertEqual(variants[-1]["lipid_acyl"], "Myr")
        self.assertEqual(
            display_sequence(("Thr", "Lys"), "NH2", "Myr"), "H-Thr-Lys(ε-Myr)-NH2",
        )
        self.assertEqual(VARIANT_CLASS_FATTY_ACYL_REFERENCE, "fatty_acyl_reference")
        self.assertEqual(VARIANT_CLASS_FATTY_ACYL_CHAIN_LENGTH, "fatty_acyl_chain_length")

    def test_no_cartesian_expansion(self):
        parent = ParentCandidate(
            "bgc4", "GCF_1", ("Thr", "Orn", "Lys", "Ser", "Ala"),
            terminal_forms=("OH", "NH2"), has_ornithine=True, lipid_rule="Lau",
        )
        self.assertEqual(len(enumerate_systematic_variants(parent)), 4)

    def test_candidate_ids_and_constrained_variants(self):
        parents = [
            ParentCandidate("bgc1", "GCF_1", ("Thr", "Lys", "Glu", "Ser", "Ala"), terminal_forms=("OH",)),
            ParentCandidate("bgc2", "GCF_2", ("Thr", "Orn", "Glu", "Ser", "Ala"),
                            terminal_forms=("OH",), has_ornithine=True),
        ]
        constrained = [{
            "bgc_id": "bgc3", "genome_accession": "GCF_3",
            "core_sequence": "Thr-Lys-Glu-Ser-Ala",
            "displayed_sequence": "H-Thr-Lys-Glu-Ser-Ala-OH",
        }]
        records = build_candidate_records(parents, constrained)
        self.assertEqual(len(records), 1 + 2 + 1)
        self.assertEqual(records[0].candidate_id, "BNP_000001")
        self.assertEqual(records[-1].candidate_id, f"BNP_{len(records):06d}")
        self.assertTrue(records[-1].is_constrained_variant)
        self.assertEqual(records[-1].variant_class, "constrained_variant")

    def test_manuscript_library_invariants(self):
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["parent_bgc_count"], 12_266)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["systematic_candidate_records"], 23_540)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["constrained_variant_records"], 29)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["constrained_variant_parents"], 19)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["unique_core_sequences"], 3_713)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["unique_displayed_sequences"], 5_303)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["nonstandard_or_modified_records"], 20_579)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["explicit_lipidation_records"], 1_625)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["direct_concatenation_records"], 12_035)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["c_terminal_amidation_records"], 9_076)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["orn_configuration_records"], 783)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["lipid_rules"], 7)
        self.assertEqual(SYSTEMATIC_LIBRARY_INVARIANTS["parents_per_lipid_rule"], 231)


class TestPhysicochemicalAnnotation(unittest.TestCase):
    def test_molecular_weight_backbone_and_terminal_corrections(self):
        # Gly-Gly free acid: 2 * 57.05 + water
        self.assertAlmostEqual(molecular_weight(("Gly", "Gly"), "OH"), 2 * 57.05 + 18.015)
        amidated = molecular_weight(("Gly", "Gly"), "NH2")
        self.assertAlmostEqual(
            molecular_weight(("Gly", "Gly"), "OH") - amidated, 17.007 - 16.019,
        )

    def test_lipidation_adds_acyl_minus_hydrogen(self):
        base = molecular_weight(("Ala", "Lys"), "OH")
        lipidated = molecular_weight(("Ala", "Lys"), "OH", lipid_acyl="Myr")
        self.assertAlmostEqual(lipidated - base, 210.36 - 1.008)

    def test_lipidated_lys_side_chain_neutralized(self):
        charge_free = net_charge_ph7(("Lys",), "OH")
        charge_lipidated = net_charge_ph7(("Lys",), "OH", lipidated_lysines=frozenset({0}))
        self.assertGreater(charge_free, 0.9)
        self.assertLess(charge_lipidated, charge_free - 0.9)

    def test_amidated_c_terminus_loses_carboxyl_charge(self):
        # amidation removes the negative carboxylate, so the charge rises
        self.assertLess(net_charge_ph7(("Ala",), "OH"), net_charge_ph7(("Ala",), "NH2"))
        self.assertAlmostEqual(net_charge_ph7(("Ala",), "OH"), -0.009890990199008853)

    def test_isoelectric_point_defined_for_acidic_peptide(self):
        pI = isoelectric_point(("Glu",), "OH")
        self.assertIsNotNone(pI)
        self.assertLess(pI, 7.0)

    def test_isoelectric_point_missing_without_zero_crossing(self):
        # strongly basic peptide stays positive across pH 0-14
        self.assertIsNone(isoelectric_point(("Lys", "Arg"), "NH2"))

    def test_gravy_uses_kyte_doolittle_and_excludes_lipids(self):
        self.assertAlmostEqual(gravy(("Ile", "Val")), (4.5 + 4.2) / 2)
        self.assertIsNone(gravy(("Orn",)))

    def test_aromatic_fraction(self):
        self.assertAlmostEqual(aromatic_fraction(("Phe", "Tyr", "Ala", "Trp")), 0.75)

    def test_annotation_block_keys(self):
        parent = ParentCandidate("bgc1", "GCF_1", ("Thr", "Lys", "Glu", "Phe", "Ala"), terminal_forms=("OH",))
        record = build_candidate_records([parent])[0]
        annotation = annotate_physicochemical(record, parent.core_monomers)
        self.assertEqual(
            set(annotation),
            {"length", "molecular_weight", "net_charge_pH7", "isoelectric_point", "gravy", "aromatic_fraction"},
        )
        self.assertEqual(annotation["length"], 5)


if __name__ == "__main__":
    unittest.main()
