"""Contract tests for the genome annotation layer (Supplementary Method 6.3).

Only the dependency-free members are tested here (domain ordering and
translation helpers); pyrodigal/pyhmmer entry points are exercised in
environments with the annotation extra installed.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bgc2activity.detection.annotation import (
    META_MODE_LENGTH_THRESHOLD,
    PFAM_E_VALUE,
    PYHMMER_VERSION,
    PYRODIGAL_VERSION,
    GeneCall,
    PfamHit,
    _reverse_complement,
    _translate,
    ordered_pfam_sequence,
)


def make_gene(gene_id, contig_id, start, end, strand):
    return GeneCall(gene_id, contig_id, start, end, strand, protein_sequence="MAK")


class TestAnnotationContract(unittest.TestCase):
    def test_manuscript_fixed_versions_and_threshold(self):
        self.assertEqual(PYRODIGAL_VERSION, "3.7.1")
        self.assertEqual(PYHMMER_VERSION, "0.12.1")
        self.assertEqual(PFAM_E_VALUE, 1e-5)
        self.assertEqual(META_MODE_LENGTH_THRESHOLD, 10_000)

    def test_domain_ordering_by_gene_orientation_and_position(self):
        genes = [
            make_gene("g1", "contig_1", 100, 900, strand=1),
            make_gene("g2", "contig_1", 1000, 1900, strand=-1),
        ]
        hits = [
            PfamHit("PF00002", "g1", "g1", envelope_start=300, envelope_end=400, bitscore=50.0),
            PfamHit("PF00001", "g1", "g1", envelope_start=100, envelope_end=200, bitscore=60.0),
            PfamHit("PF00003", "g2", "g2", envelope_start=100, envelope_end=200, bitscore=70.0),
            PfamHit("PF00004", "g2", "g2", envelope_start=300, envelope_end=400, bitscore=40.0),
        ]
        ordered = ordered_pfam_sequence(hits, genes)
        # forward gene: intra-protein ascending; reverse gene: descending
        self.assertEqual(ordered, ["PF00001", "PF00002", "PF00004", "PF00003"])


class TestTranslationHelpers(unittest.TestCase):
    def test_forward_translation(self):
        sequence = "ATGGCCAAA"  # M A K
        self.assertEqual(_translate(sequence, 1, 9, strand=1), "MAK")

    def test_reverse_strand_translation(self):
        sequence = "ATGGCCAAA"
        self.assertEqual(_reverse_complement(sequence), "TTTGGCCAT")
        # reverse-strand gene covering the whole sequence
        self.assertEqual(_translate(sequence, 1, 9, strand=-1), _translate("TTTGGCCAT", 1, 9, strand=1))

    def test_partial_interval(self):
        sequence = "NNATGNN"
        self.assertEqual(_translate(sequence, 3, 5, strand=1), "M")


if __name__ == "__main__":
    unittest.main()
