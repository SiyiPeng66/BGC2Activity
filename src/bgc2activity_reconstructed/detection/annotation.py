"""Genome annotation pipeline for genome-scale inference (Method 6.3).

RefSeq nucleotide sequences are processed with pyrodigal 3.7.1 for gene
prediction (single mode for complete genomes, meta mode for short sequences),
followed by pyhmmer 0.12.1 searches against Pfam-A HMMs (E-value 1e-5). Each
Pfam hit records the accession, intra-protein coordinates, bitscore and source
gene; multiple non-redundant hits within the same protein enter the model
independently. Genes without Pfam detections are retained in the gene table
for spacer statistics but do not serve as model input.

The heavy third-party dependencies (pyrodigal, pyhmmer) are imported lazily so
this module can be imported in dependency-free environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PYRODIGAL_VERSION = "3.7.1"
PYHMMER_VERSION = "0.12.1"
PFAM_E_VALUE = 1e-5
META_MODE_LENGTH_THRESHOLD = 10_000  # contigs shorter than this use meta mode


@dataclass(frozen=True)
class GeneCall:
    gene_id: str
    contig_id: str
    start: int
    end: int
    strand: int
    protein_sequence: str


@dataclass(frozen=True)
class PfamHit:
    pfam_accession: str
    protein_id: str
    gene_id: str
    envelope_start: int
    envelope_end: int
    bitscore: float


def predict_genes(fasta_path: str | Path, complete_genome: bool | None = None) -> list[GeneCall]:
    """Run pyrodigal gene prediction on one nucleotide FASTA.

    complete_genome=None selects single vs meta mode automatically from the
    longest sequence length (manuscript: single mode for complete genomes,
    meta mode for short sequences).
    """
    import pyrodigal

    records = _read_fasta(Path(fasta_path))
    if complete_genome is None:
        complete_genome = max((len(seq) for _header, seq in records), default=0) >= META_MODE_LENGTH_THRESHOLD
    finder = pyrodigal.GeneFinder(meta=not complete_genome)
    genes: list[GeneCall] = []
    for index, (contig_id, sequence) in enumerate(records):
        predictions = finder.find_genes(sequence)
        for gene_index, (start, end, strand, _score) in enumerate(predictions):
            protein = _translate(sequence, start, end, strand)
            genes.append(
                GeneCall(
                    gene_id=f"{contig_id}_{gene_index + 1}",
                    contig_id=contig_id,
                    start=start,
                    end=end,
                    strand=strand,
                    protein_sequence=protein,
                )
            )
    return genes


def scan_pfam(genes: list[GeneCall], pfam_hmm_path: str | Path, e_value: float = PFAM_E_VALUE) -> list[PfamHit]:
    """Scan predicted proteins against Pfam-A HMMs with pyhmmer."""
    import pyhmmer
    from pyhmmer.easel import DigitalSequence
    from pyhmmer.plan7 import HMMFile

    hits: list[PfamHit] = []
    alphabet = None
    with HMMFile(str(pfam_hmm_path)) as hmm_file:
        hmms = list(hmm_file)
        alphabet = hmms[0].alphabet if hmms else None
        sequences = []
        gene_by_protein: dict[str, GeneCall] = {}
        for gene in genes:
            if not gene.protein_sequence:
                continue
            digital = DigitalSequence(gene.protein_sequence.encode(), alphabet=alphabet, name=gene.gene_id.encode())
            sequences.append(digital)
            gene_by_protein[gene.gene_id] = gene
        for top_hits in pyhmmer.hmmsearch(hmms, sequences, E=e_value, cpus=1):
            pfam_accession = top_hits.query_accession.decode() if top_hits.query_accession else top_hits.query_name.decode()
            for hit in top_hits:
                if hit.evalue > e_value:
                    continue
                gene = gene_by_protein[hit.name.decode()]
                hits.append(
                    PfamHit(
                        pfam_accession=pfam_accession,
                        protein_id=hit.name.decode(),
                        gene_id=gene.gene_id,
                        envelope_start=hit.env_from,
                        envelope_end=hit.env_to,
                        bitscore=float(hit.score),
                    )
                )
    return hits


def ordered_pfam_sequence(hits: list[PfamHit], genes: list[GeneCall]) -> list[str]:
    """Order Pfam domains by genomic coordinates, gene orientation and
    intra-protein domain position (Supplementary Method 3)."""
    gene_order = {gene.gene_id: index for index, gene in enumerate(genes)}
    for hit in hits:
        if hit.gene_id not in gene_order:
            raise ValueError(f"Pfam hit references an unknown gene: {hit.gene_id}")
    ordered = sorted(
        hits,
        key=lambda hit: (
            gene_order[hit.gene_id],
            0 if genes[gene_order[hit.gene_id]].strand >= 0 else 1,
            hit.envelope_start if genes[gene_order[hit.gene_id]].strand >= 0 else -hit.envelope_start,
        ),
    )
    return [hit.pfam_accession for hit in ordered]


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].split()[0]
                chunks = []
            elif header is not None:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate(sequence: str, start: int, end: int, strand: int) -> str:
    """Translate a pyrodigal-style 1-based inclusive gene interval."""
    fragment = sequence[start - 1:end]
    if strand < 0:
        fragment = _reverse_complement(fragment)
    protein = []
    for position in range(0, len(fragment) - 2, 3):
        protein.append(_CODON_TABLE.get(fragment[position:position + 3].upper(), "X"))
    return "".join(protein)


_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]
