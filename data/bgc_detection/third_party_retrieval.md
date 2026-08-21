# Third-party genome/BGC resource retrieval record

Third-party resources are not redistributed in this repository. This file records
the source, version and retrieval route for every external resource used by the
BGC detection module, following the final manuscript (Table S2a) and the
publication-data policy in `docs/MANUSCRIPT_RELEASE_CONTRACT.md`.

| Resource | Version/snapshot | Role | Retrieval route |
| --- | --- | --- | --- |
| NCBI RefSeq bacterial genomes | Project-frozen accession set (56,709 genomes) | Genome-scale screening input | Accession list: `data/genomes/refseq_genome_inventory.tsv`. Assemblies retrievable via NCBI Datasets using the accession column. |
| MIBiG | v4.0 (2,636 GenBank records) | Known-BGC positive reference and novelty context | https://mibig.secondarymetabolites.org/ (v4.0 GBK bundle). Project processing produced 1,984 modelling BGCs (Supplementary Method 3). |
| GeneSwap negatives | DeepBGC v0.1.0 prepared training snapshot | Negative supervision | Released with DeepBGC v0.1.0; 1,984 negatives used after family- and Pfam-length-stratified subsampling (Supplementary Methods 3, 7). |
| Pfam-A | Pfam HMM library used by pyhmmer 0.12.1 (E-value 1e-5) | Domain annotation | https://www.ebi.ac.uk/interpro/download/ (Pfam-A HMM). |
| pfam2vec | Frozen DeepBGC/pfam2vec representation (102-dimensional) | Input embedding | Derived from the DeepBGC v0.1.0 pfam2vec corpus/framework; frozen embedding table archived by the project and distributed as `pfam2vec.csv.gz` in the DeepBGC supplementary materials. |

Notes:

- The MIBiG v4.0 publication reports a curated-entry total of 3,059; this project
  used a local file-level snapshot of 2,636 GenBank records (Manuscript Methods 2.2).
- BGC region identifiers stored in project tables use the frozen 43-character
  truncated form `{genome}__{contig}_{genome prefix}...region{NNN}`; parsed
  provenance fields (genome accession, contig label, region label) are provided
  alongside every table and constitute the non-abbreviated provenance of record.
