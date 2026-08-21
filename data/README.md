# Release core data

Study-generated data committed as plain UTF-8 TSV. Every file is listed in
`manifests/data_checksums.tsv` with its SHA-256 digest. All tables were exported
deterministically from the frozen legacy source workbooks by
`scripts/build_release_data.py`, which also validates the manuscript-fixed counts.

## Directory map

| Path | Content | Rows | Manuscript anchor |
| --- | --- | --- | --- |
| `genomes/refseq_genome_inventory.tsv` | 56,709 screened RefSeq genome accessions with keyword and NCBI taxonomy | 56,709 | Table S3; Methods 2.2 |
| `bgc_detection/nrps_positive_bgc_index.tsv` | NRPS-positive BGC regions with nearest MIBiG reference and reference-relative distance | 258,750 | Results 3.2; Supplementary Method 6 |
| `bgc_detection/substrate_predicted_bgc_index.tsv` | BGCs with ordered A-domain substrate predictions and product annotation | 77,587 | Results 3.2; Supplementary Methods 6, 8 |
| `bgc_detection/data_resources.tsv` | Data resources and project snapshots | - | Table S2a |
| `bgc_detection/partition_composition.tsv` | Random and evolutionary-family-held-out partition composition | - | Table S2b; Supplementary Method 5 |
| `bgc_detection/third_party_retrieval.md` | Retrieval record for RefSeq/MIBiG/GeneSwap/Pfam/pfam2vec | - | Data availability |
| `candidate_library/candidate_library.tsv` | Standardized NRPS-informed synthetic candidate library | 23,540 | Table S4; Supplementary Methods 8, 13 |
| `candidate_library/library_construction_funnel.tsv` | Construction funnel counts | - | Table S1a |
| `candidate_library/admission_filtering_rules.tsv` | Frozen admission/filtering rules F2-R01..F2-R06 | - | Table S1b |
| `candidate_library/constrained_variant_rules.tsv` | Constrained variant-rule composition | - | Table S1c |
| `activity_modelling/strict_mic_dataset_summary.tsv` | Descriptive strict-label funnel (Supplementary Table S2c); under the regression formulation (DC-1) it serves endpoint summaries only and never trains a model; quantitative log10(MIC) targets are derived at runtime by `bgc2activity.activity.mic_curation` (RD-8) | - | Table S2c; Supplementary Method 11; RD-8 |
| `activity_modelling/public_peptide_retrieval.md` | Retrieval record for the six public peptide databases | - | Data availability |
| `screening/two_round_mic.tsv` | Complete MIC results for the 80 screened candidates (40 per round, five indicator organisms) | 400 | Results 3.3; Supplementary Methods 18, 19 |
| `screening/screening_to_library_mapping.tsv` | Deterministic mapping of screened candidates to candidate-library entry IDs | 80 | Release contract D08 |
| `screening/screening_endpoint_summary.tsv` | Round-level screening/potent/broad-spectrum hit counts and rates | 2 | Results 3.3 |
| `nrp45/` | NRP45 source-genome traceability tables (genome GCF_026229585.1, antiSMASH 7 output, lead-candidate provenance, round summary) | - | Fig. 4 |
| `animal_model/group_assignment.tsv` | Mouse grouping, infection procedures and local dosing | 4 | Table S5; Supplementary Method 23 |

## Candidate library identifiers

`candidate_library.tsv` carries the frozen 23,540 records. Column semantics:

- `entry_id`: stable candidate identifier (BNP_000001...).
- `bgc_id`: frozen parent BGC identifier as committed in the final Table S4
  (43-character truncated form, see below); parsed into `genome_accession`,
  `contig_label` and `region_label`.
- `is_constrained_variant`: 29 project-defined constrained variants derived from
  19 parent BGCs (Supplementary Method 13).
- `is_nonstandard_or_modified`: displayed candidate carries a nonstandard monomer
  or modification signal (20,579 records).
- `is_lipidated`: displayed candidate carries explicit lipidation (1,625 records).

Library-level invariants: 12,266 systematic parent BGCs; 12,270 unique BGC
identifiers library-wide because 4 of the 19 constrained-variant parents carry no
systematic record; 3,713 unique core sequences; 5,303 unique displayed sequences;
C-terminal states OH 14,233 / NH2 9,307.

## Known limitations (declared, not edited)

1. **Truncated BGC identifiers.** Every frozen `BGC_ID` is stored in the
   43-character truncated form `{genome}__{contig}_{genome-prefix}...region{NNN}`
   inherited from the final manuscript tables. The full region identifiers were not
   preserved in any project artifact. Parsed provenance fields are complete and are
   the provenance of record; the truncated key is retained verbatim so the release
   matches the published tables.
2. **Terminal-form ambiguity for 21 screened candidates.** The wet-lab sequence
   labels do not encode the C-terminal state. 59/80 screened candidates map to a
   unique library entry; 21 map to two entries that differ only by OH/NH2 terminal
   form (`screening_to_library_mapping.tsv`, `match_status`). The original
   selection list resolving these terminals is among the lost final artifacts.
3. **Missing model artifacts.** Base MIC Regression Model score exports, five-fold
   update/calibration outputs, the Round-2 ten-file freeze package, the 384-NRP45
   derivative design table, the 20-compound local MIC/haemolysis table and
   NRP45-A07 replicate-level in-vitro/implant data are not present in the source
   workspace; see `data/manifests/release_contract.tsv`.
4. **Documentation conflicts.** Freeze-date (2025-12-21 vs 2026-07-21) and
   database-version (DRAMP/APD/dbAMP naming) conflicts between the main-text Data
   availability statement and Supplementary Tables are recorded in
   `activity_modelling/public_peptide_retrieval.md`.
   The regression-versus-classification discrepancy between the final main text
   and Supplementary Methods 11-16 (DC-1) was resolved by author decision
   2026-08-20 in favour of the regression formulation; provenance and
   reconstruction decisions RD-7-RD-13 are recorded in
   `docs/MANUSCRIPT_RELEASE_CONTRACT.md`.
