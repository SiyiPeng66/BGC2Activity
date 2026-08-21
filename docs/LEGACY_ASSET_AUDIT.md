# Legacy-asset audit

This is an inventory, not a release claim. It records the first-pass relation between the final manuscript and materials in the surrounding historical workspace.

## Potentially useful, but not yet accepted

| Legacy location | Observed content | Release decision |
| --- | --- | --- |
| `Manuscripts/V3/补充材料/最终待筛选多肽文库_0722.xlsx` | 23,540 candidate rows; 3,713 core sequences; 5,303 displayed sequences; 1,625 lipidated records | **Blocked:** 12,270 distinct `BGC_ID` values, whereas final Supplementary Method 13 requires 12,266. Must be reconciled from source reconstruction, not edited by hand. |
| `Manuscripts/V3/补充材料/最终附表/Supplementary_Tables_V3_core.xlsx` | Complete 80 x 5 MIC table reproduces 8/40 vs 18/40 screening hits and 3/40 vs 12/40 potent hits at <=64 µg mL-1 | **Partial:** contains MIC endpoints but no complete 80-candidate mapping to the 23,540-record library, no frozen scores/ranks and no Round-2 freeze package. |
| `external_datasets/Nonstandard_AMP_databases_redownload_20260721/` | Candidate public-data snapshot and processed intermediate tables | **Quarantined:** source/version and redistribution permissions must be audited against the manuscript's 2026-07-21 snapshot and the stated 2025-12-21 public-peptide freeze. |
| `EvoBGC/` | pfam2vec data flow and detector prototypes | **Reference only:** existing defaults, imports, masking and result artifacts are not sufficient evidence of the final model described in Supplementary Methods 4-7. |
| `sequence_chemical_pretraining_real/` | chemical-aware pretraining prototype | **Reference only:** its README explicitly calls the compact descriptors an engineering approximation. It cannot represent the final multimodal model without reconstruction and validation. |
| `nrp_amp_ranker/` | parser, feature and ranking prototypes | **Reference only:** legacy code has incompatible activity-model and feedback assumptions; it must not be relabelled as the final Base/Calibrated Model. |

## Explicit exclusions

The following kinds of legacy materials may never enter the release as scientific output:

- files/directories carrying `SIMULATED_LAYOUT_DEVELOPMENT_ONLY`, `SIMULATED_LAYOUT_ONLY_NOT_A_MODEL_OUTPUT`, `template`, `mockup`, `demo` or equivalent markers;
- `Manuscripts/V3/补充材料/08_Pretraining_quantitative_effectiveness`, `11_Base_Model_independent_performance` and `12_Calibrated_Model_OOF_reliability` simulation input folders, unless replaced with frozen real outputs;
- smoke runs, preview figure variants and copied baseline repositories;
- anonymous/truncated BGC identifiers where the full genomic coordinate cannot be recovered.

## Reconciliation tasks before data promotion

1. Rebuild the candidate library from the frozen parent set and identify the four extra legacy BGC identifiers; regenerate the 23,540-row table and hash it.
2. Resolve the public-peptide freeze-date conflict before any raw/processed data release: the main manuscript's Data availability statement says 2025-12-21, whereas Supplementary Method 9 says 2026-07-21.
3. Map every Round-1 and Round-2 candidate to a stable candidate-library ID, complete BGC provenance and full pre-freeze score/rank record.
4. Recover or reconstruct the ten-file Round-2 freeze package specified in Supplementary Method 17.
5. Locate and verify the complete 384-derivative design table, the 20-compound local MIC/haemolysis data, NRP45-A07 in-vitro data and implant-model data.
6. Reconstruct final model code/configuration from the methods and validate it against locked outputs; do not infer checkpoints from prototype outputs.
