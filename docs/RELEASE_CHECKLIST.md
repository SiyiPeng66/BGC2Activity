# Pre-publication checklist

This checklist accompanies the machine-readable release contract. It does not override it.

- [x] Resolve the 12,266 versus 12,270 parent-BGC discrepancy and regenerate the candidate library from frozen inputs. (Reconciled: 12,266 systematic parents; 4 constrained-variant-only parents; documented in data/README.md.)
- [ ] Resolve the 2025-12-21 versus 2026-07-21 public-peptide freeze-date discrepancy in the final documentation.
- [x] Recover/rebuild the release TSVs with stable links from Round-1/2 candidates to candidate-library records. (data/screening/screening_to_library_mapping.tsv; 59/80 unique, 21 terminal-form ambiguities declared.)
- [ ] Recover/rebuild final BGC, chemical-pretraining and Base Activity Model code/configurations from locked inputs and verify outputs. (All Supplementary-Method 3-25 modules reconstructed and contract-tested — detection incl. annotation and nine-genome evaluation, corpus curation, reconstruction/library, clustering/training, calibration, screening, experimental and statistical analysis; locked inputs/checkpoints still unrecovered.)
- [ ] Archive the complete five-fold update/calibration outputs and the Method-17 ten-role freeze package.
- [ ] Recover complete NRP45 local-library, MIC, haemolysis, in-vitro and implant-model data with dictionaries and analysis scripts.
- [ ] Check third-party database redistribution licences; replace prohibited copies with source/version/retrieval manifests.
- [ ] Choose a repository licence and complete author/contributor citation metadata.
- [ ] Run `make test` in a clean clone, then reproduce each released table and figure from tracked inputs.
- [ ] Review the final repository diff for untracked raw data, personal paths, proprietary material, credentials and simulated/template artifacts.
