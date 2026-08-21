# Manuscript-to-repository release contract

## Authority and purpose

This contract was transcribed from the final manuscript `Manuscript_SB_0813.docx` and final supplementary material `Supplementary_Materials_V2.docx` in the parent workspace on 2026-08-19. Those documents are the sole scientific specification for this release build. Historical scripts, figures, tables and checkpoints can supply implementation clues only; they cannot alter this contract.

The final manuscript's Data availability statement names `https://github.com/SiyiPeng66/BGC2Activity`. This local directory is the clean reconstruction target for that repository. It has not been connected to a remote.

## Non-negotiable interpretation rules

- BGC score, reconstruction confidence and novelty are **admission and provenance** metadata. They must not be used as input variables to an activity probability.
- Chemical-aware pretraining provides an initial representation; it is not the antimicrobial endpoint and does not by itself demonstrate activity.
- `Sbase` (base predicted log10(MIC)), `Supd` (residual-updated prediction) and `Scal` (monotone-calibrated prediction) are distinct stored quantities. Round-1 data may update/calibrate only the latter two; the Base MIC Regression Model remains frozen. Prespecified MIC thresholds (<= 256 screening, <= 64 potent) define experimental endpoints only and never enter training labels.
- Round 2 is an independently frozen, prospective cohort. Its sequential comparison with Round 1 is descriptive enrichment, not evidence of a causal feedback effect.
- Candidate structures are NRPS-informed compounds made by chemical synthesis. Neither a candidate nor NRP45/NRP45-A07 may be described as an endogenous natural product of the inferred source organism.

## Required public release units

| Unit | Manuscript-defined contents | Acceptance requirement |
| --- | --- | --- |
| Genome/BGC provenance | 56,709 RefSeq accessions; MIBiG v4.0; family partitions; Pfam/pfam2vec provenance | Frozen accession/version inventory, retrieval instructions, hashes, and split assignments. |
| BGC detector | 102-dimensional pfam2vec input; evolutionary-family holdout; four stacked Time-mixing/Channel-mixing blocks; 128 hidden units; eight-expert MoE; detection threshold 0.45 | Runnable train/inference code, exact configuration, split manifest, metrics and only the approved checkpoint (or documented checkpoint-access route). |
| Candidate reconstruction | 77,587 ordered substrate-predicted BGCs; confidence >=0.8; >=5 valid modules; 12,266 parents; 23,540 records | One row per candidate with non-abbreviated BGC provenance, reconstruction inputs, variant rule, displayed structure and traceable parent relation. |
| Peptide representation | Supplementary Method 9 reports a 2026-07-21 source freeze, 118,630 parsed records, 64,127 chemical entities and a 47-token vocabulary | Resolve the conflicting 2025-12-21 freeze date in the main Data availability statement; then provide rebuild scripts, source-specific inventory and licences/access rules. Do not bundle unreviewed third-party data. |
| Quantitative MIC supervision | 146,547 harmonised measurements; continuous log10(MIC) targets per record and per entity (RD-8); the strict-label funnel (61,545 records; 7,358 sequences; 4,739 nonstandard/modification subset) is retained as the descriptive Supplementary Table S2c summary only | Row-level provenance, units/comparator handling, quantitative-target rules, cluster split and target-generation test; strict labels tested as descriptive endpoints, never as training labels. |
| Base MIC Regression Model | Pretrained representation plus predefined chemical features, scalar regression head on log10(MIC), frozen before Round 1 | Training/fine-tuning configuration, cluster split, seed/selection record, regression metrics (MAE/RMSE/Spearman/agreement/residuals), score exports and a clear distinction between `Sbase`, `Supd` and `Scal`. |
| Sequential screening | Round 1: 40 candidates, 8 screening hits; Round 2: 40 new candidates, 18 hits; potent-hit threshold <=64 µg mL-1 gives 3/40 and 12/40 | Candidate-to-library mapping, archived scores/ranks, selection constraints, complete MIC table and frozen Round-2 package. |
| Update/calibration | Nine pre-specified features; ridge residual `Supd = Sbase + beta^T x` (Sbase coefficient 1); five-fold cross-fitting; monotone (isotonic) regression-scale calibration `Scal` | Fold assignments, out-of-fold scores, fitted coefficients, diagnostics, final frozen model, code commit/runtime/hash records. |
| Local optimisation | 384 non-redundant NRP45 derivatives (T1-T4: 54/126/168/36); 20 synthesised with MICs; Pareto selection of NRP45-A07 | Complete design table, prioritisation scores/ranks, MIC and haemolysis source data, subtype membership and selection calculation. |
| In-vitro/in-vivo validation | anti-MRSA, time-kill, resistance passage, compatibility, membrane/biofilm and implant-model outcomes | Tidy raw-or-replicate-level data, analysis scripts, figure-source tables and a readme declaring experimental units, exclusions and statistics. |

## Publication-data policy

Study-generated data should be committed as plain UTF-8 TSV/CSV with a data dictionary, source filename, SHA-256 checksum and a stable identifier. Release only values necessary to reproduce reported analyses; retain raw instrument files separately if size, ethics or supplier licensing makes GitHub unsuitable, with an explicit accession/access path.

For DBAASP, DRAMP, APD3, CAMPR4, dbAMP, SATPdb, NCBI RefSeq, MIBiG, Pfam and DeepBGC/pfam2vec-derived resources, the release must preserve source, version/snapshot date, retrieval record, transformations and licences. A download/rebuild recipe is preferable to redistributing a database dump unless redistribution is explicitly permitted.

## Release gates

Before a public push, every required artifact must have status `accepted` in `data/manifests/release_contract.tsv`; all manuscript counts and endpoint thresholds must be tested; no ignored/local-only file may be referenced by a public workflow; and a clean environment must reproduce tables/figures from the released inputs. A versioned freeze manifest must record code commit, runtime, input/output checksums and UTC timestamp.

## Reconstruction decisions (finalized under the project-reasonableness principle)

Where the final documents under-specify or truncate an algorithmic detail, the reconstruction fixes one auditable choice instead of leaving silent defaults. Each decision is implemented, named in code and covered by tests.

| ID | Manuscript anchor | Decision |
| --- | --- | --- |
| RD-1 | Method 16 (truncated sentence) | SUPERSEDED by the DC-1 author decision (regression formulation); the regression analogue is RD-12. Historical text: development-stage `Pcal` used per-fold Platt mappings fitted on the out-of-fold `Supd` of the 32 training records of the same fold. |
| RD-2 | Method 15 (class-imbalance weighting named, formula not given) | The prespecified Method-12 weights `w_k = N_train / (2 * N_k,train)` are applied within every training fold of the residual update and its inner regularization selection. |
| RD-3 | Method 13 (terminal forms are pre-frozen, rule table lost) | Terminal-form admissibility is a per-parent input; the enumeration never assumes a Cartesian OH x NH2 product (released composition: 12,035 OH + 9,076 NH2 base records over 12,266 parents). The seven lipid rules are frozen verbatim from Table S1c, including the NH2-terminal Myr-NH2 rule. |
| RD-4 | Method 10.1 (47-token vocabulary) | 20 standard + 21 library-confirmed nonstandard + 3 special tokens are implemented; the remaining 3 corpus-only nonstandard tokens fixed by Supplementary Table S1 are a declared gap (`CORPUS_ONLY_NONSTANDARD_MONOMERS`) and are not fabricated. |
| RD-5 | Method 6.3 (single vs meta gene prediction) | Complete genomes run pyrodigal single mode; the mode for a FASTA without an explicit flag is selected from the longest sequence length (>= 10 kb -> single). Declared reconstruction heuristic. |
| RD-6 | Method 12 (five fixed seeds not enumerated) | `ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)`; the manuscript fixes the count (5) and the averaging rule, not the seed values. |
| RD-7 | Main text 2.3 (regression loss not named) | Base-model loss is Huber (smooth L1, delta = 1.0 log10 unit). |
| RD-8 | Method 11 under the regression formulation | Per-record log10(MIC) targets: point estimates as-is; `<`/`<=` and `>`/`>=` anchors kept with left/right censoring flags; intervals use the geometric mean; unit buckets never converted; entity-level target = minimum log10(MIC) within the preferred ug_mL bucket (uM fallback). |
| RD-9 | Main text 3.3 / Fig. 3 (Round-1 update target across five strains) | The Round-1 update target is log10 of the minimum numeric MIC over indicator strains (most potent), mirroring the entity-level rule of RD-8. |
| RD-10 | Method 15 analogue under regression | Regularization C is selected by lowest inner-CV RMSE (replacing inner-CV log loss); the effective-dof cap of 4 and the C grid are unchanged. |
| RD-11 | Method 15 "five stratified folds" with a continuous target | Folds are stratified on quartile bins of the observed log10(MIC) target. |
| RD-12 | Main text 2.3 "monotone regression-scale calibration" | Isotonic (pool-adjacent-violators) regression maps Supd to calibrated predicted log10(MIC); development values use per-fold isotonic fits on the out-of-fold Supd of the 32 training records of the same fold (regression analogue of RD-1); the final mapping is fitted on the 40 out-of-fold Supd values; inputs truncated at the 0.5-99.5 percentiles. |
| RD-13 | Main text 2.3 (selection criterion not named) | Base-model epoch selection is validation RMSE; ties within 1e-4 broken by higher validation Spearman. |

## Documented manuscript discrepancies

| ID | Locations | Discrepancy | Interim reconstruction stance |
| --- | --- | --- | --- |
| DC-1 | Main text §2.3, §3.1, §3.3, §3.4 and Fig. 3 vs Supplementary Methods 11-16 and Suppl. Fig. 3 caption | The final main text describes Model 2 as quantitative MIC regression: a "Base MIC Regression Model" trained with log10(MIC) as a continuous target and a scalar regression head, evaluated by MAE/RMSE/Spearman/observed-versus-predicted agreement/residual structure, updated by "a low-capacity residual regression update followed by monotone regression-scale calibration", and states that prespecified MIC thresholds "were not used as binary labels for training the Base Model or feedback updater". Supplementary Methods 11-16 instead specify strict binary labels (MIC <= 8 positive, > 64 negative), a binary classification head trained with weighted binary cross-entropy and AUPRC selection, a logistic residual update selected by inner-CV log loss, and a monotone Platt probability `Pcal`; the Suppl. Fig. 3 caption reports AUROC/AUPRC while the Suppl. Figs. 5/7/8 captions use the regression framing. Provenance: the v2 main-text draft (Manuscript_SB_V2.docx) matches the classification framing ("score update and probability calibration"; "predicted-positive set from 473 candidates (2.0%) to 380 (1.6%)"); the v3 rewrite (Manuscript_SB_0813.docx) introduced the regression framing; the supplement body is text-identical between v2 and v3; the regression supplementary figures existed as internal demo drafts on 2026-08-10 (parent workspace `Manuscripts/V3/补充材料/_internal_mic_regression_demo/`: "All TSV files and metrics are demonstration data and must be replaced with locked MIC results before submission"); the final supplementary-figure folders `08_Pretraining_quantitative_effectiveness`, `11_Base_Model_independent_performance` and `12_Calibrated_Model_OOF_reliability` likewise plot `source_data_simulated` inputs marked "replace before submission", i.e. the regression formulation has no finalised quantitative outputs in the workspace, whereas the classification formulation matches the historical implementation and the real Round-1/Round-2 records. | Resolved by author decision 2026-08-20: the released code implements the regression formulation (continuous log10(MIC) target, scalar regression head, ridge residual update, isotonic regression-scale calibration; RD-7-RD-13). The classification body of Supplementary Methods 11-16 remains a documented internal inconsistency of the final supplement; its strict-label funnel is retained only as the descriptive Supplementary Table S2c summary |
