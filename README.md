# BGC2Activity

Analysis code and core data for *Hierarchical NRPS-guided global-to-local
discovery and optimisation of synthetic anti-MRSA lipopeptides*
(Peng et al.). The repository implements the complete computational workflow
of the study: genome-scale BGC detection, reconstruction of NRPS-informed
synthetic peptide candidates, chemical-aware peptide representation learning,
quantitative MIC-regression activity modelling with experimental feedback
updating, fixed-budget sequential screening, and local optimisation of the
NRP45 scaffold.

> **Release status — reconstruction in progress.** The final manuscript
> (`Manuscript_SB_0813.docx`) and final supplementary materials
> (`Supplementary_Materials_V2.docx`) are the sole scientific specification of
> this repository. Every public release unit is tracked in
> [`data/manifests/release_contract.tsv`](data/manifests/release_contract.tsv);
> interpretation rules and reconstruction decisions are recorded in
> [`docs/MANUSCRIPT_RELEASE_CONTRACT.md`](docs/MANUSCRIPT_RELEASE_CONTRACT.md).
> Nothing is pushed until every unit is accepted.

## What the study does

1. A multitask BGC detection model (frozen pfam2vec input, Time-mixing /
   Channel-mixing blocks, eight-expert MoE, position threshold 0.45) screens
   56,709 RefSeq genomes under an evolutionary-family-held-out evaluation
   design.
2. NRPS-informed synthetic candidate structures are reconstructed:
   258,750 NRPS-positive BGCs → 77,587 substrate-predicted BGCs →
   12,266 parent BGCs → 23,540 standardised candidate records.
3. Chemical-aware peptide pretraining initialises a frozen
   **Base MIC Regression Model** trained on curated quantitative public MIC
   measurements (log10(MIC) as the continuous target, scalar regression head).
4. Round 1 (40 candidates, 8 screening hits) feeds a low-capacity residual
   regression update plus monotone regression-scale calibration under
   five-fold cross-fitting; the independently frozen Round-2 cohort
   (40 new candidates, 18 hits; potent hits 3/40 → 12/40) is a prospective
   validation, not a causal feedback estimate.
5. Local optimisation around the synthetic NRP45 scaffold (384 constrained
   derivatives, 20 synthesised) yields NRP45-A07, validated against MRSA
   in vitro and in a locally treated implant-associated infection model.

## Two-model architecture

The study contains exactly two learned models; everything else is rule-based
reconstruction, screening, statistics or release tooling. The two model
lineages share no training data, vocabulary or weights.

**Model 1 — BGC detection model** (`src/bgc2activity/detection/`,
Supplementary Methods 3–7). A multitask encoder over Pfam domain sequences in
genomic order (frozen pfam2vec input; Time-mixing / Channel-mixing blocks;
eight-expert mixture-of-experts; detection, product-class and
evolutionary-contrastive objectives) screens 56,709 genomes at threshold 0.45
under an evolutionary-family-held-out evaluation design.

**Model 2 — activity prediction stack** (main text §2.3; Supplementary
Methods 9–17), three successive stages:

1. *Chemical-aware peptide pretraining* (`src/bgc2activity/representation/`,
   Methods 9–10): masked chemical-context pretraining of a 3-layer
   bidirectional Transformer over a 47-token monomer vocabulary with
   BLAKE2b chemical-text + descriptor targets; initialises the Model-2
   encoder.
2. *Base MIC Regression Model* (`src/bgc2activity/activity/`, Method 12
   freeze boundary): the pretrained encoder (first two layers frozen) plus
   chemical-feature fusion and a scalar regression head predicting
   log10(MIC); structural-cluster-disjoint splits; five-seed ensemble;
   outputs the base score `Sbase` (lower = more active).
3. *Feedback update and calibration* (`src/bgc2activity/calibration/`):
   Round-1 measured log10(MIC) values feed the low-capacity residual update
   `Supd = Sbase + beta^T x` (Sbase coefficient fixed at 1, ridge over nine
   pre-specified features, effective-dof cap 4) and a monotone (isotonic)
   regression-scale calibration to `Scal` under five-fold cross-fitting;
   frozen before Round-2 selection.

Non-negotiable rules (see contract): BGC score, reconstruction confidence and
novelty are admission/provenance metadata only and never activity inputs;
`Sbase`, `Supd` and `Scal` are distinct stored quantities; prespecified MIC
thresholds define experimental endpoints only and never enter training;
candidates are NRPS-informed synthetic compounds, never claimed as endogenous
natural products.

## Repository layout

```text
configs/    Manuscript-derived, versioned model and workflow specifications
data/       Release-ready core data (UTF-8 TSV) + manifests + retrieval records
docs/       Release contract, legacy audits, checklist
scripts/    Deterministic pipeline entry points (explicit paths, no legacy defaults)
src/        Reconstructed implementation (package bgc2activity)
tests/      Contract tests + manuscript-count assertions
```

### Modules

| Module | Manuscript anchor | Contents |
| --- | --- | --- |
| `src/bgc2activity/detection/` | Suppl. Methods 3–7 | Pfam-sequence dataset with OOV/masking; Time-mixing (causal linear attention) + Channel-mixing encoder; 8-expert top-2 MoE; detection/class/contrastive heads; multitask loss (λcls 1.0, λevo 0.5, triplet margin 0.5); composite evolutionary distance, single-linkage families, family-held-out splits; windowed inference and region calling (threshold 0.45, valid-position-weighted class probabilities for regions >256 positions, NRPS signature Pfams); genome annotation (pyrodigal 3.7.1 + pyhmmer 0.12.1, Pfam E=1e-5); nine-genome benchmark evaluation (gene-level projection, 0.5 threshold, AUROC/AUPRC, recovery classification, macro-F1/confusion) |
| `src/bgc2activity/representation/` | Suppl. Methods 9–10 | Public-corpus curation (alias normalization, D-config-only-when-explicit, lipidated-Lys tokens per acyl, non-exclusive modification flags, entity merging, 80/10/10 split seed 20260802, 47-token vocabulary census); BLAKE2b chemical-text vectors + 8-d descriptors; 3-layer bidirectional Transformer (96/4/384, pre-LN, dropout 0.10); 30% masking with 80/10/10 scheme; SmoothL1 + 0.20·cos objective; attention statistics pooling; Method-10.3 evaluation (masked recovery, chemical-neighbour retrieval, cluster-filtered k-NN enrichment, linear probe) |
| `src/bgc2activity/activity/` | Main text §2.3; Methods 11–12 | Quantitative MIC curation (unit buckets, comparator/bound/interval targets, entity-level minimum log10(MIC); strict labels retained as descriptive endpoint summaries only); Base MIC Regression Model with manuscript freeze boundary (layers 1–2 + projections frozen; layer 3 + output LN + pooling fine-tuned; recovery head removed); scalar regression head; Huber loss; 5-seed ensemble averaging; structural clustering (similarity ≥0.80 & length ratio ≥0.80 or Tanimoto ≥0.60; 70/15/15 cluster split seed 42); training protocol (AdamW 1e-5/3e-4, wd 1e-4, batch 64, ≤100 epochs, clip 1.0, validation-RMSE selection, patience 12); regression evaluation (MAE, RMSE, Spearman, observed-vs-predicted agreement, residual structure) |
| `src/bgc2activity/calibration/` | Main text §2.3, §3.3 | Ridge residual update `Supd = Sbase + beta^T x` (Sbase coefficient exactly 1; C grid; effective-dof cap ≤ 4; inner-CV RMSE selection; preprocessing nested in folds); five-fold cross-fitting stratified on target quartiles under one shared fold-index generator; development OOF calibrated scores from per-fold isotonic fits on the 32 training records' OOF Supd (RD-12); final monotone (isotonic) regression-scale calibration on the 40 OOF scores with 0.5–99.5 percentile input truncation |
| `src/bgc2activity/reconstruction/` | Suppl. Methods 8, 13 | MIBiG high-similarity filter (>80% identity, ≥80% coverage both sequences); A-domain confidence gate (≥0.8); chemical-resolvability gate (≥5 monomers); bounded variant enumeration with per-parent terminal-form admission and the seven frozen lipid rules of Table S1c (no Cartesian expansion); physicochemical annotation (MW, Henderson–Hasselbalch charge, pI, GRAVY excluding acyls) |
| `src/bgc2activity/screening/` | Suppl. Methods 14, 17–19 | Activity-independent selection stack (ascending predicted log10(MIC) → top-100 → entity dedup, cluster/per-BGC caps, feasibility → 40); two-round MIC endpoint logic with screening (≤256 any strain), broad-spectrum (≥2 organisms) and potent (≤64) hit criteria |
| `src/bgc2activity/analysis/` | Suppl. Methods 20–25 | Endpoint formulas (haemolysis/MHC5, CCK-8 viability, biofilm inhibition/reduction, time-kill, MICn/MIC0, implant burden, histopathology scores); statistics (Wilson CI, Newcombe risk-difference CI, Fisher exact, censored Spearman, paired bootstrap Δρ with cluster resampling, reliability metrics) |
| `src/bgc2activity/optimization/` | Fig. 4 | NRP45 derivative-library audit (384; T1–T4 54/126/168/36; 574.7–928.3 Da; 20 synthesised) |
| `src/bgc2activity/candidates/` | Suppl. Methods 8, 13 | Candidate-library audit contract |
| `src/bgc2activity/release/` | Suppl. Method 17 | Ten-role content-addressed freeze manifest writer |
| `src/bgc2activity/scoring_contract.py` | Methods 15–16 | Input boundary: Sbase + exactly nine pre-specified features; BGC metadata rejected as activity inputs |

### Pipeline entry points

| Script | Purpose |
| --- | --- |
| `scripts/build_release_data.py` | Rebuild committed data tables from the frozen source workbooks with manuscript-count validation and SHA-256 manifests |
| `scripts/audit_release_data.py` | Dependency-free audit of committed data against manuscript invariants |
| `scripts/train_bgc_detector.py` | Model 1 training under the frozen protocol (AdamW 5e-4/wd 1e-3, batch 64, ≤80 epochs, patience 10, seed 42) |
| `scripts/score_bgc_regions.py` | Model 1 inference from an approved detector checkpoint; applies 256-position / 128-stride window aggregation and writes position-level scores |
| `scripts/train_peptide_pretrainer.py` | Chemical-aware pretraining (AdamW 8e-4/wd 1e-3, batch 128, 30 epochs, seed 20260802, lowest-validation-loss checkpoint) |
| `scripts/train_activity_model.py` | One seed of the Base MIC Regression Model (Huber loss on log10(MIC), validation-RMSE selection; run for seeds 42–46) |
| `scripts/score_activity_candidates.py` | Model 2 inference from one or more approved Base MIC Regression checkpoints; writes `Sbase` and ensemble uncertainty without requiring MIC labels |
| `scripts/select_screening_cohort.py` | Apply the frozen selection stack for Round 1 / Round 2 |
| `scripts/feedback_update_and_freeze.py` | Round-1 residual regression update + cross-fitting + monotone calibration + Method-17 freeze manifest |
| `scripts/write_freeze_manifest.py` | Standalone ten-role freeze manifest writer |

## Installation

Requires Python ≥ 3.10. The contract/data-audit layer is dependency-free;
the model modules use an optional scientific stack.

```bash
git clone https://github.com/SiyiPeng66/BGC2Activity
cd BGC2Activity

# package (dependency-free core)
pip install .

# model modules (torch, numpy, pandas, scikit-learn, openpyxl)
pip install .[models]

# genome annotation stack with manuscript-fixed versions
pip install .[annotation]
```

For development without installation, put `src/` on `PYTHONPATH`
(the `Makefile` targets do this automatically).

## Usage

All entry points take explicit paths; there are no legacy defaults.

```bash
# contract tests + data audit (dependency-free)
make test
make audit-data

# full suite including torch modules
make test-full

# Model 1: BGC detector
python scripts/train_bgc_detector.py \
  --pfam2vec pfam2vec.csv \
  --train-samples train.tsv --validation-samples validation.tsv \
  --output-dir out/detector

# Model 1 inference on ordered Pfam-region sequences
python scripts/score_bgc_regions.py \
  --pfam2vec pfam2vec.csv --checkpoint out/detector/detector_checkpoint.pt \
  --regions ordered_regions.tsv --output out/detector/region_scores.tsv

# Model 2, stage 1: chemical-aware pretraining
python scripts/train_peptide_pretrainer.py \
  --train-corpus corpus_train.tsv --validation-corpus corpus_val.tsv \
  --output-dir out/pretrainer

# Model 2, stage 2: Base MIC Regression Model (one seed per run; 42-46)
python scripts/train_activity_model.py \
  --input-table mic_entities.tsv --seed 42 \
  --pretrainer-checkpoint out/pretrainer/pretrainer_checkpoint.pt \
  --output-dir out/activity

# Model 2 inference (one or more seed checkpoints; normally provide all 5)
python scripts/score_activity_candidates.py \
  --input-table candidate_features.tsv \
  --checkpoints out/activity/base_regression_model_seed42.pt \
  --manifests out/activity/training_manifest_seed42.json \
  --output out/activity/candidate_scores.tsv

# Model 2, stage 3: Round-1 feedback update + calibration + freeze manifest
python scripts/feedback_update_and_freeze.py \
  --round1-table round1.tsv --output-dir out/round1 \
  --freeze-artifacts model_checkpoint=... configuration=... round1_labels=... \
      candidate_features=... sbase_scores=... supd_scores=... \
      calibrated_scores=... selection_rules=... round2_candidate_list=... \
      runtime_environment=... \
  --code-commit <sha> --study-version v1

# Round-2 selection from calibrated scores (lower predicted log10(MIC) first)
python scripts/select_screening_cohort.py \
  --ranked-candidates calibrated_scores.tsv --output round2_selection.tsv
```

Input schemas: `round1.tsv` needs `candidate_library_id`, `sbase`
(predicted log10(MIC)), `observed_log10_mic` and the nine feature columns
fixed by `src/bgc2activity/scoring_contract.py`; corpus and entity tables
follow the column contracts documented in each script's docstring.

## Committed core data

56,709-genome inventory; 258,750-row NRPS-positive BGC index; 77,587-row
substrate-predicted BGC index; 23,540-row candidate library with variant
classes; construction funnel, admission rules and constrained-variant rules
(Tables S1a–c); data resources and partition composition (Tables S2a–b);
strict-MIC dataset summary (Table S2c, descriptive); 400-row two-round MIC
table with screening endpoint summary and candidate-to-library mapping;
NRP45 source traceability tables; animal grouping (Table S5). See
[`data/README.md`](data/README.md) for the data dictionary and declared
limitations (truncated BGC identifiers, 21 terminal-form ambiguities, missing
model/wet-lab artifacts, documentation conflicts).

## Testing

The suite asserts manuscript-fixed counts, the Round-1/Round-2 separation,
candidate-library provenance, the exclusion of BGC metadata from activity
scores, the Method-12 freeze boundary, the regression formulation (Sbase on
the log10(MIC) scale, Sbase coefficient 1 in the residual update, monotone
calibration, fold alignment) and the Method-17 freeze-package integrity. It
does **not** reproduce final model performance until the locked input data
and approved checkpoints are recovered.

The repository includes runnable model architectures, training protocols and
inference CLIs, but it does not claim to reproduce the manuscript's reported
performance without the missing approved checkpoints and source-level public
training tables listed in the release contract. The committed core tables are
auditable release data; they are not silently substituted for those missing
model artifacts.

## Citation

Peng et al., *Hierarchical NRPS-guided global-to-local discovery and
optimisation of synthetic anti-MRSA lipopeptides*. Source data, processed
data, analysis tables, model configurations and analysis code are available
at https://github.com/SiyiPeng66/BGC2Activity.
