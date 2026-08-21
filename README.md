# BGC2Activity

Analysis code and core data for *Hierarchical NRPS-guided global-to-local
discovery and optimisation of synthetic anti-MRSA lipopeptides*
(Peng et al.). The repository implements the complete computational workflow
of the study: genome-scale BGC detection, reconstruction of NRPS-informed
synthetic peptide candidates, chemical-aware peptide representation learning,
quantitative MIC-regression activity modelling with experimental feedback
updating, fixed-budget sequential screening, and local optimisation of the
NRP45 scaffold.

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
