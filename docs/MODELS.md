# Model notes

Default resources are relative to this project, not a user's home directory:

- `models/bgc_detector.pt`: selected BGC detector, epoch 41 of a 51-epoch run.
- `models/peptide_encoder.pt`: chemical-aware representation encoder.
- `models/activity.pt`: activity regressor including its fine-tuned encoder and preprocessing. Initially trained for 125 epochs, then warm-started for 25 epochs with a reinitialized optimizer.
- `models/pfam2vec.csv.gz`: frozen 100-dimensional Pfam embeddings.

Only load trusted checkpoints: activity and encoder files include Python-serialized metadata.

## Activity coverage

Supported nonstandard identities in the current checkpoint:

```text
2,3-dab beta-ala beta-asp beta-lys dab dap dhba dpg horn hpg hse hty
hxorn lys(lau) lys(myr) lys(ε-3r-oh-myr) lys(ε-c13:0)
lys(ε-iso-c13:0) lys(ε-myro) orn pip sal
```

Some are name-only entries with missing structural descriptors. Supporting a name does not establish verified structure or reliable activity prediction. M is absent from the current vocabulary. Maximum accepted length is 64; the supervised range is 5–41 monomers. FASTA modification states are unknown, not assumed linear/free.

The current regressor uses all 19,705 retained observations for training and evaluation, following the owner's instruction. Of 23,620 original rows, 3,915 cross-source overlaps were superseded by the preferred Data2 source. Same-source repeated observations were retained. The owner confirmed the labels are experimental log2(MIC), despite misleading original column/file names.

Training resubstitution MAE is 0.982 and RMSE 1.333 log2MIC; R² is 0.301. Predictions above 256 µg/mL number 6,203 versus 9,732 observed labels. Threshold agreement is 64.136%. Conflicting labels for identical observable inputs impose an estimated RMSE lower bound of 1.322. These are not independent test metrics. Strain and assay conditions are not separate model inputs.

## BGC interpretation

Pfam position threshold is 0.45. A gene's score is its maximum domain score. Candidate calling requires at least three positive genes and five predicted genes, permits up to two intervening negative genes, extends by two genes and merges nearby regions. Details are in `src/bgc2activity_reconstructed/detection/regions.py`.

Product classes are auxiliary predictions. Maximum position probability is not a calibrated whole-region probability. Local MIBiG/GeneSwap position-level metrics do not establish genome boundary accuracy or performance on natural negative backgrounds.

The advanced Pfam-table interface remains in `scripts/predict_bgc_regions.py`, but all user-facing BGC demos start from DNA. Its terminal-window placement differs from the genome pipeline, so their numeric scores need not match exactly.
