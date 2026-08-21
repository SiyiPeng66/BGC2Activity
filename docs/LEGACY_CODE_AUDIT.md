# Legacy-code audit and reconstruction boundary

The following decisions are based on direct source inspection. They are deliberately conservative: executable code is not evidence that it implements the final manuscript.

| Legacy component | What it can contribute | Why it cannot be released as the final implementation | Reconstruction decision |
| --- | --- | --- | --- |
| `EvoBGC/src/evobgc/data/pfam2vec.py` and input readers | 102-dimensional pfam2vec loading and basic Pfam sequence parsing | Input provenance, family partitions and final training data are not frozen in this code | Reimplement with full accession/Pfam manifests and contract tests. |
| `EvoBGC/src/evobgc/models/multitask.py` | Illustrates sequence-level detection/classification interfaces and length-aware pooling | It uses a BiLSTM encoder, while the final method specifies four Time-mixing/Channel-mixing blocks and an eight-expert MoE | Do not copy; implement the final architecture from the manuscript specification. |
| `EvoBGC/configs/multitask.yaml` | Naming conventions only | The checked configuration has hidden size 32, two epochs, CPU, and eight positive/eight negative samples, so it is a smoke configuration rather than the formal run | Keep out of the release; create a new, explicit formal configuration. |
| `EvoBGC/src/evobgc/data/datasets.py` | Basic grouped-table loading pattern | It zero-pads per-position labels without returning a loss mask; the formal loss must exclude padding | Reimplement with explicit attention/loss masks and padding tests. |
| `sequence_chemical_pretraining_real/src/model.py` | Mask-aware multimodal Transformer and attentive statistics pooling pattern | Its own README calls the descriptor catalogue an engineering approximation; final corpus, token dictionary, source freezes and trained checkpoint are not the manuscript-locked assets | Reuse no weights or outputs. A new model may borrow only independently tested, generic tensor operations. |
| `nrp_amp_ranker/parsing/` and simple IO utilities | Candidate-string parsing and plain-table IO may be useful after test cases are rebuilt | These utilities are tied to an intermediate candidate schema and must not silently change terminal/modification semantics | Port selectively only behind final-library conformance tests. |
| `nrp_amp_ranker/scripts/train_base_amp.py` | Shows a feature-table training entry point | It imports `GroupKFold` but executes `StratifiedKFold`; duplicate peptide IDs can cross folds, contrary to the final structural-cluster split | Replace with a cluster-level split implementation and leakage tests. |
| `nrp_amp_ranker/scripts/rank_candidates.py` and `models/fusion.py` | No scientific model component | They hard-code BGC evidence and product confidence and feed them into the final score, contradicting the manuscript's metadata/activity separation | Exclude completely. |
| `nrp_amp_ranker/scripts/calibrate_wetlab.py` | Basic calibration diagnostics pattern | It is built around a different 86-record/LOO workflow, while final Methods 15-17 require 40 Round-1 candidates, five-fold cross-fitting and a frozen Round-2 package | Reimplement residual update and constrained monotone calibration from the final method. |

## Code release rules

1. Every public module must cite its input/output schema in a docstring or adjacent data dictionary and must have a deterministic test using a small synthetic fixture.
2. No module may read a legacy result file by default. All paths must be explicit CLI/config inputs.
3. Model training, score generation, selection, calibration and plotting must be separate commands with content-hash manifests between stages.
4. Training code must export fold membership, random seed, parameters, score type (`Sbase`, `Supd`, `Pcal`) and a SHA-256 manifest.
5. Tests must assert the manuscript-critical counts, the Round-1/2 separation, candidate-library provenance and the exclusion of BGC metadata from activity probabilities.

## Recommended reconstruction order

1. Candidate-library schema and deterministic reconstruction/audit pipeline.
2. Study-data dictionaries, source manifests and MIC endpoint computation.
3. Selection/update/calibration workflow and freeze-package writer.
4. Local-optimisation and experimental-data analysis workflow.
5. BGC and peptide representation training/inference modules, after the formal input data and locked evaluation outputs are available.
