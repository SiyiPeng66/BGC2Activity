# BGC2Activity

Predict candidate biosynthetic gene clusters (BGCs) from bacterial or archaeal DNA, and predict peptide minimum inhibitory concentration (MIC, µg/mL) from peptide sequences.

**Pretrained models are included. You do not need to train a model or specify a checkpoint for normal use.** The two workflows are separate: BGC DNA is not a peptide sequence, and this release does not automatically reconstruct a peptide product from a detected BGC.

## 1. Install once

Use Linux or macOS; Windows users can use WSL2. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install), download or clone this repository, and open a terminal in its root directory.

```bash
conda env create -f environment.yml
conda activate bgc2activity
python -m pip install -e . --no-deps
```

Alternatively, create the environment manually:

```bash
conda create -n bgc2activity python=3.11 pip -y
conda activate bgc2activity
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The installation requires internet access. Keep the repository's folder structure intact. In a new terminal, return to the project directory and run `conda activate bgc2activity` before using the commands below. Current prediction commands run on CPU; no GPU is required.

The Conda specification uses Python 3.11 with dependency ranges, not an exact cross-platform lockfile. The existing development environment uses Python 3.13.9 and PyTorch 2.13.0. A clean Conda installation has not yet been verified; see [verification notes](docs/VERIFICATION.md).

## 2. Try peptide activity prediction

Run the included 80-peptide example:

```bash
python predict_activity.py -i examples/activity/input.tsv -o results/activity_demo.tsv
```

Open `results/activity_demo.tsv` in Excel, selecting **Tab** as the separator if necessary.

| Column | Meaning |
|---|---|
| `sample_id` | Input peptide identifier |
| `predicted_mic_ug_ml` | Predicted MIC in µg/mL; lower values indicate a lower predicted inhibitory concentration |
| `predicted_log2_mic` | log₂(MIC); a value of 8 corresponds to 256 µg/mL |
| `label_only_monomers` | Recognized monomer names without fully verified structural properties |

Numbers are displayed to three decimal places. Additional columns report whether length, monomers, configurations or modifications fall outside supervised training coverage. A neighboring `.metadata.json` records model and input hashes.

### Use your own peptides

For unmodified standard sequences, create `peptides.fasta`:

```fasta
>peptide_1
SLSTAALSSTLSASAAL
>peptide_2
TTKSSRSSSEY
```

```bash
python predict_activity.py -i peptides.fasta -o results/my_activity.tsv
```

For nonstandard monomers or modifications, copy `examples/activity/input.tsv` and replace the rows. Keep the header, use unique identifiers, and save as **tab-separated text**, not an Excel workbook renamed to `.tsv`.

```tsv
sample_id	tokens	n_terminal	c_terminal	cyclization
peptide_1	T;T;K;hxorn;S;R;S;hxorn;E;Y;dab	free	free	unknown
```

| Field | Allowed values |
|---|---|
| `sample_id` | Required, unique, nonempty identifier |
| `tokens` | Required; separate monomers with semicolons; standard sequences may also use contiguous single letters |
| `n_terminal` | `unknown`, `free`, `acetyl` |
| `c_terminal` | `unknown`, `free`, `amide` |
| `cyclization` | `unknown`, `linear`, `head_tail` |

Missing modification fields mean `unknown`. For `head_tail`, both terminal fields must be `unknown`. Do not include experimental labels or other extra columns. FASTA cannot express these modifications; use TSV when they are known. Explicit stereochemistry uses `d-` or `l-` monomer prefixes.

**Current limits:** 4–64 monomers, with supervised lengths of 5–41. Only identities covered by the checkpoint are accepted. Methionine (`M`) is currently absent from its vocabulary, so even a valid standard FASTA containing M is rejected. Do not delete residues or substitute X to bypass this restriction. See [model notes](docs/MODELS.md) for supported nonstandard identities.

| Option | Purpose |
|---|---|
| `-i`, `--input` | Required FASTA or TSV input |
| `-o`, `--output` | Output TSV; defaults to `INPUT.predictions.tsv` beside the input |
| `--overwrite` | Replace existing output files |
| `--model` | Optional compatible checkpoint; normally leave this unset |

The 80 demo peptides were used for training. Their outputs demonstrate operation, not independent test performance. Reference identifiers and user-confirmed experimental log₂MIC values are in `examples/activity/reference_labels.tsv`.

## 3. Try BGC prediction from DNA

### Download Pfam once

BGC prediction needs the external **Pfam-A.hmm** database (approximately 2.1 GB in the development installation). It is not included in this repository.

```bash
mkdir -p data/pfam
curl -L --fail --output data/pfam/Pfam-A.hmm.gz https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
gunzip data/pfam/Pfam-A.hmm.gz
```

If you already have the decompressed file, place it at `data/pfam/Pfam-A.hmm`, or provide its path using `--pfam-db`. The `current_release` download can change: different Pfam versions may change results. Each run records the database hash. The workflow uses PyHMMER and Pyrodigal directly; separate `hmmpress`, `hmmscan`, or Prodigal executables are not needed.

### Run the sequence demo

```bash
python scripts/predict_bgc_genome.py \
  --genome examples/bgc/input.fna \
  --output results/bgc_demo
```

The program predicts genes, translates proteins, searches Pfam, and calls candidate BGC regions. Wait for the completion message; database searching takes longer than peptide prediction.

Open **`results/bgc_demo/bgc_regions.tsv`**:

| Column | Meaning |
|---|---|
| `region_id` | Candidate identifier assigned during this run |
| `contig` | Input FASTA sequence identifier |
| `start`, `end` | DNA coordinates, starting at 1 and including both endpoints |
| `n_genes` | Number of predicted genes in the region |
| `product_class` | Predicted NRPS, PKS, RiPP, Terpene or Other class |
| `nrps_positive` | Whether the program's NRPS criteria are met; False does not mean “not a BGC” |

Only detected candidates appear in this table. An input can produce zero, one or multiple regions. `Other` is a product class, not a negative label. No detected region does not establish experimental absence of a BGC.

Other outputs include `bgc_regions.fna` (candidate DNA), `bgc_regions.gff3` (coordinates for genome browsers), `genes.tsv`, `proteins.faa`, `pfam_hits.tsv`, excluded overlapping hits, and `run_manifest.json` with resource hashes and parameters.

### Use your own genome

Replace the demo path with your bacterial or archaeal DNA FASTA:

```bash
python scripts/predict_bgc_genome.py --genome my_genome.fna --output results/my_genome
```

| Option | Default / purpose |
|---|---|
| `--genome` | Required DNA FASTA; sequence IDs must be unique |
| `--output` | Required new or empty output directory |
| `--pfam-db` | Project-relative `data/pfam/Pfam-A.hmm` |
| `--cpus` | 2 CPU threads |
| `--mode` | `auto`; `single` needs at least one contig ≥20 kb; `meta` uses metagenomic gene calling throughout |
| `--checkpoint`, `--pfam2vec` | Bundled model resources; normally leave unset |

Multiple contigs and IUPAC DNA ambiguity codes are accepted. The current pipeline treats contigs as linear, does not join regions across circular origins, and does not perform eukaryotic gene annotation. FASTA title text before the first space becomes the identifier.

## 4. What is included?

```text
models/                Three trained checkpoints and Pfam2Vec
src/                   Model and preprocessing implementations
scripts/               Prediction, representation export and training code
examples/bgc/          Small raw-DNA demo, reference labels and actual outputs
examples/activity/     80 peptide sequences, labels and actual outputs
results/               Your generated results (ignored by Git)
```

BGC DNA input is approximately **148 KB**, well below 10 MB. It includes five reference BGC types and a genomic background fragment. The background lies outside local antiSMASH annotations but is **not an experimentally confirmed negative**. Expected and actual labels differ: the recorded run misses the RiPP example, misclassifies the PKS example and calls a region in the background. See `examples/bgc/comparison.tsv` and [data provenance](docs/DATA_SOURCES.md).

The repository is about 22 MB before Git history, dominated by Pfam2Vec and the checkpoints. Each included file is below 50 MB; Git LFS is not required for this snapshot. Pfam HMMs, full training datasets, environments, historical outputs and archives are excluded. Do not add them with `git add -f`. Checksums and sizes are in `models/manifest.json` and `repository_manifest.json`.

## 5. Troubleshooting

| Problem | Action |
|---|---|
| Missing Python package | Activate the environment and install with `python -m pip install -r requirements.txt` |
| Pfam database not found | Download and decompress it, or pass `--pfam-db /path/to/Pfam-A.hmm` |
| Output already exists | Choose another output; activity supports `--overwrite`, BGC requires a new/empty directory |
| Unresolved / uncovered monomer | Check spelling and checkpoint vocabulary; arbitrary new identities require additional model work |
| All text appears in one spreadsheet column | Import as tab-separated text |
| No BGC rows | Check `run_manifest.json`; a completed run with no candidates is valid |
| Different results with a new database | Compare database hashes and versions; annotation changes can affect predictions |

For command options, append `--help`. To run local unit tests:

```bash
python -m unittest discover -s tests -v
```

See [model limitations](docs/MODELS.md), [training notes](docs/TRAINING.md), and [third-party provenance](THIRD_PARTY.md). This is a reconstruction and inference release candidate, not a claim to recover every detail of the lost original implementation. No independent activity test set has yet been evaluated.
