# Verification (2026-10-13)

Verified from the new BGC2Activity1013 directory:

- Activity CLI processed all 80 demo peptides; the output matches the previously selected checkpoint's demo output.
- DNA CLI processed all six sequences using the existing external Pfam database: 133 genes, 160 retained domains, five candidate BGC regions. Region output matches the recorded demo output.
- All 19 included unit tests passed.
- Included model/Pfam2Vec files are copied without changing their bytes; hashes are in models/manifest.json.

These checks used the existing local Python 3.13.9 / PyTorch 2.13.0 environment. For the genome check, PyHMMER and Pyrodigal were supplied explicitly from the existing annotation dependency installation through PYTHONPATH; the new source does not hard-code that installation. The external Pfam database was passed with --pfam-db.

This does not establish a fresh Conda installation on another machine. No new packages were downloaded for this check. Install requirements.txt and run the demos on the target environment before claiming cross-platform installation verification.

Generated verification outputs and caches are excluded from the GitHub source snapshot. Local verification was completed before preparing the GitHub commit.
