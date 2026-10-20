# Synthetic peptide mass-spectrometry dataset (n = 100)

This dataset contains 100 paired peptide-sequence tables and mzML mass-spectrometry files, published at the data owner's request.

- NRP1–NRP80: 80 peptides.
- NRP45-A01–NRP45-A20: 20 NRP45 derivatives.
- Each sample has one identically named .tsv and .mzML file.
- Total original data size: approximately 102.6 MB; largest file: less than 1.7 MB.
- index.tsv is a combined sample index; manifest.json records sizes and SHA-256 checksums of all 200 original data files.

## Sequence tables

Each original TSV contains one record with peptide_name, spectrum_name and sequence. Names, stereochemistry, lipidation, terminal groups and cyclization annotations are preserved exactly as supplied. Displayed sequences are not automatically normalized or interpreted as complete validated chemical structures.

## Mass-spectrometry files

All 100 mzML XML documents were parsed successfully. They contain 5,144 spectra, all annotated as MS1; no MS2 spectra were found. Instrument metadata identify LCMS-2020 (74 files) and LCMS-2050 (26 files). Individual files contain 30, 34 or 60 spectra.

File pairing, sample identifiers and spectrum filenames were checked. These checks establish structural consistency, not peptide identity, purity, sequence confirmation or experimental quality. No peak matching or raw-vendor-file validation was performed for this upload.

The files retain original acquisition/conversion metadata, including instrument information and source-file names. Original data bytes were preserved; the macOS .DS_Store file was excluded.

## Use

Download individual pairs from this directory or clone the repository to obtain all files. No Git LFS client is needed. Mass spectra can be opened in mzML-compatible software. TSV files can be imported into spreadsheet software using a tab delimiter.

These files do not contain MIC labels and cannot be passed directly to the activity prediction CLI. Converting their displayed sequences requires preserving modifications and checking checkpoint vocabulary and topology support. Do not interpret “synthetic peptide” in the directory name as a claim that spectra were computationally simulated.

This dataset is separate from the small prediction demos under examples/.
