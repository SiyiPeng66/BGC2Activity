# Demo provenance

## BGC DNA

Five records were extracted from the local MIBiG v4.0 GenBank archive: BGC0000019 (PKS), BGC0000319 (NRPS), BGC0001306 (RiPP), BGC0000651 (Terpene), and BGC0003066 (Other). Class labels follow the existing curated override table. The previous 300 bp RiPP fragment was replaced by BGC0001306 (6,409 bp) before inspecting predictions.

The background is NC_003888.3 (Streptomyces coelicolor A3(2)), positions 20,001–50,000 inclusive. It lies outside local antiSMASH 8.0.4 regions with at least a 10 kb buffer. This is a putative background control, not an experimentally confirmed non-BGC. It was selected without checking this model's prediction.

`examples/bgc/reference_labels.tsv` retains source-relative paths, coordinates and sequence hashes. Those paths document provenance and are not runtime dependencies. Output coordinates refer to the extracted fragments. `comparison.tsv` retains missed detections, type errors and the background candidate rather than selecting only successful predictions.

## Activity sequences

The 80 rows come from Supplementary Data(1).xlsx, Supplementary Data2, source rows 3–82. The owner confirmed that the column named “Calibrated model predicted log2MIC” contains experimental log2(MIC) values. The small reference table preserves original entry identifiers; the workbook and full training dataset are not included.

These peptides were included in supervised training and must not be described as an independent test set. Nonstandard monomer tokens and modification states are preserved rather than converted to lossy standard FASTA.
