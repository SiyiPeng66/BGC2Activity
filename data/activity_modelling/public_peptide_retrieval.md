# Public peptide resource retrieval record

Peptide representation-learning and strict MIC supervision data were obtained from
six public databases, frozen as a project snapshot on **2026-07-21** (Supplementary
Method 9). Raw third-party dumps are not redistributed here pending licence review;
use the retrieval routes below.

| Database | Snapshot/version used | Records (source-level) | Retrieval route |
| --- | --- | --- | --- |
| DBAASP | dynamic database snapshot, accessed 2026-07-21 | 6,909 source peptide entities | https://dbaasp.org/ |
| DRAMP | DRAMP 3.0 project snapshot, accessed 2026-07-21 | 35,485 source peptide records | http://dramp.cpu-bioinfor.org/downloads/ |
| APD | APD2024 FASTA subsets, accessed 2026-07-21 | 4,359 source peptide records | https://aps.unmc.edu/downloads |
| CAMPR4 | Download All snapshot, accessed 2026-07-21 | 24,243 source peptide records | https://camp.bicnirrh.res.in/ |
| dbAMP | v2.0, accessed 2026-07-21 | 28,709 source peptide records | https://ycclab.cuhk.edu.cn/dbAMP/download.php |
| SATPdb | v1 Zenodo snapshot, accessed 2026-07-21 | 19,192 source peptide records | https://zenodo.org/records/20078292 |

Manuscript-fixed curation counts (Supplementary Methods 9, 11):

- 118,897 unified source records; 118,630 parsed canonical sequences (267 excluded).
- 21,814 records with at least one nonstandard monomer or modification signal.
- 146,547 harmonised MIC measurements (DBAASP, CAMPR4, DRAMP).
- Strict labels per unit bucket: MIC <= 8 positive, MIC > 64 negative:
  61,545 labelled records -> 7,358 unique sequences (5,067 positive, 2,291 negative)
  -> nonstandard/modification subset 4,739 (3,511 positive, 1,228 negative).

Known documentation conflicts to resolve before data redistribution:

1. The main-text Data availability statement gives a freeze date of 2025-12-21,
   whereas Supplementary Method 9 and this record state 2026-07-21.
2. The main text names "DRAMP 4.0", "APD3" and "dbAMP 3.0", whereas the frozen
   Supplementary Table S2a records DRAMP 3.0, APD2024 and dbAMP v2.0.
