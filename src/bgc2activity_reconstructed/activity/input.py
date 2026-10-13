"""Read standard peptide FASTA or legacy token TSV without guessing chemistry."""
from pathlib import Path
import re
import pandas as pd


def read_peptides(path, input_format='auto'):
    path = Path(path)
    if input_format == 'auto':
        with path.open(encoding='utf-8-sig') as handle:
            first = next((line.strip() for line in handle if line.strip()), '')
        input_format = 'fasta' if first.startswith('>') else 'tsv'
    if input_format == 'tsv':
        return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    records = []
    name, parts = None, []

    def finish():
        if name is None:
            return
        sequence = ''.join(parts)
        if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]{4,64}', sequence):
            raise ValueError(f'{name}: FASTA requires 4–64 uppercase standard amino acids; '
                             'X, lowercase stereochemistry and chemical modifications are unsupported '
                             'by this development checkpoint')
        records.append((name, sequence))

    with path.open(encoding='utf-8-sig') as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                finish()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f'Empty FASTA header at line {number}')
                name, parts = header.split()[0], []
            else:
                if name is None:
                    raise ValueError(f'Sequence before FASTA header at line {number}')
                parts.append(line)
    finish()
    data = pd.DataFrame(records, columns=['sample_id', 'tokens'])
    if data.empty or data.sample_id.duplicated().any():
        raise ValueError('FASTA must have at least one sequence and unique record IDs')
    return data
