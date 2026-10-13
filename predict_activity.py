#!/usr/bin/env python3
"""Predict peptide MIC with the selected BGC2Activity activity model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models/activity.pt'


def main():
    parser = argparse.ArgumentParser(
        description='Predict peptide MIC (ug/mL) using the bundled model.',
        epilog='Use FASTA for standard peptides, TSV for nonstandard monomers and modifications. See README.md.')
    parser.add_argument('-i', '--input', type=Path, required=True, help='Input FASTA or TSV file')
    parser.add_argument('-o', '--output', type=Path, help='Output TSV; default: INPUT.predictions.tsv beside input')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Optional compatible model checkpoint')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + '.predictions.tsv')
    metadata = output.with_suffix('.metadata.json')
    if not args.input.is_file():
        parser.error(f'Input file not found: {args.input}')
    if not args.model.is_file():
        parser.error(f'Model file not found: {args.model}')
    protected = {args.input.resolve(), args.model.resolve()}
    if output.resolve() in protected or metadata.resolve() in protected:
        parser.error('Output must not overwrite input or model')
    if not args.overwrite and (output.exists() or metadata.exists()):
        parser.error(f'Output already exists: {output}; choose another path or use --overwrite')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from predict_chemical_mic import main as predict
    try:
        predict(['--checkpoint', str(args.model), '--peptides', str(args.input),
                 '--output', str(output)])
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Prediction completed: {output.resolve()}')


if __name__ == '__main__':
    main()
