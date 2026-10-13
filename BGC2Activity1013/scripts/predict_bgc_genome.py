#!/usr/bin/env python3
"""Detect candidate BGC coordinates from a prokaryotic genome FASTA."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--genome',type=Path,required=True,help='Prokaryotic nucleotide FASTA; unique contig identifiers')
    p.add_argument('--output',type=Path,required=True,help='New or empty output directory')
    p.add_argument('--pfam-db',type=Path,default=ROOT/'data/pfam/Pfam-A.hmm')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/bgc_detector.pt')
    p.add_argument('--pfam2vec',type=Path,default=ROOT/'models/pfam2vec.csv.gz')
    p.add_argument('--mode',choices=['auto','single','meta'],default='auto')
    p.add_argument('--cpus',type=int,default=2)
    a=p.parse_args()
    if a.cpus<1:p.error('--cpus must be positive')
    if not a.pfam_db.is_file():p.error('Pfam database not found: '+str(a.pfam_db)+'. See README: download and decompress Pfam-A.hmm, or use --pfam-db /path/to/Pfam-A.hmm.')
    from bgc2activity_reconstructed.detection.genome import run
    run(a)
if __name__=='__main__':main()
