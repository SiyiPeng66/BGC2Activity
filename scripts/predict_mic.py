#!/usr/bin/env python3
"""Predict exploratory continuous MIC for standard unmodified peptides."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bgc2activity_reconstructed.activity.model import MICRegressor,chemistry
from bgc2activity_reconstructed.activity.input import read_peptides
from bgc2activity_reconstructed.pretraining.chemistry import parse_tokens,normalize
from bgc2activity_reconstructed.pretraining.data import encode,collate
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--peptides', '--fasta', dest='peptides',type=Path,required=True,help='Peptide FASTA or legacy TSV; FASTA headers provide sample IDs');p.add_argument('--input-format',choices=['auto','fasta','tsv'],default='auto');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if ck.get('input_contract')=='explicit_chemical_v1':
  from predict_chemical_mic import main as chemical_main
  return chemical_main(['--checkpoint',str(a.checkpoint),'--peptides',str(a.peptides),'--output',str(a.output),'--input-format',a.input_format])
 model=MICRegressor(ck['encoder_config']);model.load_state_dict(ck['model_state']);model.eval()
 d=read_peptides(a.peptides,a.input_format)
 if not {'sample_id','tokens'}.issubset(d) or d.empty or d.sample_id.duplicated().any():raise ValueError('Need unique sample_id and tokens')
 if d.sample_id.str.strip().eq('').any():raise ValueError('Empty sample_id')
 tokens=[parse_tokens(t) for t in d.tokens]
 if any(normalize(t)[0] not in ck['preprocessor']['vocabulary'] for row in tokens for t in row):raise ValueError('Monomer outside trained vocabulary')
 if any(any(t not in 'ACDEFGHIKLMNPQRSTVWY' or len(t)!=1 for t in row) for row in tokens):raise ValueError('Checkpoint supports standard unmodified monomers only')
 for column in set(d.columns)-{'sample_id','tokens'}:
  if d[column].astype(str).str.strip().ne('').any():raise ValueError('This checkpoint cannot consume additional modification fields: '+column)
 prep=ck['numeric'];values=[]
 with torch.no_grad():
  for start in range(0,len(tokens),128):
   rows=tokens[start:start+128];raw=np.stack([chemistry(''.join(t)) for t in rows]);missing=~np.isfinite(raw)
   filled=np.where(missing,prep['median'],raw);numeric=np.concatenate([(filled-prep['mean'])/np.asarray(prep['std']),missing],1)
   batch=collate([encode(t,ck['preprocessor']) for t in rows])
   result=model(batch,torch.tensor(numeric,dtype=torch.float32),torch.zeros(len(rows),4,dtype=torch.long))
   values.extend(result.tolist())
 if not np.isfinite(values).all() or not np.isfinite(np.exp2(values)).all():raise ValueError('Nonfinite prediction')
 out=d[['sample_id']].copy();out['predicted_log2_mic']=values;out['predicted_mic_ug_ml']=np.exp2(values)
 out['development_only']=True;a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,sep='\t',index=False,float_format='%.3f')
 a.output.with_suffix('.metadata.json').write_text(json.dumps(dict(scope=ck['scope'],development_only=True,
  input_sha256=hashlib.sha256(a.peptides.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
  warning='Runtime-only development model; not for activity ranking or experimental decisions.'),indent=2))
 print(f'Wrote {len(out)} exploratory MIC predictions to {a.output}')
if __name__=='__main__':main()
