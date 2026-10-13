#!/usr/bin/env python3
"""Predict MIC with explicit nonstandard monomers and terminal/cycle states."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bgc2activity_reconstructed.activity.input import read_peptides
from bgc2activity_reconstructed.activity.model import MICRegressor
from bgc2activity_reconstructed.activity.chemical import candidate,transform_numeric
from bgc2activity_reconstructed.pretraining.chemistry import parse_tokens,normalize
from bgc2activity_reconstructed.pretraining.data import encode,collate

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--peptides','--fasta',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--input-format',choices=['auto','fasta','tsv'],default='auto');a=p.parse_args(argv)
 ck=torch.load(a.checkpoint,weights_only=False,map_location='cpu')
 if ck.get('input_contract')!='explicit_chemical_v1':raise ValueError('Requires a newly trained chemical checkpoint; old standard-only checkpoint is incompatible')
 from bgc2activity_reconstructed.pretraining.registry import install
 install(ck.get('monomer_registry',{}))
 d=read_peptides(a.peptides,a.input_format)
 allowed={'sample_id','tokens','n_terminal','c_terminal','cyclization'}
 if set(d.columns)-allowed:raise ValueError('Unrecognized chemical fields: '+str(set(d.columns)-allowed))
 if not {'sample_id','tokens'}<=set(d) or d.empty or d.sample_id.duplicated().any() or d.sample_id.str.strip().eq('').any():raise ValueError('Unique nonempty sample_id and tokens required')
 for name in ['n_terminal','c_terminal','cyclization']:
  if name not in d:d[name]='unknown'
  d[name]=d[name].replace('','unknown')
 parsed=[parse_tokens(s) for s in d.tokens]
 for row in parsed:
  unseen={normalize(t)[0] for t in row}-set(ck['preprocessor']['vocabulary'])
  if unseen:raise ValueError('Not covered by this pretrained checkpoint: '+str(sorted(unseen)))
 chem=[candidate(r.tokens,r.n_terminal,r.c_terminal,r.cyclization) for r in d.itertuples()]
 raw=np.stack([x[0] for x in chem]);numeric=transform_numeric(raw,ck['numeric'])
 states=np.stack([x[1] for x in chem]);torch.set_num_threads(2)
 model=MICRegressor(ck['encoder_config']);model.load_state_dict(ck['model_state']);model.eval()
 values=[]
 with torch.no_grad():
  for start in range(0,len(d),128):
   batch=collate([encode(t,ck['preprocessor']) for t in parsed[start:start+128]])
   values.extend(model(batch,torch.tensor(numeric[start:start+128]),torch.tensor(states[start:start+128])).tolist())
 if 'target_transform' in ck:
  values=(np.asarray(values)*ck['target_transform']['std']+ck['target_transform']['mean']).tolist()
 if not np.isfinite(values).all() or not np.isfinite(np.exp2(values)).all():raise ValueError('Nonfinite prediction')
 out=d.copy();out['monomer_count']=[len(t) for t in parsed];out['predicted_log2_mic']=values;out['predicted_mic_ug_ml']=np.exp2(values)
 low,high=ck['supervised_length_range'];out['outside_supervised_length_range']=[not low<=len(t)<=high for t in parsed]
 out['identities_without_MIC_training']=[';'.join(sorted({normalize(t)[0] for t in row}-set(ck['training_identity_vocabulary']))) for row in parsed]
 out['states_without_MIC_training']=[';'.join(field+'='+str(row[field]) for field in ['n_terminal','c_terminal','cyclization'] if row[field] not in ck['trained_states'][field]) for _,row in d.iterrows()]
 out['configuration_without_MIC_training']=[bool({normalize(t)[1] for t in row}-set(ck['trained_configurations'])) for row in parsed]
 out['label_only_monomers']=[';'.join(sorted({normalize(t)[0] for t in row if ck.get('monomer_registry',{}).get(normalize(t)[0],{}).get('status')=='label_only'})) for row in parsed]
 out['model_status']='teacher_distillation' if ck.get('label_kind')=='teacher' else 'development';out['label_kind']=ck.get('label_kind','experimental');a.output.parent.mkdir(parents=True,exist_ok=True)
 out.to_csv(a.output,sep='\t',index=False,float_format='%.3f')
 metadata=dict(input_sha256=hashlib.sha256(a.peptides.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),scope=ck['scope'],supported_length=[4,64],supervised_length_range=[low,high],pretrained_identity_vocabulary=ck['preprocessor']['vocabulary'],development_only=True,warning='Chemical support and running inference do not establish reliable MIC accuracy; no strain-specific endpoint.')
 a.output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(f'Wrote {len(out)} chemical MIC predictions to {a.output}')
if __name__=='__main__':main()
