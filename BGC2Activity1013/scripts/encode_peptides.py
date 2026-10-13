#!/usr/bin/env python3
"""Export 96-dimensional peptide representations from a saved encoder."""
import argparse,sys,json,hashlib
from pathlib import Path
import pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bgc2activity_reconstructed.pretraining.chemistry import parse_tokens,normalize
from bgc2activity_reconstructed.pretraining.data import encode,collate
from bgc2activity_reconstructed.pretraining.model import ChemicalAwarePeptideEncoder
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--peptides',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-oov',action='store_true',help='Permit understood monomer identities absent from checkpoint training vocabulary; exported metadata lists them')
    a=p.parse_args();torch.set_num_threads(2)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    from bgc2activity_reconstructed.pretraining.registry import install
    install(ckpt.get('monomer_registry',{}))
    model=ChemicalAwarePeptideEncoder(**ckpt['config']);model.load_state_dict(ckpt['model_state']);model.eval()
    d=pd.read_csv(a.peptides,sep='\t',dtype=str,keep_default_na=False)
    if not {'sample_id','tokens'}.issubset(d):raise ValueError('Input needs sample_id and tokens')
    if d.empty or d.sample_id.duplicated().any():raise ValueError('Input must be nonempty with unique sample IDs')
    parsed=[parse_tokens(t) for t in d.tokens]
    oov=sorted({normalize(t)[0] for tokens in parsed for t in tokens if normalize(t)[0] not in ckpt['preprocessor']['vocabulary']})
    if oov and not a.allow_oov:
        raise ValueError('Monomers absent from this checkpoint training vocabulary: '+', '.join(oov)+'. Use a chemically covered checkpoint, or explicitly --allow-oov for diagnostic export.')
    rows=[]
    with torch.no_grad():
        for start in range(0,len(d),128):
            batch=collate([encode(t,ckpt['preprocessor']) for t in parsed[start:start+128]])
            _,embedding=model(batch);rows.extend(embedding.tolist())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.concat([d[['sample_id']].reset_index(drop=True),pd.DataFrame(rows,columns=[f'embedding_{i:03d}' for i in range(96)])],axis=1).to_csv(a.output,sep='\t',index=False)
    a.output.with_suffix('.metadata.json').write_text(json.dumps(dict(n=len(d),dimension=96,
        development_only=ckpt['development_only'],pooling_status=ckpt['pooling_status'],
        oov_identities=oov,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(a.peptides.read_bytes()).hexdigest()),indent=2))
    print(f'Exported {len(d)} x 96 representations to {a.output}; development_only={ckpt["development_only"]}')
if __name__=='__main__':main()
