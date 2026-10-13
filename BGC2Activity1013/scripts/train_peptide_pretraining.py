#!/usr/bin/env python3
"""Train a chemical-aware peptide encoder with auditable selection and export."""
import argparse,json,hashlib,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bgc2activity_reconstructed.pretraining.data import read_corpus,fit_preprocessor,encode,collate
from bgc2activity_reconstructed.pretraining.model import ChemicalAwarePeptideEncoder
from bgc2activity_reconstructed.pretraining.objective import corrupt,loss_values
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def batches(items,batch_size,order=None):
    if order is None:order=range(len(items))
    order=list(order)
    for start in range(0,len(order),batch_size):yield collate([items[i] for i in order[start:start+batch_size]])
@torch.no_grad()
def evaluate(model,items,bank,batch_size,protocol):
    generator=torch.Generator().manual_seed(20260803);model.eval();total=np.zeros(4);count=0
    for batch in batches(items,batch_size):
        changed,selected,mask=corrupt(batch,bank,generator)
        prediction,_=model(changed,mask);values=loss_values(prediction,batch,selected,protocol)
        n=int(selected.sum());total+=np.array([v.item() for v in values])*n;count+=n
    return dict(zip(['total','smooth_l1','cosine_similarity','mse'],(total/count).tolist()))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--protocol',choices=['method10','table_s11'],required=True)
    p.add_argument('--development',action='store_true',help='Explicitly permit incomplete corpus evidence; checkpoint marked development-only')
    p.add_argument('--registry',type=Path);p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    from bgc2activity_reconstructed.pretraining.registry import install,entries
    if a.registry:install(json.loads(a.registry.read_text()))
    evidence=json.loads(a.manifest.read_text())
    if evidence['corpus_sha256']!=sha(a.corpus):raise ValueError('Corpus checksum mismatch')
    if not a.development:
        if not evidence.get('formal_ready') or not evidence.get('mic_heldout_excluded') or evidence.get('split_rule')!='global_alignment_0.80_length_ratio_0.80':
            raise ValueError('Formal training requires reviewed chemistry, structural clusters and MIC heldout-neighbour exclusion evidence')
        for key in ['cluster_evidence_path','mic_exclusion_evidence_path']:
            if key not in evidence or not Path(evidence[key]).is_file():raise ValueError('Missing '+key)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise ValueError('Use a new output directory')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(20260802)
    frame=read_corpus(a.corpus);prep=fit_preprocessor(frame.loc[frame.split=='train','parsed'].tolist())
    items={split:[encode(t,prep) for t in frame.loc[frame.split==split,'parsed']] for split in ['train','validation','test']}
    bank_items=[encode([t],prep) for t in prep['random_tokens']]
    bank_batch=collate(bank_items);bank={k:v[:,0] for k,v in bank_batch.items() if k!='valid'}
    cfg=dict(dropout=.1 if a.protocol=='method10' else .15)
    lr=8e-4 if a.protocol=='method10' else 1e-4;wd=1e-3 if a.protocol=='method10' else 1e-4
    epochs=a.epochs or (30 if a.protocol=='method10' else 200)
    model=ChemicalAwarePeptideEncoder(**cfg)
    # Token-level objective has no path to pooling: explicitly freeze unused
    # pooling parameters during pretraining; downstream MIC can fine-tune them.
    for module in [model.pool_attention,model.pool_projection]:
        for parameter in module.parameters():parameter.requires_grad=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=wd)
    initial=evaluate(model,items['validation'],bank,a.batch_size,a.protocol)
    initial_state={k:v.clone() for k,v in model.state_dict().items()}
    gen=torch.Generator().manual_seed(20260802)
    manifest=dict(status='running',development_only=a.development,protocol=a.protocol,epochs=epochs,
      batch_size=a.batch_size,learning_rate=lr,weight_decay=wd,config=cfg,seed=20260802,
      split_counts={k:len(v) for k,v in items.items()},initial_validation=initial,history=[],
      pooling_status='initialized_untrained_by_position_loss; to_be_finetuned_in_MIC',
      vocabulary_size=len(prep['vocabulary']),corpus_sha256=sha(a.corpus),input_manifest_sha256=sha(a.manifest),
      code_sha256={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'src/bgc2activity_reconstructed/pretraining').glob('*.py'))+[Path(__file__)]})
    best=float('inf');atomic(a.output_dir/'training_manifest.json',manifest)
    for epoch in range(1,epochs+1):
        t=time.monotonic();model.train();loss_sum=0;positions=0
        order=torch.randperm(len(items['train']),generator=gen).tolist()
        for batch in batches(items['train'],a.batch_size,order):
            changed,selected,mask=corrupt(batch,bank,gen);opt.zero_grad(set_to_none=True)
            prediction,_=model(changed,mask);loss,*_=loss_values(prediction,batch,selected,a.protocol)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            n=int(selected.sum());positions+=n;loss_sum+=loss.item()*n
        val=evaluate(model,items['validation'],bank,a.batch_size,a.protocol)
        row=dict(epoch=epoch,train_loss=loss_sum/positions,validation=val,seconds=round(time.monotonic()-t,2))
        manifest['history'].append(row)
        checkpoint=dict(model_state=model.state_dict(),preprocessor=prep,config=cfg,epoch=epoch,
             protocol=a.protocol,monomer_registry=entries(),development_only=a.development,pooling_status=manifest['pooling_status'])
        if val['total']<best:
            best=val['total'];manifest['best_epoch']=epoch;torch.save(checkpoint,a.output_dir/'encoder.pt')
        torch.save({**checkpoint,'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'corruption_rng':gen.get_state()},a.output_dir/'last_training_state.pt')
        atomic(a.output_dir/'training_manifest.json',manifest);print(json.dumps(row),flush=True)
    ckpt=torch.load(a.output_dir/'encoder.pt',weights_only=False,map_location='cpu');model.load_state_dict(ckpt['model_state'])
    manifest['test']=evaluate(model,items['test'],bank,a.batch_size,a.protocol)
    manifest['encoder_weights_changed']=not torch.equal(initial_state['context_encoder.layers.0.linear1.weight'],model.state_dict()['context_encoder.layers.0.linear1.weight'])
    manifest['status']='completed';manifest['checkpoint_sha256']=sha(a.output_dir/'encoder.pt')
    atomic(a.output_dir/'training_manifest.json',manifest)
    print(json.dumps(dict(best_epoch=manifest['best_epoch'],test=manifest['test'])),flush=True)
if __name__=='__main__':main()
