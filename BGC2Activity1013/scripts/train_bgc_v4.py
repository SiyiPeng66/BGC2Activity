#!/usr/bin/env python3
"""Train with family batches and inner validation, then evaluate frozen test data."""
import argparse, json, random, sys, time, platform
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from train_bgc_detector import load_samples
import hashlib
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
from bgc2activity_reconstructed.detection.data import load_pfam2vec,collate_regions
from bgc2activity_reconstructed.detection.model import MultitaskBGCDetector,load_balancing_loss
from bgc2activity_reconstructed.detection.losses import multitask_loss
from bgc2activity_reconstructed.detection.training import AdaptiveWeights,family_batches

def atomic_json(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)
def save(path,obj):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

@torch.no_grad()
def evaluate(model,samples,vocab,matrix):
    model.eval(); labels=[]; scores=[]
    for start in range(0,len(samples),32):
        b=collate_regions(samples[start:start+32],vocab,matrix)
        o=model(b['embeddings'],b['valid']); valid=b['valid'].bool()
        labels.extend(b['position_labels'][valid].tolist()); scores.extend(o.detection_logits.sigmoid()[valid].tolist())
    return metrics(labels,scores)
def metrics(labels,scores):
    p,r,f,_=precision_recall_fscore_support(labels,np.asarray(scores)>=.45,average='binary',zero_division=0)
    return dict(auprc=float(average_precision_score(labels,scores)),auroc=float(roc_auc_score(labels,scores)),
                precision=float(p),recall=float(r),f1=float(f),n_positions=len(labels),threshold=.45)

@torch.no_grad()
def test_and_predict(model,samples,vocab,matrix,out):
    from predict_bgc_regions import predict_region
    import csv
    model.eval(); labels=[];scores=[]
    with (out/'test_predictions.tsv').open('w',newline='') as handle:
        writer=None
        for s in samples:
            row=predict_region(model,s,vocab,matrix)
            if writer is None:
                writer=csv.DictWriter(handle,fieldnames=list(row),delimiter='\t'); writer.writeheader()
            writer.writerow(row)
            labels.extend(s.position_labels)
            scores.extend(float(x) for x in row['position_probabilities'].split(';'))
    result=metrics(labels,scores); result['n_regions']=len(samples)
    result['mode']='full_length_windowed_256_stride128; checkpoint frozen before test'
    atomic_json(out/'test_metrics.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data/processed/bgc_detector_v4')
    ap.add_argument('--pfam2vec',type=Path,default=ROOT/'models/pfam2vec.csv.gz')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if (out/'last_training_state.pt').exists() and not args.resume: raise SystemExit('Run exists; use --resume or a new directory')
    torch.set_num_threads(args.threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    vocab,m=load_pfam2vec(args.pfam2vec); matrix=torch.tensor(m)
    train=load_samples(args.data_dir/'train_samples.tsv');val=load_samples(args.data_dir/'validation_samples.tsv')
    test=load_samples(args.data_dir/'test_samples.tsv')
    for field in ['sample_id','family_label','pfam_ids']:
        sets=[{getattr(s,field) for s in rows} for rows in [train,val,test]]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)): raise ValueError(field+' leakage')
    model=MultitaskBGCDetector(pfam2vec_dimension=m.shape[1]); adaptive=AdaptiveWeights()
    params=list(model.parameters())+list(adaptive.parameters())
    optimizer=torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-3)
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in list((ROOT/'src').rglob('*.py'))+[Path(__file__)]}
    manifest=dict(status='running',n_train=len(train),n_validation=len(val),n_test=len(test),
      dataset_sha256=sha256(args.data_dir/'dataset_manifest.json'),pfam2vec_sha256=sha256(args.pfam2vec),
      source_sha256=hashes,pfam2vec_dimension=int(m.shape[1]),
      command=' '.join(sys.argv),environment=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',threads=args.threads),
      protocol=dict(optimizer='AdamW',learning_rate=5e-4,weight_decay=1e-3,batch_size=args.batch_size,gradient_clip=1.,
       max_epochs=args.epochs,patience=args.patience,selection='inner_validation_position_average_precision',seed=args.seed,
       auxiliary_weights='1.5*softmax(learnable_logits), initialized [1,.5]; explicit reconstruction choice',
       sampler='complete-coverage family quartets; 2 support + 2 query where >=4 unique members; smaller families retain unique members',
       training_max_positions=256,inference_window=256,inference_stride=128),history=[])
    best=-1.;stale=0;start=0
    if args.resume:
        previous=json.loads((out/'training_manifest.json').read_text())
        for key in ['dataset_sha256','pfam2vec_sha256','source_sha256','protocol']:
            if previous[key]!=manifest[key]: raise ValueError('Resume configuration mismatch: '+key)
        state=torch.load(out/'last_training_state.pt',weights_only=False,map_location='cpu')
        model.load_state_dict(state['model']);adaptive.load_state_dict(state['adaptive']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);start=state['epoch'];best=state['best'];stale=state['stale']
        manifest['history']=previous['history']
    atomic_json(out/'training_manifest.json',manifest)
    for epoch in range(start,args.epochs):
        if stale>=args.patience: break
        t=time.monotonic();model.train(); totals=np.zeros(5);steps=0;seen=set();anchor_count=0
        for indices in family_batches(train,args.seed+epoch,args.batch_size):
            seen.update(indices);b=collate_regions([train[i] for i in indices],vocab,matrix)
            optimizer.zero_grad(set_to_none=True);o=model(b['embeddings'],b['valid']);weights=adaptive()
            losses=multitask_loss(o.detection_logits,b['position_labels'],b['valid'],o.class_logits,b['class_labels'],
                o.region_representation,b['family_labels'],load_balancing_loss(o.router_probabilities,o.expert_fraction,b['valid']),
                lambda_class=weights[0],lambda_evo=weights[1])
            if not torch.isfinite(losses.total): raise RuntimeError('Non-finite training loss')
            losses.total.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            totals+=np.array([x.detach().item() for x in [losses.total,losses.detection,losses.classification,losses.evolutionary,losses.load_balancing]])
            labs=b['family_labels'];same=labs[:,None]==labs[None,:]
            anchor_count+=int(((same.sum(1)>1)&(~same).any(1)).sum());steps+=1
        assert len(seen)==len(train),'Epoch omitted training samples'
        validation=evaluate(model,val,vocab,matrix);ap=validation['auprc'];improved=ap>best+1e-6
        if improved:
            best=ap;stale=0;save(out/'detector_checkpoint.pt',model.state_dict());manifest['best_epoch']=epoch+1
        else: stale+=1
        row=dict(epoch=epoch+1,seconds=round(time.monotonic()-t,2),validation=validation,
          loss=dict(zip(['total','detection','classification','evolutionary','balance'],(totals/steps).tolist())),
          weights=adaptive().detach().tolist(),unique_training_samples=len(seen),triplet_anchors=anchor_count,batches=steps)
        manifest['history'].append(row);manifest['best_validation_auprc']=best;manifest['epochs_completed']=epoch+1
        save(out/'last_training_state.pt',dict(epoch=epoch+1,model=model.state_dict(),adaptive=adaptive.state_dict(),
          optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),best=best,stale=stale))
        atomic_json(out/'training_manifest.json',manifest)
        print(f'epoch {epoch+1:03d} val_AP={ap:.6f} loss={totals[0]/steps:.4f} anchors={anchor_count} seconds={row["seconds"]}',flush=True)
    model.load_state_dict(torch.load(out/'detector_checkpoint.pt',weights_only=True,map_location='cpu'))
    print('Training finished; evaluating frozen checkpoint on full-length independent test regions',flush=True)
    manifest['test_metrics']=test_and_predict(model,test,vocab,matrix,out)
    manifest['status']='completed';manifest['stop_reason']='early_stopping' if stale>=args.patience else 'max_epochs'
    manifest['checkpoint_sha256']=sha256(out/'detector_checkpoint.pt')
    atomic_json(out/'training_manifest.json',manifest)
    print(json.dumps(dict(epochs=manifest['epochs_completed'],best_epoch=manifest['best_epoch'],test=manifest['test_metrics']),indent=2),flush=True)
if __name__=='__main__':main()
