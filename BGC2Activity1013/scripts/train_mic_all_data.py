#!/usr/bin/env python3
"""All-data fitting diagnosis: no heldout split, no validation early stopping.

Label standardization, MSE and reduced regularization are explicit optimization
changes for this user-authorized fitting experiment, not manuscript equivalence.
"""
import argparse,json,sys,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bgc2activity_reconstructed.activity.model import MICRegressor
from bgc2activity_reconstructed.activity.chemical import candidate,fit_numeric,transform_numeric
from bgc2activity_reconstructed.pretraining.registry import install
from bgc2activity_reconstructed.pretraining.chemistry import parse_tokens,normalize
from bgc2activity_reconstructed.pretraining.data import encode,collate

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/processed/supplementary_records/activity.tsv');p.add_argument('--encoder',type=Path,default=ROOT/'models/peptide_encoder.pt');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=150);p.add_argument('--probe',action='store_true');p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--warm-start',type=Path);a=p.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise ValueError('Use a new output directory')
 torch.set_num_threads(2);torch.manual_seed(42);d=pd.read_csv(a.data,sep='\t');ck=torch.load(a.encoder,weights_only=False,map_location='cpu');install(ck.get('monomer_registry',{}))
 if a.probe:
  keys=['tokens','n_terminal','c_terminal','cyclization'];group=d.groupby(keys,dropna=False).log2_mic.agg(['min','max','size']);eligible=group[(group['max']-group['min']<1e-6)&group['size'].eq(1)].reset_index()
  chosen=d.merge(eligible[keys],on=keys).sort_values('log2_mic');idx=np.linspace(0,len(chosen)-1,32,dtype=int);d=chosen.iloc[idx].copy()
 d=d.reset_index(drop=True);assert d.sample_id.is_unique
 parsed=[parse_tokens(s) for s in d.tokens]
 assert all(normalize(t)[0] in ck['preprocessor']['vocabulary'] for row in parsed for t in row)
 chem=[candidate(r.tokens,r.n_terminal,r.c_terminal,r.cyclization) for r in d.itertuples()];raw=np.stack([x[0] for x in chem]);stats=fit_numeric(raw)
 numeric=torch.tensor(transform_numeric(raw,stats));states=torch.tensor(np.stack([x[1] for x in chem]));items=[encode(t,ck['preprocessor']) for t in parsed]
 y=d.log2_mic.to_numpy(dtype=np.float32);target_mean=float(y.mean());target_std=float(y.std());target_std=max(target_std,1e-6);targets=torch.tensor((y-target_mean)/target_std)
 model=MICRegressor(ck['config']);model.encoder.load_state_dict(ck['model_state'])
 warm=None
 if a.warm_start:
  warm=torch.load(a.warm_start,weights_only=False,map_location='cpu')
  assert warm['preprocessor']==ck['preprocessor'] and warm['numeric']==stats
  assert warm['target_transform']==dict(mean=target_mean,std=target_std)
  model.load_state_dict(warm['model_state'])
 # No dropout for fitting diagnosis; the saved inference architecture is identical.
 for m in model.modules():
  if isinstance(m,torch.nn.Dropout):m.p=0.
 for m in model.encoder.context_encoder.layers:m.self_attn.dropout=0.
 frozen={n:v.detach().clone() for n,v in model.named_parameters() if not v.requires_grad}
 params=[v for v in model.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr*.1)
 def forward(ix):return model(collate([items[i] for i in ix]),numeric[ix],states[ix])
 @torch.no_grad()
 def evaluate():
  model.eval();values=[]
  for st in range(0,len(d),a.batch_size):values.extend(forward(np.arange(st,min(st+a.batch_size,len(d)))).tolist())
  pred=np.asarray(values)*target_std+target_mean
  return pred,dict(mae=float(np.abs(pred-y).mean()),rmse=float(np.sqrt(np.mean((pred-y)**2))),r2=float(1-np.mean((pred-y)**2)/np.var(y)),pred_std=float(pred.std()),pred_min=float(pred.min()),pred_max=float(pred.max()),predicted_gt256=int((pred>8).sum()))
 a.output_dir.mkdir(parents=True);d.to_csv(a.output_dir/'training_records.tsv',sep='\t',index=False)
 history=[];best=float('inf');start=time.monotonic();gradient=None
 for epoch in range(1,a.epochs+1):
  model.train();order=torch.randperm(len(d)).numpy();loss_sum=0
  for st in range(0,len(d),a.batch_size):
   ix=order[st:st+a.batch_size];opt.zero_grad();pred=forward(ix);loss=torch.nn.functional.mse_loss(pred,targets[ix]);loss.backward()
   if gradient is None:gradient={n:float(v.grad.norm()) for n,v in model.named_parameters() if v.grad is not None}
   torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();loss_sum+=float(loss.detach())*len(ix)
  scheduler.step()
  if epoch==1 or epoch%5==0 or epoch==a.epochs:
   pred,metrics=evaluate();row=dict(epoch=epoch,train_mse_standardized=loss_sum/len(d),**metrics,elapsed_seconds=round(time.monotonic()-start,2));history.append(row)
   if metrics['rmse']<best:
    best=metrics['rmse'];best_epoch=epoch
    checkpoint=dict(model_state=model.state_dict(),encoder_config=ck['config'],preprocessor=ck['preprocessor'],monomer_registry=ck.get('monomer_registry',{}),numeric=stats,target_transform=dict(mean=target_mean,std=target_std),best_epoch=epoch,development_only=True,input_contract='explicit_chemical_v1',label_kind='experimental_user_confirmed',scope='User-confirmed experimental log2MIC; all-data fitting only; no independent evaluation',trained_states={field:sorted(d[field].unique().tolist()) for field in ['n_terminal','c_terminal','cyclization']},trained_configurations=sorted({normalize(t)[1] for row in parsed for t in row}),training_identity_vocabulary=sorted({normalize(t)[0] for row in parsed for t in row}),supervised_length_range=[min(map(len,parsed)),max(map(len,parsed))])
    torch.save(checkpoint,a.output_dir/'mic_checkpoint.pt')
   (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
 model.load_state_dict(torch.load(a.output_dir/'mic_checkpoint.pt',weights_only=False)['model_state']);pred,metrics=evaluate()
 for n,v in model.named_parameters():
  if n in frozen:assert torch.equal(v,frozen[n]),n
 out=d[['sample_id','tokens','log2_mic']].copy();out['predicted_log2_mic']=pred;out['predicted_mic_ug_ml']=np.exp2(pred);out.to_csv(a.output_dir/'training_predictions.tsv',sep='\t',index=False,float_format='%.6f')
 report=dict(status='completed',evaluation='training_resubstitution_only',n=len(d),epochs=a.epochs,best_epoch=best_epoch,metrics=metrics,constant_mean_rmse=float(y.std()),constant_median_mae=float(np.abs(y-np.median(y)).mean()),frozen_parameters_unchanged=True,first_batch_gradient_norms=gradient,optimizer=dict(lr=a.lr,weight_decay=0.,loss='standardized_target_MSE',dropout=0.,scheduler='cosine'),checkpoint_sha256=hashlib.sha256((a.output_dir/'mic_checkpoint.pt').read_bytes()).hexdigest(),data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),encoder_sha256=hashlib.sha256(a.encoder.read_bytes()).hexdigest())
 report['warm_start']=None if warm is None else dict(path=str(a.warm_start),sha256=hashlib.sha256(a.warm_start.read_bytes()).hexdigest(),source_best_epoch=warm['best_epoch'],optimizer_reinitialized=True)
 (a.output_dir/'training_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
