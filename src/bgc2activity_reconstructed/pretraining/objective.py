"""Independent 30% selection; 80/10/10 corruption with uncontaminated targets."""
import torch
import torch.nn.functional as F
def corrupt(batch,bank,generator):
    valid=batch['valid'];shape=valid.shape
    selected=(torch.rand(shape,generator=generator)<.3)&valid
    for i in range(len(valid)):
        if not selected[i].any():
            ids=valid[i].nonzero().flatten()
            selected[i,ids[torch.randint(len(ids),(1,),generator=generator)]]=True
    draw=torch.rand(shape,generator=generator)
    masked=selected&(draw<.8);random=selected&(draw>=.8)&(draw<.9)
    changed={k:v.clone() for k,v in batch.items()}
    count=int(random.sum())
    ids=torch.randint(len(bank['text']),(count,),generator=generator)
    for k in ['text','descriptors','missing','configuration']:changed[k][random]=bank[k][ids]
    return changed,selected,masked
def loss_values(pred,batch,selected,protocol='method10'):
    target=torch.cat([batch['text'],batch['descriptors']],-1)
    # Imputed values are training targets, with missingness supplied to encoder.
    p=pred[selected];t=target[selected]
    smooth=F.smooth_l1_loss(p,t);mse=F.mse_loss(p,t);cos=F.cosine_similarity(p,t,dim=-1).mean()
    total=smooth+.2*(1-cos) if protocol=='method10' else mse
    return total,smooth,cos,mse
