"""Standard-peptide MIC regression with manuscript encoder freeze boundary."""
import numpy as np, torch
from torch import nn
from ..pretraining.model import ChemicalAwarePeptideEncoder
from ..pretraining.chemistry import features
FIELDS=['length','mass','charge_pH7','gravy','aromatic_fraction','nonstandard_fraction','lipidation']
def chemistry(sequence):
    x=np.stack([features(t)[1] for t in sequence])
    # Unmodified free termini are an explicit development assumption.
    positive=1/(1+10**(7-9.69))
    negative=1/(1+10**(2.34-7))
    for aa,pka in [('K',10.5),('R',12.5),('H',6.)]:positive+=sequence.count(aa)/(1+10**(7-pka))
    for aa,pka in [('D',3.86),('E',4.25),('C',8.33),('Y',10.07)]:negative+=sequence.count(aa)/(1+10**(pka-7))
    return np.array([len(sequence),x[:,4].sum()-18.01528*(len(sequence)-1),positive-negative,
                     x[:,0].mean(),np.mean([t in 'FWHY' for t in sequence]),0.,0.],np.float32)
class MICRegressor(nn.Module):
    def __init__(self,encoder_config):
        super().__init__();self.encoder=ChemicalAwarePeptideEncoder(**encoder_config)
        self.encoder.freeze_for_mic()
        # Unknown/absent/present for N terminus, C terminus, lipidation, cyclization.
        self.states=nn.ModuleList([nn.Embedding(3,4) for _ in range(4)])
        self.chemistry_projection=nn.Sequential(nn.Linear(14+16,32),nn.GELU(),nn.LayerNorm(32))
        self.head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128),nn.Dropout(.30),
              nn.Linear(128,64),nn.GELU(),nn.LayerNorm(64),nn.Dropout(.20),nn.Linear(64,1))
    def train(self,mode=True):
        super().train(mode)
        # Frozen lower layers must not inject stochastic dropout while fine-tuning.
        for layer in self.encoder.context_encoder.layers[:2]:layer.eval()
        return self
    def forward(self,batch,numeric,states):
        _,e=self.encoder(batch)
        c=self.chemistry_projection(torch.cat([numeric]+[emb(states[:,i]) for i,emb in enumerate(self.states)],-1))
        return self.head(torch.cat([e,c],-1)).squeeze(-1)
