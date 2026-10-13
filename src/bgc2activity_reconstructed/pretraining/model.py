"""Three-layer chemical-context encoder and 96-dimensional statistics pooling."""
import torch
from torch import nn
class ChemicalAwarePeptideEncoder(nn.Module):
    def __init__(self,dropout=.10):
        super().__init__()
        self.text_projection=nn.Linear(24,96)
        self.descriptor_projection=nn.Linear(16,96)
        self.configuration_embedding=nn.Embedding(3,96)
        self.fusion=nn.Linear(192,96)
        self.position=nn.Parameter(torch.randn(1,64,96)*.02)
        self.mask_embedding=nn.Parameter(torch.randn(96)*.02)
        layer=nn.TransformerEncoderLayer(96,4,384,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.context_encoder=nn.TransformerEncoder(layer,3,enable_nested_tensor=False)
        self.output_norm=nn.LayerNorm(96);self.chemistry_head=nn.Linear(96,32)
        self.pool_attention=nn.Sequential(nn.Linear(96,48),nn.Tanh(),nn.Linear(48,1))
        self.pool_projection=nn.Sequential(nn.Linear(288,96),nn.GELU(),nn.LayerNorm(96))
    def forward(self,batch,mask=None):
        valid=batch['valid']
        h=self.fusion(torch.cat([self.text_projection(batch['text']),
            self.descriptor_projection(torch.cat([batch['descriptors'],batch['missing']],-1))],-1))
        h=h+self.configuration_embedding(batch['configuration'])
        if mask is not None:h=torch.where(mask[...,None],self.mask_embedding,h)
        h=h+self.position[:,:h.shape[1]]
        h=self.output_norm(self.context_encoder(h,src_key_padding_mask=~valid))
        weights=self.pool_attention(h).squeeze(-1).masked_fill(~valid,float('-inf')).softmax(1)[...,None]
        mean=(weights*h).sum(1);sd=((weights*(h-mean[:,None]).square()).sum(1)+1e-8).sqrt()
        maximum=h.masked_fill(~valid[...,None],float('-inf')).max(1).values
        return self.chemistry_head(h),self.pool_projection(torch.cat([mean,sd,maximum],-1))
    def freeze_for_mic(self):
        for p in self.parameters():p.requires_grad=False
        for module in [self.context_encoder.layers[2],self.output_norm,self.pool_attention,self.pool_projection]:
            for p in module.parameters():p.requires_grad=True
