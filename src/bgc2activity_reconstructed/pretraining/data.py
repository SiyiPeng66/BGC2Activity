"""Training-only feature statistics and strict prepared-corpus loading."""
import json
import numpy as np
import pandas as pd
import torch
from .chemistry import features,normalize,parse_tokens
def fit_preprocessor(rows):
    tokens=[t for row in rows for t in row]
    x=np.stack([features(t)[1] for t in tokens])
    med=np.array([np.median(x[np.isfinite(x[:,i]),i]) if np.isfinite(x[:,i]).any() else 0 for i in range(8)])
    filled=np.where(np.isfinite(x),x,med)
    vocab=sorted({normalize(t)[0] for t in tokens})
    return dict(median=med.tolist(),mean=filled.mean(0).tolist(),std=np.maximum(filled.std(0),1e-6).tolist(),
                vocabulary=['<pad>','<unk>','<mask>']+vocab,random_tokens=sorted(set(tokens)),
                all_missing=[bool(~np.isfinite(x[:,i]).any()) for i in range(8)])
def encode(tokens,prep):
    text=[];desc=[];missing=[];config=[]
    for token in tokens:
        identity,c=normalize(token)
        t,d=features(token)
        if identity not in prep['vocabulary']:
            t=np.zeros(24,dtype=np.float32);d=np.full(8,np.nan,dtype=np.float32)
        absent=~np.isfinite(d);d=np.where(absent,np.asarray(prep['median']),d)
        d=(d-np.asarray(prep['mean']))/np.asarray(prep['std'])
        text.append(t);desc.append(d);missing.append(absent);config.append({0:0,1:1,-1:2}[c])
    return np.asarray(text,np.float32),np.asarray(desc,np.float32),np.asarray(missing,np.float32),np.asarray(config,np.int64)
def collate(items):
    n=max(len(x[0]) for x in items);b=len(items)
    text=torch.zeros(b,n,24);desc=torch.zeros(b,n,8);missing=torch.ones(b,n,8)
    config=torch.zeros(b,n,dtype=torch.long);valid=torch.zeros(b,n,dtype=torch.bool)
    for i,(t,d,m,c) in enumerate(items):
        k=len(t);text[i,:k]=torch.from_numpy(t);desc[i,:k]=torch.from_numpy(d)
        missing[i,:k]=torch.from_numpy(m);config[i,:k]=torch.from_numpy(c);valid[i,:k]=True
    return dict(text=text,descriptors=desc,missing=missing,configuration=config,valid=valid)
def read_corpus(path):
    d=pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False)
    required={'sample_id','tokens','cluster_id','split'}
    if not required.issubset(d):raise ValueError('Corpus requires '+str(sorted(required)))
    if d.sample_id.duplicated().any():raise ValueError('Duplicate sample IDs')
    if not set(d.split).issubset({'train','validation','test'}):raise ValueError('Invalid split')
    if d.groupby('cluster_id').split.nunique().max()>1:raise ValueError('Structural cluster leakage')
    if d.groupby('tokens').split.nunique().max()>1:raise ValueError('Exact token leakage')
    d['parsed']=d.tokens.map(parse_tokens)
    if set(d.split)!={'train','validation','test'}:raise ValueError('All three partitions required')
    return d
