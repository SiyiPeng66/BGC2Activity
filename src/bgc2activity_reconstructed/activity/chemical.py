"""Explicit monomer/terminal chemistry. Unknown chemical quantities stay missing."""
import numpy as np
from ..pretraining.chemistry import features, normalize, NAMES, parse_tokens
FIELDS=['length','mass','charge_pH7','gravy','aromatic_fraction','nonstandard_fraction','lipidation']
N_STATES=['unknown','free','acetyl']
C_STATES=['unknown','free','amide']
CYCLE_STATES=['unknown','linear','head_tail']

def candidate(tokens,n_terminal='unknown',c_terminal='unknown',cyclization='unknown'):
    if isinstance(tokens,str):tokens=parse_tokens(tokens)
    n=N_STATES.index(n_terminal);c=C_STATES.index(c_terminal);cy=CYCLE_STATES.index(cyclization)
    if cy==2 and (n!=0 or c!=0):raise ValueError('Head-tail cycle must not also specify terminal groups')
    ids=[normalize(t)[0] for t in tokens];x=np.stack([features(t)[1] for t in tokens])
    lipid=bool(np.any(x[:,6]>0));states=np.array([n,c,2 if lipid else (0 if not np.isfinite(x[:,6]).all() else 1),cy],np.int64)
    mass=float(x[:,4].sum()-18.01528*(len(tokens)-1))
    charge=0.;known_charge=True
    basic={'K':10.5,'R':12.5,'H':6.,'orn':10.5,'dab':10.5,'dap':10.5,'harg':12.5}
    acidic={'D':3.86,'E':4.25,'C':8.33,'Y':10.07,'beta-asp':3.86}
    for identity,desc in zip(ids,x):
        if identity in basic:charge+=1/(1+10**(7-basic[identity]))
        elif identity in acidic:charge-=1/(1+10**(acidic[identity]-7))
        elif desc[1]!=0:known_charge=False
    if cy==2:mass-=18.01528
    elif cy==0:mass=np.nan;known_charge=False
    else:
        if n==0 or c==0:mass=np.nan;known_charge=False
        if n==2:mass+=42.03668
        if c==2:mass-=.98476
        if n==1:charge+=1/(1+10**(7-9.69))
        if c==1:charge-=1/(1+10**(2.34-7))
    # A partial average would silently substitute standard residues for missing
    # nonstandard hydropathy. Mark GRAVY missing unless all residues are known.
    gravy=float(x[:,0].mean()) if np.isfinite(x[:,0]).all() else np.nan
    return np.array([len(tokens),mass,charge if known_charge else np.nan,gravy,
                     x[:,3].mean(),np.mean([i not in NAMES for i in ids]),x[:,6].sum()],np.float32),states

def fit_numeric(raw):
    raw=np.asarray(raw)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in raw.T])
    fill=np.where(np.isfinite(raw),raw,med)
    return dict(median=med.tolist(),mean=fill.mean(0).tolist(),std=np.maximum(fill.std(0),1e-6).tolist())

def transform_numeric(raw,prep):
    raw=np.asarray(raw);missing=~np.isfinite(raw)
    filled=np.where(missing,prep['median'],raw)
    return np.concatenate([(filled-prep['mean'])/np.asarray(prep['std']),missing],1).astype(np.float32)
