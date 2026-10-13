"""Auditable monomer features; unresolved chemistry is never invented."""
import hashlib,re
import numpy as np
from .registry import get as registry_get
TEXT_DIM=24
DESCRIPTOR_DIM=8
NAMES=dict(zip('ACDEFGHIKLMNPQRSTVWY','alanine cysteine aspartate glutamate phenylalanine glycine histidine isoleucine lysine leucine methionine asparagine proline glutamine arginine serine threonine valine tryptophan tyrosine'.split()))
MASS=dict(zip(NAMES,[89.094,121.154,133.104,147.131,165.192,75.067,155.156,131.175,146.189,131.175,149.208,132.119,115.132,146.146,174.203,105.093,119.119,117.148,204.228,181.191]))
HYDRO=dict(zip(NAMES,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
ALIASES=dict(zip('ala cys asp glu phe gly his ile lys leu met asn pro gln arg ser thr val trp tyr'.split(),NAMES))
ALIASES.update({'24dab':'dab','asp-gamma':'beta-asp','ornithine':'orn','pipecolate':'pip'})
# Masses describe free monomers, not peptide residue masses. Hydrophobicity for
# nonstandard entities stays missing rather than using the old proxy estimates.
SPECIAL={
 'orn':('ornithine diamino aliphatic',132.162,1,1,0,0,0),
 'dab':('2 4 diaminobutyric acid',118.136,1,1,0,0,0),
 'dap':('2 3 diaminopropionic acid',104.109,1,1,0,0,0),
 'hse':('homoserine hydroxyl aliphatic',119.119,0,1,0,0,0),
 'pip':('pipecolic acid cyclic secondary amine',129.159,0,0,0,0,1),
 'beta-ala':('beta alanine',89.094,0,0,0,0,0),
 'beta-asp':('beta linked aspartic acid',133.104,-1,1,0,0,0),
 'lys(lau)':('epsilon lauroyl lysine lipid amide',328.496,0,1,0,12,0),
 'lys(myr)':('epsilon myristoyl lysine lipid amide',356.55,0,1,0,14,0),
}
# Additional identities recovered from explicit DBAASP modification descriptions.
# Names/formula evidence is exported by prepare_chemical_models.py. Missing
# hydropathy remains missing. This is not a claim to recover the original 24.
SPECIAL.update({
 'aib':('alpha aminoisobutyric acid',103.121,0,0,0,0,0),
 'nle':('norleucine 2 aminohexanoic acid',131.175,0,0,0,0,0),
 'nva':('norvaline',117.148,0,0,0,0,0),
 'abu':('alpha aminobutyric acid',103.121,0,0,0,0,0),
 'sar':('sarcosine N methylglycine',89.094,0,0,0,0,0),
 '4-hyp':('4 hydroxyproline unspecified ring stereochemistry',131.131,0,1,0,0,1),
 'dha':('2 3 didehydroalanine',87.078,0,0,0,0,0),
 'dhb':('Z 2 3 didehydrobutyrine',101.105,0,0,0,0,0),
 'me-leu':('N methylleucine',145.202,0,0,0,0,0),
 'me-val':('N methylvaline',131.175,0,0,0,0,0),
 'me-phe':('N methylphenylalanine',179.219,0,0,1,0,1),
 'me-ala':('N methylalanine',103.121,0,0,0,0,0),
 'harg':('homoarginine guanidino',188.230,1,1,0,0,0),
 'ahx':('6 aminohexanoic acid epsilon aminocaproic acid',131.175,0,0,0,0,0),
 '1-nal':('1 naphthylalanine',215.252,0,0,1,0,1),
 '2-nal':('2 naphthylalanine',215.252,0,0,1,0,1),
})
ALIASES.update({'βala':'beta-ala','β-ala':'beta-ala'})
def normalize(token):
    token=token.strip();config=0
    if token[:2].lower() in ['d-','l-']:
        config=-1 if token[:2].lower()=='d-' else 1;token=token[2:]
    identity=token if token in NAMES else ALIASES.get(token.lower(),token.lower())
    if identity not in NAMES and identity not in SPECIAL and registry_get(identity) is None:raise ValueError(f'Unresolved monomer: {token}')
    return identity,config
def text_vector(text):
    v=np.zeros(24,dtype=np.float32)
    for word in re.findall('[a-z0-9]+',text.lower()):
        h=hashlib.blake2b(word.encode(),digest_size=8).digest()
        v[int.from_bytes(h[:4],'little')%24]+=1 if h[4]%2==0 else -1
    return v/max(np.linalg.norm(v),1e-12)
def features(token):
    identity,config=normalize(token)
    if identity in NAMES:
        charge=1 if identity in 'KR' else (-1 if identity in 'DE' else (np.nan if identity=='H' else 0))
        desc=[HYDRO[identity],charge,int(identity in 'CDEHKNQRSTY'),int(identity in 'FWHY'),MASS[identity],
              float(config) if config else np.nan,0,int(identity in 'PFWHY')]
        text=NAMES[identity]+' amino acid'
    elif identity not in SPECIAL:
        entry=registry_get(identity);text=entry['description']
        desc=[np.nan if v is None else float(v) for v in entry['descriptors']]
        desc[5]=float(config) if config else np.nan
    else:
        text,mass,charge,polar,aromatic,lipid,cyclic=SPECIAL[identity]
        desc=[np.nan,charge,polar,aromatic,mass,float(config) if config else np.nan,lipid,cyclic]
    text+=' configuration '+{0:'unspecified',1:'L',-1:'D'}[config]
    return text_vector(text),np.asarray(desc,dtype=np.float32)
def parse_tokens(value):
    if ';' in value:tokens=value.split(';')
    elif re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',value):tokens=list(value)
    else: tokens=[value]
    if not 4<=len(tokens)<=64:raise ValueError('Peptide must contain 4–64 monomers; use semicolons for nonstandard tokens')
    for token in tokens:normalize(token)
    return tokens
