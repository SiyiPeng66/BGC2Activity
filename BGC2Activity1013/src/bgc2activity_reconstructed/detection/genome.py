"""Prokaryotic FASTA -> genes -> Pfam hits -> detector -> genomic BGC intervals."""
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from .data import RegionSample, collate_regions, load_pfam2vec
from .model import MultitaskBGCDetector
from .regions import (GeneRecord, call_regions, aggregate_window_probabilities,
                      aggregate_window_class_probabilities, PRODUCT_CLASSES, region_is_nrps)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def fasta(path):
    records=[];name=None;parts=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith('>'):
                if name is not None: records.append((name,''.join(parts).upper()))
                if not line[1:].strip(): raise ValueError('Empty FASTA identifier')
                name=line[1:].split()[0];parts=[]
            else:
                if name is None: raise ValueError('Sequence before FASTA header')
                parts.append(line)
    if name is not None: records.append((name,''.join(parts).upper()))
    if not records: raise ValueError('Empty FASTA')
    if len({n for n,s in records})!=len(records): raise ValueError('Duplicate FASTA identifiers')
    for n,s in records:
        if not s or set(s)-set('ACGTRYSWKMBDHVN'): raise ValueError(f'{n}: empty or non-DNA sequence')
    return sorted(records)

def write_table(path,rows,fields):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)

def nonredundant(hits):
    """Greedy overlap filter: discard >=50% overlap of shorter domain, any family.

    No clan file is assumed. This explicit conservative policy can discard
    genuine nested domains; both retained and excluded hits are exported.
    """
    grouped=defaultdict(list)
    for h in hits: grouped[h['gene_id']].append(h)
    kept=[];excluded=[]
    for group in grouped.values():
        selected=[]
        for h in sorted(group,key=lambda x:(-x['bitscore'],x['i_evalue'],x['pfam_id'],x['domain_start'])):
            conflict=next((k for k in selected if
                max(0,min(h['domain_end'],k['domain_end'])-max(h['domain_start'],k['domain_start'])+1)
                >=.5*min(h['domain_end']-h['domain_start']+1,k['domain_end']-k['domain_start']+1)),None)
            if conflict:
                excluded.append({**h,'reason':'overlap_ge_half_shorter_domain','retained_pfam':conflict['pfam_id']})
            else: selected.append(h)
        kept.extend(selected)
    return kept,excluded

def fixed_windows(n):
    """Fixed stride, with one terminal padded window when necessary."""
    if n<=256: return [0]
    starts=[0]
    while starts[-1]+256<n: starts.append(starts[-1]+128)
    return starts

@torch.no_grad()
def score_tokens(model,tokens,vocab,matrix):
    positions=[];classes=[]
    for start in fixed_windows(len(tokens)):
        window=tokens[start:start+256]; n=len(window)
        s=RegionSample('window',tuple(window),(0,)*n,0,0)
        b=collate_regions([s],vocab,matrix)
        pad=256-n
        x=torch.nn.functional.pad(b['embeddings'],(0,0,0,pad))
        mask=torch.nn.functional.pad(b['valid'],(0,pad))
        o=model(x,mask)
        positions.append((start,o.detection_logits[0,:n].sigmoid().numpy()))
        classes.append((o.class_logits[0].softmax(-1).numpy(),n))
    return aggregate_window_probabilities(positions,len(tokens)),aggregate_window_class_probabilities(classes)

def run(args):
    import pyrodigal,pyhmmer
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): raise ValueError('Output directory must be new or empty')
    for p in [args.genome,args.pfam_db,args.checkpoint,args.pfam2vec]:
        if not Path(p).is_file(): raise ValueError(f'Missing file: {p}')
    records=fasta(args.genome)
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    manifest=dict(status='running',input=str(Path(args.genome).resolve()),
        coordinates='1-based inclusive for both genomic and amino-acid intervals',
        versions=dict(pyrodigal=pyrodigal.__version__,pyhmmer=pyhmmer.__version__,torch=torch.__version__),
        parameters=dict(mode=args.mode,cpus=args.cpus,evalue=1e-5,window=256,stride=128,threshold=.45,
        overlap_filter='greedy bitscore; discard >=50% overlap of shorter domain; no clan mapping',
        gene_mode='auto: single trained on >=20kb contigs, meta for shorter contigs; single mode requires training sequence',
        limitations='prokaryotic genes only; linear contigs; no circular-origin joining; predicted product classes are auxiliary'),
        inputs={k:dict(path=str(Path(v).resolve()),sha256=digest(v)) for k,v in
        dict(genome=args.genome,pfam_db=args.pfam_db,checkpoint=args.checkpoint,pfam2vec=args.pfam2vec).items()})
    def save(): (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        torch.set_num_threads(args.cpus)
        meta=pyrodigal.GeneFinder(meta=True)
        long=[s.encode() for _,s in records if len(s)>=20000]
        single=None
        if args.mode!='meta':
            if long:
                single=pyrodigal.GeneFinder(meta=False);single.train(*long)
            elif args.mode=='single': raise ValueError('Single mode needs at least one contig >=20kb; use --mode meta for short sequences')
        genes=[];mode_counts=defaultdict(int)
        for ci,(contig,seq) in enumerate(records,1):
            finder=meta if args.mode=='meta' or (args.mode=='auto' and len(seq)<20000) else single
            mode_counts['meta' if finder is meta else 'single']+=1
            for gi,g in enumerate(sorted(finder.find_genes(seq.encode()),key=lambda g:(g.begin,g.end)),1):
                genes.append(dict(contig=contig,gene_id=f'c{ci:06d}_g{gi:06d}',start=int(g.begin),end=int(g.end),
                                  strand=int(g.strand),protein=g.translate().rstrip('*')))
        print(f'Predicted {len(genes)} genes on {len(records)} contigs',flush=True)
        with (out/'proteins.faa').open('w') as f:
            for g in genes: f.write(f">{g['gene_id']} contig={g['contig']} start={g['start']} end={g['end']} strand={g['strand']}\n{g['protein']}\n")
        alphabet=pyhmmer.easel.Alphabet.amino()
        sequences=[pyhmmer.easel.TextSequence(name=g['gene_id'].encode(),sequence=g['protein']).digitize(alphabet) for g in genes]
        hits=[];n_hmms=0
        def text(v): return v.decode() if isinstance(v,bytes) else str(v)
        if sequences:
            with pyhmmer.plan7.HMMFile(str(args.pfam_db)) as hmms:
                for top in pyhmmer.hmmer.hmmsearch(hmms,sequences,cpus=args.cpus,E=1e-5,domE=1e-5,incE=1e-5,incdomE=1e-5):
                    n_hmms+=1
                    accession=text(top.query.accession or top.query.name).split('.')[0]
                    for hit in top:
                        if hit.evalue>1e-5: continue
                        for d in hit.domains:
                            if d.i_evalue>1e-5: continue
                            a=d.alignment
                            hits.append(dict(gene_id=text(hit.name),pfam_id=accession,domain_start=int(a.target_from),
                                             domain_end=int(a.target_to),bitscore=float(d.score),i_evalue=float(d.i_evalue)))
                    if n_hmms%5000==0: print(f'Searched {n_hmms} Pfam profiles',flush=True)
        hits,excluded=nonredundant(hits)
        print(f'Kept {len(hits)} domains; excluded {len(excluded)} overlapping hits',flush=True)
        hit_fields=['gene_id','pfam_id','domain_start','domain_end','bitscore','i_evalue']
        write_table(out/'excluded_pfam_hits.tsv',excluded,hit_fields+['reason','retained_pfam'])
        by_gene=defaultdict(list)
        for h in hits: by_gene[h['gene_id']].append(h)
        vocab,m=load_pfam2vec(args.pfam2vec);matrix=torch.tensor(m)
        model=MultitaskBGCDetector(pfam2vec_dimension=m.shape[1])
        model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True));model.eval()
        contig_genes=defaultdict(list)
        for g in genes: contig_genes[g['contig']].append(g)
        regions=[];domain_rows=[];unknown=0
        for contig,seq in records:
            gs=contig_genes[contig]; ordered=[];gene_records=[]
            for g in gs:
                hs=sorted(by_gene[g['gene_id']],key=lambda h:((-h['domain_end'] if g['strand']==-1 else h['domain_start']),h['pfam_id']))
                first=len(ordered)
                ordered.extend(hs)
                gene_records.append(GeneRecord(g['gene_id'],g['start'],g['end'],tuple(range(first,len(ordered))),tuple(h['pfam_id'] for h in hs)))
            tokens=[h['pfam_id'] for h in ordered];unknown+=sum(t not in vocab for t in tokens)
            probabilities,_=score_tokens(model,tokens,vocab,matrix) if tokens else (np.zeros(0),None)
            for i,h in enumerate(ordered):
                domain_rows.append({**h,'contig':contig,'position_index':i+1,'bgc_probability':float(probabilities[i]),'oov':h['pfam_id'] not in vocab})
            for g,gr in zip(gs,gene_records):
                g['n_pfams']=len(gr.pfam_positions)
                g['bgc_probability']=float(max((probabilities[i] for i in gr.pfam_positions),default=0.))
                g['positive']=int(g['bgc_probability']>=.45)
            for region in call_regions(gene_records,probabilities):
                members=gs[region.start_index:region.end_index]
                region_tokens=[p for gr in gene_records[region.start_index:region.end_index] for p in gr.pfam_ids]
                _,class_probs=score_tokens(model,region_tokens,vocab,matrix)
                classes=dict(zip(PRODUCT_CLASSES,map(float,class_probs)))
                regions.append(dict(region_id=f'BGC{len(regions)+1:06d}',contig=contig,
                    start=min(g['start'] for g in members),end=max(g['end'] for g in members),
                    n_genes=len(members),n_positive_genes=region.positive_genes,n_pfams=len(region_tokens),
                    max_gene_probability=max(g['bgc_probability'] for g in members),
                    mean_gene_probability=float(np.mean([g['bgc_probability'] for g in members])),
                    product_class=max(classes,key=classes.get),nrps_positive=region_is_nrps(classes,set(region_tokens)),
                    gene_ids=';'.join(g['gene_id'] for g in members),**{f'class_probability_{k}':v for k,v in classes.items()}))
        gene_fields=['contig','gene_id','start','end','strand','n_pfams','bgc_probability','positive']
        write_table(out/'genes.tsv',genes,gene_fields)
        write_table(out/'pfam_hits.tsv',domain_rows,['contig','position_index']+hit_fields+['bgc_probability','oov'])
        fields=['region_id','contig','start','end','n_genes','n_positive_genes','n_pfams','max_gene_probability',
                'mean_gene_probability','product_class','nrps_positive','gene_ids']+[f'class_probability_{k}' for k in PRODUCT_CLASSES]
        write_table(out/'bgc_regions.tsv',regions,fields)
        with (out/'bgc_regions.gff3').open('w') as f:
            f.write('##gff-version 3\n')
            for contig,seq in records: f.write(f'##sequence-region {quote(contig,safe="")} 1 {len(seq)}\n')
            for r in regions:
                attributes=f"ID={r['region_id']};product_class={r['product_class']};nrps_positive={str(r['nrps_positive']).lower()}"
                f.write(f"{quote(r['contig'],safe='')}\tBGC2Activity\tregion\t{r['start']}\t{r['end']}\t.\t.\t.\t{attributes}\n")
        sequences=dict(records)
        with (out/'bgc_regions.fna').open('w') as f:
            for r in regions:
                f.write(f">{r['region_id']} {r['contig']}:{r['start']}-{r['end']} {r['product_class']}\n")
                dna=sequences[r['contig']][r['start']-1:r['end']]
                for i in range(0,len(dna),80): f.write(dna[i:i+80]+'\n')
        manifest.update(status='completed',counts=dict(contigs=len(records),genes=len(genes),pfam_profiles=n_hmms,
          domains=len(hits),oov_domains=unknown,genes_without_pfam=sum(g['n_pfams']==0 for g in genes),
          bgc_regions=len(regions),gene_prediction_modes=dict(mode_counts)),seconds=round(time.monotonic()-start,2))
        manifest['outputs']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='run_manifest.json'}
        save();print(f"Completed: {len(regions)} candidate BGC regions. See {out/'bgc_regions.tsv'}",flush=True)
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));save();raise
