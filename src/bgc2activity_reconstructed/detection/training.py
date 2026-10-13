"""Explicit reconstruction choices for adaptive loss weights and family batches."""
from collections import defaultdict, deque
import random
import torch

class AdaptiveWeights(torch.nn.Module):
    """Learn relative class/evolution weights, with fixed total mass 1.5.

    Manuscript does not specify an update parameterization. Normalized positive
    coefficients prevent the trivial solution of setting both losses to zero.
    Initial weights are 1.0 and 0.5; detection and balance weights stay 1 and .01.
    """
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.log(torch.tensor([1., .5])))
    def forward(self):
        return 1.5 * self.logits.softmax(0)


def family_batches(samples, seed, batch_size=64):
    """Cover all samples each epoch, grouped into 2 support + 2 query slots.

    Families with >=4 members use disjoint support/query identities; an
    incomplete last quartet is filled from other members of the same family.
    Families with <4 members contribute all distinct members without inventing
    copies. Singleton families cannot be triplet anchors. The support/query
    distinction determines grouping only: the manuscript loss uses every valid
    anchor (no inner-loop adaptation). All indices remain unique within a batch.
    """
    rng=random.Random(seed)
    families=defaultdict(list)
    for i,s in enumerate(samples): families[s.family_label].append(i)
    queues={}
    for f,indices in families.items():
        rng.shuffle(indices)
        chunks=[]
        for start in range(0,len(indices),4):
            chunk=indices[start:start+4]
            if len(chunk)<4 and len(indices)>=4:
                chunk+=rng.sample([i for i in indices if i not in chunk],4-len(chunk))
            chunks.append(chunk)
        queues[f]=deque(chunks)
    batch=[]; used=set()
    while queues:
        order=list(queues); rng.shuffle(order)
        for f in order:
            chunk=queues[f].popleft()
            if len(batch)+len(chunk)>batch_size or f in used:
                if batch: yield batch
                batch=[]; used=set()
            batch.extend(chunk); used.add(f)
            if not queues[f]: del queues[f]
    if batch: yield batch
