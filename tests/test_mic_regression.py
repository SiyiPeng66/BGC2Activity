import sys, unittest, tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bgc2activity_reconstructed.activity.model import MICRegressor, chemistry
from bgc2activity_reconstructed.pretraining.chemistry import MASS
from bgc2activity_reconstructed.pretraining.data import fit_preprocessor, encode, collate

from bgc2activity_reconstructed.activity.input import read_peptides

class MICTests(unittest.TestCase):
    def test_fasta_input_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'peptides.fasta'
            path.write_text('>first description\nAKK\nWK\n>second\nRRWQWRMKKLG\n')
            data=read_peptides(path)
            self.assertEqual(data.tokens.tolist(),['AKKWK','RRWQWRMKKLG'])
            self.assertEqual(data.sample_id.tolist(),['first','second'])
            for content in ['>same\nAKKWK\n>same\nAKKWK\n','>empty\n','>bad\nAKXWK\n','>bad\nAkkWK\n','AKKWK\n']:
                path.write_text(content)
                with self.assertRaises(ValueError):read_peptides(path,'fasta')

    def test_sequence_chemistry(self):
        x=chemistry('AKDE')
        self.assertAlmostEqual(float(x[1]),sum(MASS[t] for t in 'AKDE')-3*18.01528,places=3)
        self.assertLess(x[2],0)
        self.assertGreater(chemistry('AKKK')[2],2.9)
        self.assertTrue(np.isfinite(chemistry('AHHH')).all())
        self.assertEqual(x[5],0)

    def test_supervised_gradient_and_frozen_boundary(self):
        torch.manual_seed(42)
        model=MICRegressor({});model.train()
        prep=fit_preprocessor([list('AKDE'),list('AAAA')])
        batch=collate([encode(list('AKDE'),prep),encode(list('AAAA'),prep)])
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        loss=torch.nn.functional.huber_loss(model(batch,torch.zeros(2,14),torch.zeros(2,4,dtype=torch.long)),torch.tensor([3.,5.]))
        loss.backward()
        self.assertIsNone(model.encoder.context_encoder.layers[0].linear1.weight.grad)
        for param in [model.encoder.context_encoder.layers[2].linear1.weight,model.encoder.pool_projection[0].weight,model.head[0].weight]:
            self.assertGreater(float(param.grad.abs().sum()),0)
        optimizer.step()
        for n,p in model.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]),n)
        self.assertFalse(model.encoder.context_encoder.layers[0].training)
        self.assertTrue(model.encoder.context_encoder.layers[2].training)

if __name__=='__main__':unittest.main()
