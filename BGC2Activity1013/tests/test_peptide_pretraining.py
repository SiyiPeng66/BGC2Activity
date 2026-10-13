import sys,unittest
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bgc2activity_reconstructed.pretraining.chemistry import normalize,features,parse_tokens
from bgc2activity_reconstructed.pretraining.data import fit_preprocessor,encode,collate
from bgc2activity_reconstructed.pretraining.objective import corrupt,loss_values
from bgc2activity_reconstructed.pretraining.model import ChemicalAwarePeptideEncoder
class PeptideTests(unittest.TestCase):
    def test_chemistry_unknown_and_configuration(self):
        self.assertEqual(normalize('d-Orn'),('orn',-1))
        self.assertEqual(normalize('Orn'),('orn',0))
        self.assertAlmostEqual(float(features('dap')[1][4]),104.109,places=3)
        self.assertEqual(features('lys(lau)')[1][1],0)
        with self.assertRaises(ValueError):normalize('dpg')
        with self.assertRaises(ValueError):parse_tokens('ACXK')
    def test_masking_counts_and_original_targets(self):
        prep=fit_preprocessor([list('ACDEFGHIK')]);b=collate([encode(list('ACDEFGHIK'),prep)]*1000)
        before=b['descriptors'].clone();bank={k:v[0] for k,v in b.items() if k!='valid'}
        changed,selected,masked=corrupt(b,bank,torch.Generator().manual_seed(5))
        self.assertTrue(selected.any(1).all());self.assertTrue((~selected|b['valid']).all())
        self.assertAlmostEqual(float(selected.float().mean()),.3,delta=.03)
        self.assertAlmostEqual(float(masked.sum()/selected.sum()),.8,delta=.03)
        torch.testing.assert_close(before,b['descriptors'])
    def test_padding_invariance_and_pooling_gradient_contract(self):
        torch.manual_seed(4);prep=fit_preprocessor([list('ACDEFGHIK')]);model=ChemicalAwarePeptideEncoder();model.eval()
        one=collate([encode(list('ACDE'),prep)])
        two=collate([encode(list('ACDE'),prep),encode(list('ACDEFGHIK'),prep)])
        p,e=model(one);p2,e2=model(two)
        torch.testing.assert_close(e[0],e2[0],atol=1e-5,rtol=1e-5)
        selected=one['valid'];loss_values(p,one,selected)[0].backward()
        self.assertIsNotNone(model.text_projection.weight.grad)
        self.assertIsNone(model.pool_projection[0].weight.grad)
        model.freeze_for_mic()
        self.assertFalse(model.context_encoder.layers[0].linear1.weight.requires_grad)
        self.assertTrue(model.context_encoder.layers[2].linear1.weight.requires_grad)
        self.assertTrue(model.pool_projection[0].weight.requires_grad)
if __name__=='__main__':unittest.main()
