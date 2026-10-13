import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bgc2activity_reconstructed.pretraining.registry import install,entries
from bgc2activity_reconstructed.pretraining.chemistry import normalize,features
from bgc2activity_reconstructed.activity.chemical import candidate
class RegistryTests(unittest.TestCase):
 def tearDown(self):install({})
 def test_distinct_named_missing_chemistry(self):
  install({k:dict(description='source monomer '+k,source='test fixture',status='label_only',descriptors=[None]*8) for k in ['hxorn','other-monomer']})
  self.assertEqual(normalize('D-HxOrn'),('hxorn',-1))
  a,x=features('hxorn');b,y=features('other-monomer')
  self.assertFalse(np.array_equal(a,b));self.assertTrue(np.isnan(x).all())
  chem,states=candidate(['A','hxorn','K','F'])
  self.assertTrue(np.isnan(chem[1]));self.assertEqual(states[2],0)
  snapshot=entries();install({})
  with self.assertRaises(ValueError):normalize('hxorn')
  install(snapshot);self.assertEqual(normalize('hxorn')[0],'hxorn')
 def test_reject_unregistered(self):
  with self.assertRaises(ValueError):normalize('never-registered-monomer')
if __name__=='__main__':unittest.main()
