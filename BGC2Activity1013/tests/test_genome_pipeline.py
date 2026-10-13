import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bgc2activity_reconstructed.detection.genome import fasta,nonredundant,fixed_windows,score_tokens
from bgc2activity_reconstructed.detection.regions import GeneRecord,call_regions

class Model(torch.nn.Module):
    def forward(self,x,mask):
        assert x.shape[1]==256
        return SimpleNamespace(detection_logits=torch.zeros(mask.shape),
                               class_logits=torch.zeros(1,5))
class GenomeTests(unittest.TestCase):
    def test_fixed_stride_terminal_padding_and_coverage(self):
        self.assertEqual(fixed_windows(363),[0,128])
        self.assertEqual(fixed_windows(905),[0,128,256,384,512,640,768])
        for n in [1,256,257,363,905]:
            p,c=score_tokens(Model(),['PF1']*n,{'<OOV>':0,'PF1':1},torch.zeros(2,100))
            self.assertEqual(len(p),n);np.testing.assert_allclose(p,.5);np.testing.assert_allclose(c,.2)
    def test_overlap_filter_keeps_separate_repeats(self):
        base=dict(gene_id='g',pfam_id='PF1',i_evalue=1e-9)
        hits=[dict(base,domain_start=1,domain_end=100,bitscore=50),
              dict(base,domain_start=10,domain_end=90,bitscore=30),
              dict(base,domain_start=150,domain_end=220,bitscore=40)]
        kept,removed=nonredundant(hits)
        self.assertEqual(len(kept),2);self.assertEqual(len(removed),1)
    def test_no_pfam_genes_are_gaps_and_extension_is_bounded(self):
        genes=[GeneRecord(str(i),i*100+1,i*100+90,(i,) if i in [0,2,4] else (),()) for i in range(7)]
        regions=call_regions(genes,np.ones(7))
        self.assertEqual(len(regions),1)
        self.assertEqual((regions[0].start_index,regions[0].end_index,regions[0].positive_genes),(0,7,3))
        self.assertEqual(call_regions([GeneRecord('x',1,100,(),())],np.zeros(0)),[])
    def test_duplicate_contigs_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.fna';p.write_text('>x\nACGT\n>x\nACGT\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):fasta(p)
if __name__=='__main__':unittest.main()
