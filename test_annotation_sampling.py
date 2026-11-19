import tempfile
import unittest
from pathlib import Path
import pandas as pd
from prepare_annotation_sampling import prepare

class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.input=Path(self.tmp.name)/'input.csv'
        self.output=Path(self.tmp.name)/'out.csv'
        pd.DataFrame({'latitude':[12+i*.01 for i in range(25)],'longitude':[77+i*.012 for i in range(25)]}).to_csv(self.input,index=False)
    def test_reproducible(self):
        a=prepare(self.input,self.output,2,3)
        b=prepare(self.input,self.output,2,3)
        pd.testing.assert_frame_equal(a,b)
    def test_no_labels_invented(self):
        a=prepare(self.input,self.output)
        self.assertTrue(a.adjudicated_reference_label.eq('').all())
        self.assertTrue(a.review_status.eq('NOT_ANNOTATED').all())
    def test_duplicate_rejected(self):
        d=pd.read_csv(self.input)
        pd.concat([d,d.iloc[[0]]]).to_csv(self.input,index=False)
        with self.assertRaises(ValueError):prepare(self.input,self.output)
    def test_invalid_grid(self):
        with self.assertRaises(ValueError):prepare(self.input,self.output,grid=1)
if __name__=='__main__':unittest.main()
