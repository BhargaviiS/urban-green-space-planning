import unittest
import pandas as pd
from audit_annotation_readiness import audit
class AnnotationReadinessTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame([{'sample_id':f's{i}','reviewer_id':r,'label':y,
          'label_source':'manual_visual_review','imagery_id':f'im{i}',
          'spatial_block':f'b{i}','scene_id':f'scene{i}'}
          for i in range(4) for r,y in [('A',i%2),('B',i%2 if i!=3 else 0)]])
    def test_agreement(self):
        self.assertEqual(audit(self.df)['disagreement_sample_ids'],['s3'])
    def test_duplicates(self):
        with self.assertRaises(ValueError): audit(pd.concat([self.df,self.df.iloc[[0]]]))
    def test_pseudolabel(self):
        d=self.df.copy();d.loc[0,'label_source']='NDVI threshold'
        with self.assertRaises(ValueError): audit(d)
    def test_single_reviewer(self):
        with self.assertRaises(ValueError): audit(self.df[self.df.reviewer_id=='A'])
    def test_imagery_mismatch(self):
        d=self.df.copy();d.loc[0,'imagery_id']='different'
        with self.assertRaises(ValueError): audit(d)
    def test_bad_label(self):
        d=self.df.copy();d.loc[0,'label']=2
        with self.assertRaises(ValueError): audit(d)
if __name__=='__main__':unittest.main()
