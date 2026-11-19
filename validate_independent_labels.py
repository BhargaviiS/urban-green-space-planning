"""Evaluate independent binary vegetation labels and audit split overlap."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support,roc_auc_score
REQUIRED={'sample_id','scene_id','spatial_block','reference_label','predicted_probability','label_source'}
REJECTED=('ndvi','pseudo','synthetic','generated','heuristic','threshold','self-label')
def validate(df):
    if not REQUIRED.issubset(df.columns): raise ValueError(f'Missing: {sorted(REQUIRED-set(df.columns))}')
    if df.empty or df[list(REQUIRED)].isna().any().any(): raise ValueError('Missing values')
    if df.sample_id.duplicated().any(): raise ValueError('Duplicate sample ID')
    if not set(df.reference_label.unique()).issubset({0,1}): raise ValueError('Nonbinary label')
    if not df.predicted_probability.between(0,1).all(): raise ValueError('Probability out of bounds')
    if df.label_source.astype(str).str.lower().apply(lambda s:any(x in s for x in REJECTED)).any(): raise ValueError('Pseudo-labels rejected')
    if df.spatial_block.nunique()<3 or df.scene_id.nunique()<2: raise ValueError('Insufficient blocks or scenes')
    return df.copy()
def evaluate(df,threshold=.5):
    df=validate(df);y=df.reference_label.to_numpy(dtype=int);pred=(df.predicted_probability.to_numpy()>=threshold).astype(int)
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return {'n':len(df),'scenes':int(df.scene_id.nunique()),'spatial_blocks':int(df.spatial_block.nunique()),
      'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'precision_per_class':p.tolist(),'recall_per_class':r.tolist(),'f1_per_class':f.tolist(),
      'support_per_class':s.tolist(),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
      'roc_auc':float(roc_auc_score(y,df.predicted_probability)) if len(np.unique(y))==2 else None,
      'warning':'Does not prove reference-label independence or training/test separation.'}
def audit_splits(train,test):
    a,b=validate(train),validate(test)
    return {'overlapping_samples':sorted(set(a.sample_id)&set(b.sample_id)),
      'overlapping_spatial_blocks':sorted(set(a.spatial_block)&set(b.spatial_block)),
      'overlapping_scenes':sorted(set(a.scene_id)&set(b.scene_id)),
      'spatially_disjoint':not bool(set(a.spatial_block)&set(b.spatial_block)),
      'temporally_disjoint':not bool(set(a.scene_id)&set(b.scene_id))}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',type=Path,required=True)
    p.add_argument('--train',type=Path);p.add_argument('--output',type=Path,default=Path('research_outputs/independent_evaluation.json'))
    args=p.parse_args();test=validate(pd.read_csv(args.test));result=evaluate(test)
    if args.train:
        audit=audit_splits(pd.read_csv(args.train),test);result['split_audit']=audit
        if not audit['spatially_disjoint'] or audit['overlapping_samples']: raise ValueError('Split leakage')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
