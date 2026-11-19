"""Audit double-reviewed independent annotations; does not generate labels."""
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score
REQUIRED={'sample_id','reviewer_id','label','label_source','imagery_id','spatial_block','scene_id'}
BANNED=('ndvi','pseudo','synthetic','generated','heuristic','threshold','self-label')
def audit(df):
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.empty or df[list(REQUIRED)].isna().any().any(): raise ValueError('Missing annotation values')
    if df.duplicated(['sample_id','reviewer_id']).any(): raise ValueError('Duplicate reviewer annotation')
    if not set(df.label.unique()).issubset({0,1}): raise ValueError('Labels must be binary')
    if df.label_source.astype(str).str.lower().apply(lambda s:any(x in s for x in BANNED)).any(): raise ValueError('Pseudo-label source')
    for col in ('imagery_id','spatial_block','scene_id'):
        if (df.groupby('sample_id')[col].nunique()>1).any(): raise ValueError(f'Inconsistent {col}')
    counts=df.groupby('sample_id').reviewer_id.nunique()
    ids=counts[counts==2].index
    if len(ids)==0 or (counts>2).any(): raise ValueError('Need exactly two reviewers per double-reviewed sample')
    pairs=df[df.sample_id.isin(ids)]
    if pairs.groupby('sample_id').reviewer_id.apply(lambda x:tuple(sorted(x))).nunique()!=1: raise ValueError('Inconsistent reviewer pairs')
    pivot=pairs.pivot(index='sample_id',columns='reviewer_id',values='label').sort_index()
    a,b=pivot.iloc[:,0].to_numpy(),pivot.iloc[:,1].to_numpy()
    return {'total_samples':int(len(counts)),'double_reviewed_samples':int(len(ids)),
      'double_review_fraction':float(len(ids)/len(counts)),'reviewer_ids':list(map(str,pivot.columns)),
      'raw_agreement':float((a==b).mean()),
      'cohen_kappa':float(cohen_kappa_score(a,b)) if len(set(a)|set(b))>1 else None,
      'disagreement_sample_ids':[str(i) for i in pivot.index[a!=b]],
      'warning':'Agreement does not establish ground-truth validity; adjudicate disagreements.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('research_outputs/annotation_readiness.json'))
    args=p.parse_args();result=audit(pd.read_csv(args.annotations))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
