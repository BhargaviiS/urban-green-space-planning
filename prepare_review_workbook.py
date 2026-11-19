"""Prepare two blind annotation worksheets and spatial holdout assignments."""
from pathlib import Path
import argparse
import pandas as pd
REQUIRED={'sample_id','latitude','longitude','spatial_block','review_status'}
def build(manifest, output_dir):
    df=pd.read_csv(manifest,keep_default_na=False)
    if not REQUIRED.issubset(df): raise ValueError('Missing sampling columns')
    if df.empty or df.sample_id.duplicated().any(): raise ValueError('Invalid sample IDs')
    if df[['latitude','longitude']].isna().any().any(): raise ValueError('Missing coordinates')
    if not df.review_status.eq('NOT_ANNOTATED').all(): raise ValueError('Input must be unannotated')
    blocks=sorted(df.spatial_block.unique())
    assignments={b:('held_out_test' if i%5==0 else 'development') for i,b in enumerate(blocks)}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    base=df[['sample_id','latitude','longitude','spatial_block']].copy()
    base['split']=base.spatial_block.map(assignments)
    for c,v in [('reference_imagery_url',''),('image_acquisition_date',''),('imagery_license',''),('cloud_or_shadow',''),('label',''),('label_source','manual_visual_review'),('reviewer_confidence',''),('reviewer_notes','')]:base[c]=v
    for reviewer in ('A','B'):
        sheet=base.copy();sheet.insert(1,'reviewer_id',reviewer)
        sheet.to_csv(out/f'reviewer_{reviewer}_blind.csv',index=False)
    base[['sample_id','spatial_block','split']].to_csv(out/'spatial_holdout_assignment.csv',index=False)
    return {'samples':len(base),'blocks':len(blocks),'held_out_test_samples':int((base.split=='held_out_test').sum()),'held_out_blocks':sum(x=='held_out_test' for x in assignments.values())}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='research_outputs/annotation_sampling_manifest.csv');p.add_argument('--output-dir',default='research_outputs/reviewer_pack');a=p.parse_args();print(build(a.manifest,a.output_dir))
