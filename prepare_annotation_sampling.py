"""Deterministic spatially stratified *candidate locations* for manual imagery review.

This does not retrieve imagery, label vegetation, or produce ground truth.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def prepare(input_csv, output_csv, per_cell=3, grid=6, seed=2026):
    df=pd.read_csv(input_csv)
    if not {'latitude','longitude'}.issubset(df.columns): raise ValueError('Coordinates required')
    if df[['latitude','longitude']].isna().any().any() or df[['latitude','longitude']].duplicated().any(): raise ValueError('Invalid coordinates')
    if not (df.latitude.between(-90,90)&df.longitude.between(-180,180)).all(): raise ValueError('Out of bounds')
    if grid<2 or per_cell<1: raise ValueError('Invalid sampling parameters')
    d=df[['latitude','longitude']].copy()
    d['lat_bin']=np.minimum(np.floor((d.latitude-d.latitude.min())/(d.latitude.max()-d.latitude.min()+1e-12)*grid).astype(int),grid-1)
    d['lon_bin']=np.minimum(np.floor((d.longitude-d.longitude.min())/(d.longitude.max()-d.longitude.min()+1e-12)*grid).astype(int),grid-1)
    d['spatial_block']=d.apply(lambda r:f"B{int(r.lat_bin):02d}_{int(r.lon_bin):02d}",axis=1)
    picks=[]
    for _,group in d.groupby('spatial_block',sort=True):
        picks.append(group.sample(n=min(per_cell,len(group)),random_state=seed))
    result=pd.concat(picks).sort_values(['spatial_block','latitude','longitude']).reset_index(drop=True)
    result.insert(0,'sample_id',[f'UGS-{i:04d}' for i in range(1,len(result)+1)])
    result['imagery_id']=''
    result['imagery_date']=''
    result['imagery_license']=''
    result['reviewer_a_label']=''
    result['reviewer_b_label']=''
    result['adjudicated_reference_label']=''
    result['label_source']=''
    result['review_status']='NOT_ANNOTATED'
    result['notes']=''
    result.drop(columns=['lat_bin','lon_bin'],inplace=True)
    output_csv=Path(output_csv);output_csv.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_csv,index=False)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='all_candidates_scored.csv');p.add_argument('--output',default='research_outputs/annotation_sampling_manifest.csv');p.add_argument('--per-cell',type=int,default=3);p.add_argument('--grid',type=int,default=6)
    a=p.parse_args();r=prepare(a.input,a.output,a.per_cell,a.grid)
    print(f'Prepared {len(r)} unannotated locations across {r.spatial_block.nunique()} spatial blocks')
