"""Copernicus Sentinel-2 L2A B04/B08 retrieval for reproducible CNN inference.

Requires COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in the server
environment or Streamlit secrets. Never commit credentials.
"""
import io
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
EVALSCRIPT = """//VERSION=3
function setup() {
  return {input:[{bands:["B04","B08","dataMask","SCL"],units:"REFLECTANCE"}],
    output:{bands:4,sampleType:"FLOAT32"}};
}
function evaluatePixel(s) {
  return [s.B04,s.B08,s.dataMask,s.SCL];
}
"""

def _credential(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except (ImportError, FileNotFoundError, KeyError, AttributeError):
        return None

def access_token():
    client_id = _credential("COPERNICUS_CLIENT_ID")
    secret = _credential("COPERNICUS_CLIENT_SECRET")
    if not client_id or not secret:
        raise RuntimeError("Configure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET as server secrets.")
    r = requests.post(TOKEN_URL, data={"grant_type":"client_credentials",
        "client_id":client_id,"client_secret":secret}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

def fetch_red_nir(bbox, start, end, width=320, height=320, max_cloud=30):
    """Retrieve reflectance B04/B08 and quality bands in GeoTIFF.

    bbox: [west,south,east,north] in WGS84. For scientific comparisons,
    pin exact acquisition separately; a time window may mosaic scenes.
    """
    if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError("bbox must be [west,south,east,north]")
    if not (1 <= width <= 2500 and 1 <= height <= 2500):
        raise ValueError("Invalid image dimensions")
    payload = {
        "input":{"bounds":{"bbox":list(bbox),"properties":{"crs":"http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
          "data":[{"type":"S2L2A",
            "dataFilter":{"timeRange":{"from":start,"to":end},
              "maxCloudCoverage":max_cloud,"mosaickingOrder":"mostRecent"}}]},
        "output":{"width":width,"height":height,
          "responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
        "evalscript":EVALSCRIPT,
    }
    response = requests.post(PROCESS_URL,json=payload,
        headers={"Authorization":f"Bearer {access_token()}","Accept":"image/tiff"},
        timeout=120)
    if not response.ok:
        try:
            detail = response.json()
            message = str(detail.get("error", detail.get("message", detail)))
        except ValueError:
            message = response.text[:600]
        raise RuntimeError(f"Copernicus Process API HTTP {response.status_code}: {message[:600]}")
    import rasterio
    from rasterio.io import MemoryFile
    with MemoryFile(response.content) as mem:
        with mem.open() as dataset:
            bands = dataset.read().astype("float32")
            transform = dataset.transform
            crs = str(dataset.crs)
    red,nir,mask,scl = bands
    # SCL 3=shadow, 8/9/10=cloud, 11=snow; exclude unclassified 7 conservatively.
    valid = (mask > 0) & ~np.isin(scl.astype("int16"), [0,1,3,7,8,9,10,11])
    valid &= np.isfinite(red) & np.isfinite(nir) & ((red+nir)>0)
    ndvi = np.full(red.shape,np.nan,dtype="float32")
    ndvi[valid] = (nir[valid]-red[valid])/(nir[valid]+red[valid])
    return {"red":red,"nir":nir,"valid":valid,"ndvi":ndvi,
            "transform":transform,"crs":crs,
            "requested_window":{"start":start,"end":end},
            "note":"Response can mosaic scenes; record acquisition IDs separately for publication."}

def cnn_patches(scene, patch_size=32, min_valid_fraction=.95):
    """Create CNN inputs using original per-scene 2nd/98th percentile scaling.

    Only fully supported patches returned. Model outputs reflect NDVI
    pseudo-label training, not independently validated land-cover truth.
    """
    red,nir,valid = scene["red"],scene["nir"],scene["valid"]
    if not valid.any():
        raise ValueError("No valid clear pixels")
    channels=[]
    for band in (red,nir):
        low,high=np.percentile(band[valid],[2,98])
        if high <= low:
            raise ValueError("Constant band values")
        channels.append(np.clip((band-low)/(high-low),0,1))
    arr=np.stack(channels,axis=-1).astype("float32")
    patches,locations=[],[]
    h,w=valid.shape
    for row in range(0,h-patch_size+1,patch_size):
        for col in range(0,w-patch_size+1,patch_size):
            if valid[row:row+patch_size,col:col+patch_size].mean() >= min_valid_fraction:
                patches.append(arr[row:row+patch_size,col:col+patch_size])
                locations.append((row,col))
    if not patches:
        return np.empty((0,patch_size,patch_size,2),dtype="float32"),locations
    return np.stack(patches),locations
