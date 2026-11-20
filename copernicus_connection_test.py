"""Development-only Copernicus connectivity test for Streamlit Community Cloud.

Deploy with main file copernicus_connection_test.py from
research/publication-upgrade. Does not display or log credentials.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import streamlit as st

from copernicus_cnn_source import fetch_red_nir, cnn_patches

st.set_page_config(page_title="Copernicus Connection Test", page_icon="🛰️")
st.title("🛰️ Copernicus Sentinel-2 Connection Test")
st.caption("Diagnostic build: 2026-11-20 / API-ERROR-V2")
st.caption("Development-only test — does not alter the production dashboard.")
st.info("This test uses a small Bengaluru area and reports NDVI. It does not validate CNN accuracy.")
st.write("**Region:** central Bengaluru (approx. 3 km wide)")
st.write("**Satellite:** Sentinel-2 Level-2A, red B04 and near-infrared B08")
if st.button("Test Copernicus API", type="primary"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=60)
    with st.spinner("Requesting satellite data from Copernicus..."):
        try:
            scene = fetch_red_nir(
                [77.578, 12.959, 77.606, 12.986],
                start.strftime("%Y-%m-%dT00:00:00Z"),
                end.strftime("%Y-%m-%dT23:59:59Z"),
                width=320,
                height=320,
                max_cloud=40,
            )
            valid = scene["valid"]
            st.success("API request succeeded and GeoTIFF bands were decoded.")
            st.metric("Valid pixels", f"{int(valid.sum()):,} / {valid.size:,}")
            if valid.any():
                st.metric("Mean NDVI (valid pixels)", f"{float(np.nanmean(scene['ndvi'])):.3f}")
                st.image(np.nan_to_num(scene["ndvi"], nan=0.0), caption="NDVI raster (nodata displayed as 0)", clamp=True)
            else:
                st.warning("Response succeeded but no clear pixels passed quality masking.")
            patches, locations = cnn_patches(scene)
            st.metric("CNN-compatible 32×32 patches", len(patches))
            st.caption("Patch preprocessing checked; CNN inference not yet executed.")
            st.caption("The selected time window may mosaic acquisitions. This is a connectivity check, not a publication-ready scene record.")
        except Exception as exc:
            st.error(f"Satellite request failed: {type(exc).__name__}: {exc}")
            st.caption("Do not paste API credentials into issue reports or screenshots.")
