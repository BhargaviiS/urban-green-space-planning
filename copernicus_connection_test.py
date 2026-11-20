"""Development-only Copernicus connectivity test for Streamlit Community Cloud.

Deploy with main file copernicus_connection_test.py from
research/publication-upgrade. Does not display or log credentials.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import streamlit as st

from copernicus_cnn_source import fetch_red_nir, cnn_patches
import hashlib
import io
import pandas as pd

st.set_page_config(page_title="Copernicus Connection Test", page_icon="🛰️")
st.title("🛰️ Copernicus Sentinel-2 Connection Test")
st.caption("Diagnostic build: 2026-11-20 / API-ERROR-V2")
st.caption("Development-only test — does not alter the production dashboard.")
st.info("This test uses a small Bengaluru area and reports NDVI. It does not validate CNN accuracy.")
st.subheader("Optional recovered CNN inference")
model_file = st.file_uploader("Upload recovered green_space_cnn_model.keras (1.1 MB)", type=["keras"], help="Model stays in this Streamlit session; do not upload untrusted model archives.")
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
            st.caption("Patch extraction verified; CNN inference requires the recovered model.")
            if model_file is not None and len(patches):
                try:
                    import tempfile
                    from pathlib import Path
                    import tensorflow as tf
                    raw = model_file.getvalue()
                    model_sha256 = hashlib.sha256(raw).hexdigest()
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / "green_space_cnn_model.keras"
                        path.write_bytes(raw)
                        model = tf.keras.models.load_model(str(path), compile=False, safe_mode=True)
                        if tuple(model.input_shape[1:]) != (32,32,2):
                            raise ValueError(f"Unexpected model input shape: {model.input_shape}")
                        probabilities = model.predict(patches, verbose=0).reshape(-1)
                    output = pd.DataFrame({
                        "patch_row": [int(row) for row,col in locations],
                        "patch_col": [int(col) for row,col in locations],
                        "cnn_probability": probabilities.astype(float),
                        "cnn_prediction_at_0_5": (probabilities >= .5).astype(int),
                    })
                    st.success("Recovered CNN inference completed on live Copernicus patches.")
                    st.metric("Predicted positive patches (threshold 0.5)", int((probabilities >= .5).sum()))
                    st.caption("Positive means the model's NDVI-derived training class, not independently verified green space.")
                    st.caption("Model SHA-256: " + model_sha256)
                    st.dataframe(output, hide_index=True)
                    st.download_button("Download CNN predictions CSV", output.to_csv(index=False).encode(), "copernicus_cnn_predictions.csv", "text/csv")
                except ImportError as exc:
                    st.error("TensorFlow unavailable in deployment: " + str(exc) + ". Use a TensorFlow-compatible Python version (3.12 recommended) and install tensorflow-cpu.")
                except Exception as exc:
                    st.error(f"CNN inference failed: {type(exc).__name__}: {exc}")

            st.caption("The selected time window may mosaic acquisitions. This is a connectivity check, not a publication-ready scene record.")
        except Exception as exc:
            st.error(f"Satellite request failed: {type(exc).__name__}: {exc}")
            st.caption("Do not paste API credentials into issue reports or screenshots.")
