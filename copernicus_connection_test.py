"""Development-only Copernicus connectivity test for Streamlit Community Cloud.

Deploy with main file copernicus_connection_test.py from
research/publication-upgrade. Does not display or log credentials.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import streamlit as st

from copernicus_cnn_source import fetch_red_nir, cnn_patches, search_candidate_acquisitions
import hashlib
import io
import zipfile
import json
import pandas as pd

st.set_page_config(page_title="Copernicus Connection Test", page_icon="🛰️")
st.title("🛰️ Copernicus Sentinel-2 Connection Test")
st.caption("Research build: 2026-11-20 / CATALOG-V5")
st.caption("Development-only test — does not alter the production dashboard.")
st.info("This test uses a small Bengaluru area and reports NDVI. It does not validate CNN accuracy.")
st.subheader("Optional recovered CNN inference")
model_file = st.file_uploader("Upload recovered green_space_cnn_model.keras (1.1 MB)", type=["keras"], help="Model stays in this Streamlit session; do not upload untrusted model archives.")
if model_file is None:
    st.warning("Model upload not confirmed in the current session. Satellite-only testing remains available.")
else:
    try:
        uploaded_size = model_file.size
        if uploaded_size < 100_000:
            st.error(f"Uploaded file appears incomplete ({uploaded_size:,} bytes). Please retry after the app stabilizes.")
            model_file = None
        else:
            st.success(f"Model upload received by server: {uploaded_size:,} bytes")
    except Exception as exc:
        st.error(f"Unable to confirm model upload: {type(exc).__name__}: {exc}")
        model_file = None
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
            st.subheader("Satellite acquisition candidates")
            st.caption("Catalog results intersect the requested area/time; they do NOT identify which acquisitions the Process API mosaic actually used.")
            try:
                candidates = search_candidate_acquisitions(
                    [77.578, 12.959, 77.606, 12.986],
                    start.strftime("%Y-%m-%dT00:00:00Z"),
                    end.strftime("%Y-%m-%dT23:59:59Z"), max_cloud=40)
                if candidates:
                    st.dataframe(pd.DataFrame(candidates), hide_index=True)
                    st.download_button("Download candidate acquisition catalog CSV",
                        pd.DataFrame(candidates).to_csv(index=False).encode(),
                        "copernicus_candidate_acquisitions.csv", "text/csv")
                else:
                    st.warning("Catalog returned no candidate acquisitions for this query.")
            except Exception as catalog_exc:
                st.warning(f"Acquisition catalog lookup unavailable: {type(catalog_exc).__name__}: {catalog_exc}")
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
            # Export RGB thumbnails without CNN probabilities or NDVI for blind review.
            if locations:
                from PIL import Image
                rgb = np.stack([scene["red"], scene["green"], scene["blue"]], axis=-1)
                valid_rgb = rgb[valid]
                lo = np.percentile(valid_rgb, 2, axis=0)
                hi = np.percentile(valid_rgb, 98, axis=0)
                display_rgb = np.uint8(255 * np.clip((rgb - lo) / np.maximum(hi-lo, 1e-6), 0, 1))
                display_rgb[~valid] = 0
                st.image(display_rgb, caption="Sentinel-2 RGB preview (contrast stretched; not ground truth)")
                review_rows = []
                archive_bytes = io.BytesIO()
                with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
                    for row, col in locations:
                        patch_id = f"r{row:03d}_c{col:03d}"
                        tile = display_rgb[row:row+32, col:col+32]
                        png = io.BytesIO()
                        Image.fromarray(tile).resize((256,256), Image.Resampling.NEAREST).save(png, format="PNG")
                        archive.writestr(f"rgb_patches/{patch_id}.png", png.getvalue())
                        review_rows.append({"patch_row":row, "patch_col":col,
                            "reference_label":"", "annotator":"", "source_id":"",
                            "patch_id":patch_id})
                    review_csv = pd.DataFrame(review_rows).to_csv(index=False)
                    archive.writestr("reference_labels_BLANK.csv", review_csv)
                    archive.writestr("README.txt",
                        "Blind RGB review images; CNN probabilities and NDVI are intentionally omitted.\\n"
                        "Assign labels only after independent visual inspection and a fixed vegetation definition.\\n"
                        "Source is Sentinel-2 Process API mosaicking across a rolling 60-day window.\\n"
                        "This diagnostic scene is not a spatially independent publication test set.\\n"
                        "source_id must identify imagery and date; acquire exact scene IDs for formal validation.\\n")
                st.download_button("Download blind RGB patch review bundle (ZIP)",
                    archive_bytes.getvalue(), "copernicus_blind_rgb_review.zip", "application/zip")

            st.caption("Patch extraction verified; CNN inference requires the recovered model.")
            if model_file is not None and len(patches):
                try:
                    import tempfile
                    from pathlib import Path
                    import os
                    os.environ.setdefault("KERAS_BACKEND", "numpy")
                    import keras
                    raw = model_file.getvalue()
                    model_sha256 = hashlib.sha256(raw).hexdigest()
                    with tempfile.TemporaryDirectory() as folder:
                        path = Path(folder) / "green_space_cnn_model.keras"
                        path.write_bytes(raw)
                        model = keras.models.load_model(str(path), compile=False, safe_mode=True)
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
                    output["patch_mean_ndvi"] = [
                        float(np.nanmean(scene["ndvi"][row:row+32, col:col+32]))
                        for row, col in locations
                    ]
                    output["model_sha256"] = model_sha256
                    output["requested_start_utc"] = scene["requested_window"]["start"]
                    output["requested_end_utc"] = scene["requested_window"]["end"]
                    st.caption("NDVI is a descriptive baseline, not independent ground truth.")
                    metadata = {
                        "model_sha256": model_sha256,
                        "source": "Copernicus Sentinel-2 Level-2A Process API",
                        "bbox_wgs84": [77.578, 12.959, 77.606, 12.986],
                        "request_window_utc": scene["requested_window"],
                        "image_size": [320, 320],
                        "patch_size": 32,
                        "patch_count": len(patches),
                        "channels": ["B04", "B08"],
                        "threshold": 0.5,
                        "preprocessing": "Per-scene 2nd/98th percentile scaling on valid pixels",
                        "acquisition_ids": None,
                        "warning": "Mosaicking possible; no independent reference labels.",
                    }
                    st.download_button("Download experiment metadata JSON",
                        json.dumps(metadata, indent=2).encode("utf-8"),
                        "copernicus_cnn_experiment_metadata.json", "application/json")
                    st.dataframe(output, hide_index=True)
                    st.download_button("Download CNN predictions CSV", output.to_csv(index=False).encode(), "copernicus_cnn_predictions.csv", "text/csv")
                except ImportError as exc:
                    st.error("Keras inference dependency unavailable: " + str(exc))
                except Exception as exc:
                    st.error(f"CNN inference failed: {type(exc).__name__}: {exc}")

            st.caption("The selected time window may mosaic acquisitions. This is a connectivity check, not a publication-ready scene record.")
        except Exception as exc:
            st.error(f"Satellite request failed: {type(exc).__name__}: {exc}")
            st.caption("Do not paste API credentials into issue reports or screenshots.")
