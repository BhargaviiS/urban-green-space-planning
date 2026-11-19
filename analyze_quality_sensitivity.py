"""Descriptive cloud-quality and aggregation sensitivity of archived Sentinel-2 administrative-unit NDVI."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)

def analyze(root=ROOT):
    df = pd.read_csv(root / "data/ward_timeseries.csv")
    required = {"scene_id", "ward_id", "date", "valid_fraction", "valid_px", "mean_ndvi"}
    if df.empty or not required.issubset(df.columns) or df.duplicated(["scene_id", "ward_id"]).any():
        raise ValueError("Invalid scene-unit input")
    if not df.valid_fraction.between(0, 1).all() or (df.valid_px < 0).any():
        raise ValueError("Invalid coverage values")
    scenes = sorted(df.scene_id.unique())
    rows = []
    for t in THRESHOLDS:
        subset = df[(df.valid_fraction >= t) & (df.valid_px >= 30)]
        for scene in scenes:
            p = subset[subset.scene_id == scene]
            rows.append({"scene_id": scene, "minimum_valid_fraction": t,
                         "usable_units": len(p),
                         "unit_mean_ndvi": p.mean_ndvi.mean() if len(p) else None,
                         "pixel_weighted_unit_mean_ndvi": np.average(p.mean_ndvi, weights=p.valid_px) if p.valid_px.sum() else None})
    table = pd.DataFrame(rows)
    a = table[table.minimum_valid_fraction == .5].set_index("scene_id")
    b = table[table.minimum_valid_fraction == .9].set_index("scene_id")
    shift = (a.unit_mean_ndvi - b.unit_mean_ndvi).abs().dropna()
    weighted_shift = (a.unit_mean_ndvi - a.pixel_weighted_unit_mean_ndvi).abs().dropna()
    summary = {
        "scenes": len(scenes),
        "median_usable_units": {str(t): float(table[table.minimum_valid_fraction == t].usable_units.median()) for t in THRESHOLDS},
        "median_abs_threshold_shift": float(shift.median()),
        "max_abs_threshold_shift": float(shift.max()),
        "median_abs_weighting_shift": float(weighted_shift.median()),
        "max_abs_weighting_shift": float(weighted_shift.max()),
        "caution": "Descriptive only. No causal inference, annual trend, or confidence interval; overlapping OSM units may double-count pixels."
    }
    out = root / "research_outputs"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "quality_threshold_sensitivity.csv", index=False)
    (out / "quality_threshold_sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
