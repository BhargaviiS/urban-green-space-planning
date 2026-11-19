"""Reproduce monthly scene coverage; do not infer an annual vegetation trend."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent

def analyze(root=ROOT):
    df = pd.read_csv(root / "data/city_timeseries.csv", parse_dates=["date"])
    required = {"date", "scene_id", "mean_ndvi", "pct_ge_30", "valid_fraction"}
    if not required.issubset(df.columns) or df.empty or df.scene_id.duplicated().any():
        raise ValueError("Invalid scene-level input table")
    months = pd.period_range(df.date.min().to_period("M"), df.date.max().to_period("M"))
    monthly = df.assign(month=df.date.dt.to_period("M")).groupby("month").agg(
        scenes=("scene_id", "size"),
        mean_scene_ndvi=("mean_ndvi", "mean"),
        mean_scene_pct_ge_30=("pct_ge_30", "mean"),
        mean_valid_fraction=("valid_fraction", "mean")).reindex(months)
    monthly["scenes"] = monthly.scenes.fillna(0).astype(int)
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    out = root / "research_outputs"
    out.mkdir(exist_ok=True)
    monthly.to_csv(out / "monthly_scene_coverage.csv", float_format="%.6f")
    summary = {
        "scenes": int(len(df)),
        "first_scene": df.date.min().strftime("%Y-%m-%d"),
        "last_scene": df.date.max().strftime("%Y-%m-%d"),
        "months_in_window": len(months),
        "months_without_scenes": monthly.index[monthly.scenes.eq(0)].tolist(),
        "min_scene_mean_ndvi": round(float(df.mean_ndvi.min()), 6),
        "max_scene_mean_ndvi": round(float(df.mean_ndvi.max()), 6),
        "min_scene_pct_ge_30": round(float(df.pct_ge_30.min()), 6),
        "max_scene_pct_ge_30": round(float(df.pct_ge_30.max()), 6),
        "caution": "Descriptive scene statistics only: seasonality, missing months, and cloud masks preclude annual trend claims."
    }
    (out / "satellite_coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
