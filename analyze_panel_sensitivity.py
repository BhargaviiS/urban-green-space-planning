"""Compare changing and fixed administrative-unit panels in satellite NDVI time series."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent

def analyze(root=ROOT):
    df = pd.read_csv(root / "data/ward_timeseries.csv")
    city = pd.read_csv(root / "data/city_timeseries.csv")
    usable = df.loc[(df.valid_fraction >= .5) & (df.valid_px >= 30)].copy()
    if usable.duplicated(["scene_id", "ward_id"]).any():
        raise ValueError("Duplicate scene-unit keys")
    if set(usable.scene_id) - set(city.scene_id):
        raise ValueError("Unexpected scene IDs")
    n_scenes = len(set(city.scene_id))
    unit_counts = usable.groupby("ward_id").scene_id.nunique()
    fixed_ids = set(unit_counts[unit_counts == n_scenes].index)
    usable["fixed"] = usable.ward_id.isin(fixed_ids)
    rows = []
    for scene_id, part in usable.groupby("scene_id"):
        fixed = part[part.fixed]
        rows.append({
            "scene_id": scene_id, "date": part.date.iloc[0],
            "usable_units": len(part), "fixed_units": len(fixed),
            "variable_panel_mean_ndvi": part.mean_ndvi.mean(),
            "fixed_panel_mean_ndvi": fixed.mean_ndvi.mean() if len(fixed) else None
        })
    table = pd.DataFrame(rows).sort_values("date")
    table["difference"] = table.variable_panel_mean_ndvi - table.fixed_panel_mean_ndvi
    out = root / "research_outputs"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "panel_sensitivity.csv", index=False)
    summary = {
        "scenes": n_scenes, "units_usable_in_any_scene": int(len(unit_counts)),
        "units_usable_in_all_scenes": len(fixed_ids),
        "median_absolute_ndvi_difference": float(table.difference.abs().median()),
        "maximum_absolute_ndvi_difference": float(table.difference.abs().max()),
        "caveat": "Descriptive geographic sample-composition sensitivity, not vegetation change."
    }
    (out / "panel_sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
