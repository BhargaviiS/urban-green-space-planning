"""Export an auditable snapshot of the repository's archived research data."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES = ["all_candidates_scored.csv", "top_10_recommended_sites.csv",
         "ward_level_summary_reliable.csv", "data/city_timeseries.csv",
         "data/ward_timeseries.csv", "data/status.json"]

def main():
    candidates = pd.read_csv(ROOT / FILES[0])
    sites = pd.read_csv(ROOT / FILES[1])
    units = pd.read_csv(ROOT / FILES[2])
    scenes = pd.read_csv(ROOT / FILES[3])
    status = json.loads((ROOT / FILES[5]).read_text(encoding="utf-8"))
    assert len(scenes) == status["scenes_in_history"]
    assert scenes["date"].max() == status["latest_scene_date"]
    assert candidates["suitability"].between(0, 1).all()
    output = ROOT / "research_outputs"
    output.mkdir(exist_ok=True)
    manifest = {
        "snapshot_type": "archived; not real-time",
        "candidate_points": int(len(candidates)),
        "top_sites": int(len(sites)),
        "administrative_units": int(len(units)),
        "satellite_scenes": int(len(scenes)),
        "first_scene": str(scenes["date"].min()),
        "last_scene": str(scenes["date"].max()),
        "pipeline_last_run_utc": status["last_run_utc"],
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in FILES},
        "caveats": [
            "Park distance is a screening indicator, not land suitability.",
            "Euclidean access is not pedestrian-network access.",
            "OSM administrative units are not official BBMP wards.",
            "NDVI pseudo-label agreement is not independent CNN accuracy."
        ]
    }
    (output / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    scenes.to_csv(output / "city_scene_evidence.csv", index=False)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
