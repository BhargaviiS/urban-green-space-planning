"""Audit geographic proximity in random versus contiguous validation splits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parent

def audit(root=ROOT, seed=2026):
    df = pd.read_csv(root / "all_candidates_scored.csv")
    if df[["latitude", "longitude"]].isna().any().any() or df[["latitude", "longitude"]].duplicated().any():
        raise ValueError("Invalid or duplicate coordinates")
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n = round(.2 * len(df))
    random_test, random_train = order[:n], order[n:]
    east_order = np.argsort(df.longitude.to_numpy(), kind="stable")
    east_train, east_test = east_order[:-n], east_order[-n:]
    def stats(train, test):
        dist, _ = BallTree(coords[train], metric="haversine").query(coords[test], k=1)
        km = dist[:, 0] * 6371.0088
        return {"train": len(train), "test": len(test), "median_nearest_train_km": float(np.median(km)),
                "fraction_within_1km": float(np.mean(km < 1)), "fraction_within_2km": float(np.mean(km < 2))}
    result = {"method": "geometry-only split proximity diagnostic", "seed": seed, "points": len(df),
              "random": stats(random_train, random_test), "eastern_contiguous": stats(east_train, east_test),
              "caution": "Not an ML validation score; only one spatial split and no independent labels."}
    out = root / "research_outputs"
    out.mkdir(exist_ok=True)
    (out / "spatial_leakage_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
