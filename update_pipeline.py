#!/usr/bin/env python3
"""
Live data pipeline for the Bengaluru Green Space dashboard.

What it does on every run
  1. Asks the public Earth Search STAC API (Sentinel-2 L2A on AWS) for scenes over Bengaluru.
  2. Skips scenes it has already processed (or already rejected as too cloudy).
  3. Reads ONLY the Bengaluru window of the Red, NIR and Scene-Classification bands
     straight from the cloud-optimised GeoTIFFs. No 1 GB downloads, no login.
  4. Converts digital numbers to surface reflectance (applies the L2A offset),
     masks cloud / shadow / cirrus, and computes NDVI.
  5. Aggregates NDVI per ward and for the whole city, appends to data/*.csv.
  6. Refreshes data/latest_ndvi.png, data/ndvi_change.png and data/status.json.

Usage
  python update_pipeline.py --dry-run                      # just list candidate scenes
  python update_pipeline.py --months 12 --max-scenes 40    # first run: backfill a year
  python update_pipeline.py                                # routine run (last 3 months)

Needs data/wards.geojson first (run build_wards.py once).
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
STATE_DIR = DATA_DIR / "_state"
WARDS_FILE = DATA_DIR / "wards.geojson"
WARD_TS = DATA_DIR / "ward_timeseries.csv"
CITY_TS = DATA_DIR / "city_timeseries.csv"
STATUS_FILE = DATA_DIR / "status.json"
GRID_META = STATE_DIR / "grid_meta.json"
STATE_FILE = STATE_DIR / "state.json"
SKIPPED_FILE = STATE_DIR / "skipped_scenes.json"

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
CITY_CENTRE = (77.5946, 12.9716)
CITY_BBOX = (77.4599, 12.8335, 77.7841, 13.1426)  # lon/lat extent of Bengaluru
BBOX_MARGIN = 0.01  # degrees of padding around the city
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
# SCL classes treated as unusable: no data, saturated, cloud shadow, cloud (medium/high), cirrus
BAD_SCL = (0, 1, 3, 8, 9, 10)
BLOCK = 10  # pixels per preview cell (10 m -> 100 m)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- small helpers
def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- STAC search
def search_scenes(months: int, max_cloud: float) -> list[dict]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(months * 30.5))
    body = {
        "collections": [COLLECTION],
        "intersects": {"type": "Point", "coordinates": list(CITY_CENTRE)},
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": 200,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    items: list[dict] = []
    for _ in range(10):  # follow pagination, hard stop at 10 pages
        resp = requests.post(STAC_URL, json=body, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items.extend(payload.get("features", []))
        nxt = next((l for l in payload.get("links", []) if l.get("rel") == "next"), None)
        if not nxt or nxt.get("body") is None:
            break
        body = {**body, **nxt["body"]} if nxt.get("merge") else nxt["body"]
    return items


def item_date(item: dict) -> str:
    return item["properties"]["datetime"][:10]


def band_params(item: dict, key: str) -> tuple[float, float]:
    """(scale, offset) that turn stored digital numbers into surface reflectance."""
    asset = item["assets"][key]
    rb = asset.get("raster:bands") or []
    if rb and ("scale" in rb[0] or "offset" in rb[0]):
        return float(rb[0].get("scale", 1e-4)), float(rb[0].get("offset", 0.0))
    try:
        baseline = float(item["properties"].get("s2:processing_baseline", 0))
    except (TypeError, ValueError):
        baseline = 0.0
    # Processing baseline >= 04.00 (Jan 2022 onwards) adds BOA_ADD_OFFSET = -1000 DN.
    return 1e-4, (-0.1 if baseline >= 4.0 else 0.0)


# --------------------------------------------------------------------------- NDVI maths
def to_reflectance(dn: np.ndarray, scale: float, offset: float):
    valid = dn > 0  # DN 0 = no data
    refl = dn.astype(np.float32) * scale + offset
    refl = np.clip(refl, 0.0, None)
    return refl, valid


def compute_ndvi(red_dn, nir_dn, red_params, nir_params, scl=None) -> np.ndarray:
    red, vr = to_reflectance(red_dn, *red_params)
    nir, vn = to_reflectance(nir_dn, *nir_params)
    valid = vr & vn
    if scl is not None:
        valid &= ~np.isin(scl, BAD_SCL)
    denom = red + nir
    valid &= denom > 1e-4
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
    return ndvi


def block_mean(arr: np.ndarray, block: int = BLOCK) -> np.ndarray:
    h = (arr.shape[0] // block) * block
    w = (arr.shape[1] // block) * block
    a = arr[:h, :w].reshape(h // block, block, w // block, block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=(1, 3)).astype(np.float32)


def zonal_stats(ndvi: np.ndarray, labels: np.ndarray, n: int) -> pd.DataFrame:
    lab = labels.ravel()
    val = ndvi.ravel()
    valid = np.isfinite(val)
    lv, vv = lab[valid], val[valid]
    total = np.bincount(lab, minlength=n + 1)
    vcount = np.bincount(lv, minlength=n + 1)
    vsum = np.bincount(lv, weights=vv, minlength=n + 1)
    out = pd.DataFrame({"ward_id": np.arange(n + 1), "total_px": total, "valid_px": vcount})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mean_ndvi"] = vsum / vcount
        for t in THRESHOLDS:
            c = np.bincount(lv[vv >= t], minlength=n + 1)
            out[f"pct_ge_{int(round(t * 100))}"] = 100.0 * c / vcount
    return out[out["ward_id"] > 0].set_index("ward_id")


# --------------------------------------------------------------------------- raster reading
def city_window(src):
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    left, bottom, right, top = transform_bounds(
        "EPSG:4326", src.crs,
        CITY_BBOX[0] - BBOX_MARGIN, CITY_BBOX[1] - BBOX_MARGIN,
        CITY_BBOX[2] + BBOX_MARGIN, CITY_BBOX[3] + BBOX_MARGIN,
    )
    w = from_bounds(left, bottom, right, top, transform=src.transform)
    col0 = max(int(math.floor(w.col_off / BLOCK) * BLOCK), 0)
    row0 = max(int(math.floor(w.row_off / BLOCK) * BLOCK), 0)
    width = int(math.ceil((w.col_off + w.width - col0) / BLOCK) * BLOCK)
    height = int(math.ceil((w.row_off + w.height - row0) / BLOCK) * BLOCK)
    width = min(width, ((src.width - col0) // BLOCK) * BLOCK)
    height = min(height, ((src.height - row0) // BLOCK) * BLOCK)
    return Window(col0, row0, width, height)


def read_ndvi(item: dict):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import bounds as win_bounds
    from rasterio.windows import from_bounds

    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
               GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2")
    with rasterio.Env(**env):
        with rasterio.open(item["assets"]["red"]["href"]) as red_src, \
                rasterio.open(item["assets"]["nir"]["href"]) as nir_src:
            window = city_window(red_src)
            red_dn = red_src.read(1, window=window)
            nir_dn = nir_src.read(1, window=window)
            crs = red_src.crs
            transform = red_src.window_transform(window)
            bounds = win_bounds(window, red_src.transform)
        scl = None
        if "scl" in item["assets"]:
            with rasterio.open(item["assets"]["scl"]["href"]) as scl_src:
                sw = from_bounds(*bounds, transform=scl_src.transform)
                scl = scl_src.read(1, window=sw, out_shape=red_dn.shape,
                                   resampling=Resampling.nearest, boundless=True, fill_value=0)
    ndvi = compute_ndvi(red_dn, nir_dn, band_params(item, "red"), band_params(item, "nir"), scl)
    return ndvi, transform, crs


# --------------------------------------------------------------------------- one scene
def process_scene(item: dict, wards, cache: dict, min_valid: float):
    from rasterio.features import rasterize

    ndvi, transform, crs = read_ndvi(item)
    sig = {"crs": crs.to_string(), "transform": [round(v, 6) for v in tuple(transform)[:6]],
           "shape": list(ndvi.shape)}
    meta = load_json(GRID_META)
    if meta is None:
        save_json(GRID_META, sig)
    elif meta != sig:
        return None, "different pixel grid (another tile / orbit window)"

    if "labels" not in cache:
        w = wards.to_crs(crs)
        shapes = [(g, int(i)) for g, i in zip(w.geometry, w["ward_id"]) if g is not None and not g.is_empty]
        cache["labels"] = rasterize(shapes, out_shape=ndvi.shape, transform=transform, fill=0, dtype="int32")
        cache["n"] = int(w["ward_id"].max())
        cache["names"] = dict(zip(w["ward_id"].astype(int), w["name"]))
    labels, n = cache["labels"], cache["n"]

    city = zonal_stats(ndvi, (labels > 0).astype(np.int32), 1).loc[1]
    city_valid = city["valid_px"] / max(city["total_px"], 1)
    if city_valid < min_valid:
        return None, f"only {city_valid:.0%} of the city is usable (clouds / no-data)"

    props = item["properties"]
    date = item_date(item)
    cloud = float(props.get("eo:cloud_cover", float("nan")))

    ward_df = zonal_stats(ndvi, labels, n)
    ward_df = ward_df[ward_df["total_px"] > 0].copy()
    ward_df["valid_fraction"] = ward_df["valid_px"] / ward_df["total_px"]
    ward_df = ward_df.reset_index()
    ward_df.insert(0, "date", date)
    ward_df.insert(1, "scene_id", item["id"])
    ward_df.insert(2, "cloud_cover", cloud)
    ward_df.insert(5, "name", ward_df["ward_id"].map(cache["names"]))

    city_row = {"date": date, "scene_id": item["id"], "cloud_cover": cloud,
                "valid_fraction": float(city_valid), "mean_ndvi": float(city["mean_ndvi"])}
    for t in THRESHOLDS:
        city_row[f"pct_ge_{int(round(t * 100))}"] = float(city[f"pct_ge_{int(round(t * 100))}"])
    return {"wards": ward_df, "city": city_row, "block": block_mean(ndvi)}, None


# --------------------------------------------------------------------------- outputs
def append_csv(path: Path, new_df: pd.DataFrame, keys: list[str]) -> None:
    df = pd.concat([pd.read_csv(path), new_df], ignore_index=True) if path.exists() else new_df
    df = df.drop_duplicates(subset=keys, keep="last").sort_values(["date"] + [k for k in keys if k != "date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def update_preview_state(date: str, block: np.ndarray) -> bool:
    state = load_json(STATE_FILE) or {}
    latest = state.get("latest_date")
    if latest and date <= latest:
        return False
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cur, prev = STATE_DIR / "ndvi_latest.npy", STATE_DIR / "ndvi_previous.npy"
    if latest and cur.exists():
        cur.replace(prev)
        state["previous_date"] = latest
    np.save(cur, block.astype(np.float16))
    state["latest_date"] = date
    save_json(STATE_FILE, state)
    return True


def render_previews() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = load_json(STATE_FILE)
    cur = STATE_DIR / "ndvi_latest.npy"
    if not state or not cur.exists():
        return
    latest = np.load(cur).astype(np.float32)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(latest, cmap="RdYlGn", vmin=-0.1, vmax=0.8)
    ax.set_title(f"NDVI over Bengaluru, scene of {state['latest_date']}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI (surface reflectance, cloud-masked)")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "latest_ndvi.png", dpi=130)
    plt.close(fig)

    prev_file = STATE_DIR / "ndvi_previous.npy"
    if prev_file.exists() and state.get("previous_date"):
        prev = np.load(prev_file).astype(np.float32)
        if prev.shape == latest.shape:
            fig, ax = plt.subplots(figsize=(7, 6.5))
            im = ax.imshow(latest - prev, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
            ax.set_title(f"Change in NDVI: {state['previous_date']} to {state['latest_date']}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI change (green = gain, red = loss)")
            fig.tight_layout()
            fig.savefig(DATA_DIR / "ndvi_change.png", dpi=130)
            plt.close(fig)


def write_status(items_seen: int) -> None:
    status = {"last_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "candidate_scenes_seen": items_seen,
              "source": "Sentinel-2 L2A via Earth Search (AWS public COGs)",
              "ndvi_method": "surface reflectance (L2A offset applied), SCL cloud/shadow/cirrus mask"}
    if CITY_TS.exists():
        city = pd.read_csv(CITY_TS).sort_values("date")
        last = city.iloc[-1]
        status.update({"scenes_in_history": int(len(city)),
                       "latest_scene_date": str(last["date"]),
                       "latest_scene_id": str(last["scene_id"]),
                       "latest_cloud_cover": float(last["cloud_cover"]),
                       "latest_valid_fraction": float(last["valid_fraction"])})
    save_json(STATUS_FILE, status)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=3, help="how far back to look for scenes")
    ap.add_argument("--max-cloud", type=float, default=30.0, help="scene-level cloud cover limit (%%)")
    ap.add_argument("--max-scenes", type=int, default=12, help="most scenes to process in one run")
    ap.add_argument("--per-month", type=int, default=0, help="thin to the N clearest scenes per month (0 = keep all)")
    ap.add_argument("--min-valid", type=float, default=0.5, help="min share of city pixels that must be usable")
    ap.add_argument("--dry-run", action="store_true", help="list candidate scenes and stop")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Searching Sentinel-2 L2A scenes over Bengaluru (last {args.months} months, cloud < {args.max_cloud}%)...")
    items = search_scenes(args.months, args.max_cloud)
    log(f"Found {len(items)} candidate scenes.")
    if args.dry_run:
        for it in items[:40]:
            p = it["properties"]
            log(f"  {item_date(it)}  cloud {p.get('eo:cloud_cover', float('nan')):5.1f}%  {it['id']}")
        return 0

    done = set(pd.read_csv(CITY_TS)["scene_id"]) if CITY_TS.exists() else set()
    skipped = set(load_json(SKIPPED_FILE) or [])
    todo = [it for it in items if it["id"] not in done and it["id"] not in skipped]

    if args.per_month > 0:
        by_month: dict[str, list[dict]] = {}
        for it in todo:
            by_month.setdefault(item_date(it)[:7], []).append(it)
        todo = [it for grp in by_month.values()
                for it in sorted(grp, key=lambda x: x["properties"].get("eo:cloud_cover", 100))[:args.per_month]]

    todo.sort(key=item_date, reverse=True)
    todo = todo[:args.max_scenes]
    todo.sort(key=item_date)  # process oldest -> newest so "latest / previous" stay in order
    log(f"{len(todo)} new scene(s) to process.")

    if todo and not WARDS_FILE.exists():
        log(f"ERROR: {WARDS_FILE} not found. Run build_wards.py first.")
        return 1

    if todo:
        import geopandas as gpd
        wards = gpd.read_file(WARDS_FILE)
    cache: dict = {}
    processed = 0
    for it in todo:
        sid = it["id"]
        log(f"Processing {item_date(it)}  {sid} ...")
        try:
            result, reason = process_scene(it, wards, cache, args.min_valid)
        except Exception as exc:  # network hiccup etc. -> try again next run
            log(f"  failed, will retry next run: {exc}")
            continue
        if result is None:
            log(f"  skipped: {reason}")
            skipped.add(sid)
            save_json(SKIPPED_FILE, sorted(skipped))
            continue
        append_csv(WARD_TS, result["wards"], ["scene_id", "ward_id"])
        append_csv(CITY_TS, pd.DataFrame([result["city"]]), ["scene_id"])
        update_preview_state(item_date(it), result["block"])
        c = result["city"]
        log(f"  ok: city mean NDVI {c['mean_ndvi']:.3f}, {c['pct_ge_30']:.1f}% of pixels >= 0.3, "
            f"{c['valid_fraction']:.0%} usable")
        processed += 1

    render_previews()
    write_status(len(items))
    log(f"Done. {processed} scene(s) added. Outputs are in {DATA_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
