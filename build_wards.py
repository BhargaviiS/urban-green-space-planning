#!/usr/bin/env python3
"""
Build data/wards.geojson once (re-run occasionally to refresh boundaries).

Pulls OpenStreetMap administrative polygons for Bengaluru, drops anything bigger than
20 km2 (district / state shapes that OSM mixes into the same query), gives every unit a
numeric ward_id and a unique name, and saves them in lon/lat (EPSG:4326).

Note: these are OSM admin_level 8-10 units. They are NOT the official BBMP ward set,
so call them "administrative units / localities" in anything formal.
"""
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

PLACE = "Bengaluru, Karnataka, India"
CITY_BBOX = (77.4599, 12.8335, 77.7841, 13.1426)
MAX_AREA_KM2 = 20
OUT = Path("data/wards.geojson")

found = None
for level in ["8", "9", "10"]:
    try:
        feats = ox.features_from_place(PLACE, {"boundary": "administrative", "admin_level": level})
    except Exception as exc:
        print(f"admin_level={level}: {exc}")
        continue
    feats = feats[feats.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(feats) > 5:
        found = feats
        print(f"Using admin_level={level}: {len(feats)} polygons")
        break

if found is None:
    raise SystemExit("No ward-like boundaries found in OpenStreetMap.")

gdf = found.reset_index(drop=True)
gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

# size filter in a metric projection (UTM 43N)
area_km2 = gdf.to_crs(32643).geometry.area / 1e6
gdf = gdf[area_km2 < MAX_AREA_KM2].copy()

# keep only units that touch the city extent
gdf = gdf[gdf.intersects(box(*CITY_BBOX))].reset_index(drop=True)

name_col = next((c for c in ["name", "name:en", "official_name"] if c in gdf.columns), None)
gdf["ward_id"] = range(1, len(gdf) + 1)
gdf["name"] = gdf[name_col].fillna("").astype(str) if name_col else ""
gdf.loc[gdf["name"].str.strip() == "", "name"] = "Unnamed unit"
dup = gdf["name"].duplicated(keep=False)
gdf.loc[dup, "name"] = gdf.loc[dup, "name"] + " (" + gdf.loc[dup, "ward_id"].astype(str) + ")"

OUT.parent.mkdir(exist_ok=True)
gdf[["ward_id", "name", "geometry"]].to_file(OUT, driver="GeoJSON")
print(f"Saved {len(gdf)} administrative units to {OUT}")
