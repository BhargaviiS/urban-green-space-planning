import json
from datetime import datetime, timezone
from pathlib import Path

import folium
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Bengaluru Green Space Platform", layout="wide", page_icon="🌳")

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #66bb6a 100%);
    padding: 2.2rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.95; }
div[data-testid="stMetric"] {
    background: #f1f8f2;
    border: 1px solid #c8e6c9;
    border-radius: 12px;
    padding: 1rem;
}
.source-link {
    display: inline-block;
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px 6px 3px 0;
    text-decoration: none;
    color: #1b5e20;
    font-size: 0.85rem;
}
.problem-box {
    background: #fff8e1;
    border-left: 5px solid #f9a825;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

DATA = Path("data")


# ----------------------------------------------------------------------------- loaders
@st.cache_data
def load_top_sites():
    return pd.read_csv("top_10_recommended_sites.csv")


@st.cache_data
def load_all_candidates():
    return pd.read_csv("all_candidates_scored.csv")


@st.cache_data
def load_ward_data():
    return pd.read_csv("ward_level_summary_reliable.csv")


@st.cache_data(ttl=900)  # re-read the live files at most every 15 minutes
def load_live():
    status_f, city_f, ward_f = DATA / "status.json", DATA / "city_timeseries.csv", DATA / "ward_timeseries.csv"
    if not (status_f.exists() and city_f.exists() and ward_f.exists()):
        return None
    status = json.loads(status_f.read_text(encoding="utf-8"))
    city = pd.read_csv(city_f, parse_dates=["date"]).sort_values("date")
    wards = pd.read_csv(ward_f, parse_dates=["date"]).sort_values("date")
    return status, city, wards


top_sites = load_top_sites()
all_candidates = load_all_candidates()
ward_data = load_ward_data()


# ----------------------------------------------------------------------------- live tab
def render_trends_tab():
    st.header("Satellite trends: how Bengaluru's vegetation is changing")
    live = load_live()
    if live is None:
        st.info(
            "Satellite trend data has not been generated yet. Run `python build_wards.py` once, then "
            "`python update_pipeline.py --months 12 --max-scenes 40`, and commit the `data/` folder."
        )
        return
    status, city, wards = live

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Newest satellite scene", status.get("latest_scene_date", "n/a"))
    c2.metric("Scene cloud cover", f"{status.get('latest_cloud_cover', float('nan')):.0f}%")
    c3.metric("Scenes in history", status.get("scenes_in_history", 0))
    c4.metric("Last checked (UTC)", str(status.get("last_run_utc", "n/a"))[:16].replace("T", " "))
    st.caption(
        "Data refreshes automatically: a scheduled job looks for new Sentinel-2 passes over Bengaluru, "
        "masks clouds, recomputes ward-level vegetation, and republishes this page. Sentinel-2 revisits "
        "the city every few days, but cloudy scenes (common in the monsoon) are rejected."
    )

    threshold = st.select_slider("What NDVI value counts as 'green'?", options=[0.2, 0.3, 0.4, 0.5], value=0.3)
    col = f"pct_ge_{int(round(threshold * 100))}"
    label = f"% of pixels with NDVI ≥ {threshold}"

    st.subheader("City-wide green cover over time")
    st.line_chart(city.set_index("date")[[col]].rename(columns={col: label}))
    st.caption("Each point is one cloud-screened satellite scene. Seasonal swings (wet vs dry season) are "
               "normal; compare the same season across years to judge real change.")

    usable = wards[(wards["valid_fraction"] >= 0.5) & (wards["valid_px"] >= 30)]
    if usable.empty:
        st.warning("No wards have enough cloud-free pixels yet.")
    else:
        st.subheader("Compare wards")
        counts = usable.groupby("name")["scene_id"].nunique().sort_values(ascending=False)
        chosen = st.multiselect("Choose administrative units", sorted(counts.index), default=list(counts.index[:3]))
        if chosen:
            pivot = usable[usable["name"].isin(chosen)].pivot_table(index="date", columns="name", values=col)
            st.line_chart(pivot)

        st.subheader("What changed between two dates?")
        dates = sorted(usable["date"].dt.strftime("%Y-%m-%d").unique())
        if len(dates) >= 2:
            d1, d2 = st.columns(2)
            base = d1.selectbox("Baseline date", dates, index=len(dates) - 2)
            comp = d2.selectbox("Comparison date", dates, index=len(dates) - 1)
            a = usable[usable["date"] == base][["ward_id", "name", col]].rename(columns={col: "baseline"})
            b = usable[usable["date"] == comp][["ward_id", col]].rename(columns={col: "comparison"})
            diff = a.merge(b, on="ward_id")
            diff["change (pct. points)"] = diff["comparison"] - diff["baseline"]
            if diff.empty:
                st.info("No ward has usable data on both dates.")
            else:
                g, l = st.columns(2)
                cols = ["name", "baseline", "comparison", "change (pct. points)"]
                g.markdown("**Biggest gains**")
                g.dataframe(diff.nlargest(10, "change (pct. points)")[cols].round(1), width="stretch", hide_index=True)
                l.markdown("**Biggest losses**")
                l.dataframe(diff.nsmallest(10, "change (pct. points)")[cols].round(1), width="stretch", hide_index=True)
            if base[:4] != comp[:4] or abs(int(base[5:7]) - int(comp[5:7])) not in (0, 1, 11):
                st.caption("These two dates are in different seasons, so part of the difference is the normal "
                           "seasonal cycle and not necessarily land-cover change.")
        else:
            st.info("Only one usable date so far; change tables appear once a second scene is processed.")

    m1, m2 = st.columns(2)
    if (DATA / "latest_ndvi.png").exists():
        m1.image(str(DATA / "latest_ndvi.png"), caption="Latest cloud-masked NDVI", width="stretch")
    if (DATA / "ndvi_change.png").exists():
        m2.image(str(DATA / "ndvi_change.png"), caption="NDVI change between the two newest scenes", width="stretch")

    st.download_button("Download ward time series (CSV)", wards.to_csv(index=False).encode("utf-8"),
                       file_name="ward_timeseries.csv", mime="text/csv")
    st.caption(
        "Method: Sentinel-2 L2A Red and NIR bands converted to surface reflectance (the L2A offset is applied), "
        "cloud, shadow and cirrus pixels removed using the scene classification layer, NDVI averaged per "
        "OpenStreetMap administrative unit. Units are OSM admin_level 8 areas, not the official BBMP ward set. "
        "The static analysis tabs were produced earlier and are being regenerated with this same NDVI method."
    )



# ----------------------------------------------------------------------------- live conditions (real time)
LIVE_POINTS = [(12.9716, 77.5946), (13.07, 77.59), (12.87, 77.59), (12.97, 77.72), (12.97, 77.47)]  # centre, N, S, E, W
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def _as_list(js):
    return js if isinstance(js, list) else [js]


@st.cache_data(ttl=600, show_spinner=False)  # one shared fetch every 10 minutes, whoever is watching
def fetch_live():
    common = {
        "latitude": ",".join(str(p[0]) for p in LIVE_POINTS),
        "longitude": ",".join(str(p[1]) for p in LIVE_POINTS),
        "timezone": "Asia/Kolkata", "past_days": 1, "forecast_days": 2,
    }
    w = requests.get(WEATHER_URL, timeout=20, params={
        **common,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m",
        "hourly": "temperature_2m,apparent_temperature"})
    w.raise_for_status()
    a = requests.get(AIR_URL, timeout=20, params={
        **common, "current": "pm2_5,pm10,us_aqi,uv_index", "hourly": "pm2_5,us_aqi"})
    a.raise_for_status()
    return {"weather": _as_list(w.json()), "air": _as_list(a.json()),
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}


def current_values(blocks, key):
    return [b["current"][key] for b in blocks if b.get("current") and b["current"].get(key) is not None]


def mean_or_none(vals):
    return sum(vals) / len(vals) if vals else None


def hourly_frame(blocks, keys):
    frames = []
    for b in blocks:
        h = b.get("hourly") or {}
        if "time" in h:
            frames.append(pd.DataFrame({"time": pd.to_datetime(h["time"]), **{k: h.get(k) for k in keys}}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).groupby("time").mean(numeric_only=True)


def past_and_forecast(df, now, cols, names):
    out = pd.DataFrame(index=df.index)
    for c, n in zip(cols, names):
        out[f"{n} (past 24 h)"] = df[c].where(df.index <= now)
        out[f"{n} (next 24 h)"] = df[c].where(df.index >= now)
    return out[(df.index >= now - pd.Timedelta(hours=24)) & (df.index <= now + pd.Timedelta(hours=24))]


def aqi_category(v):
    for limit, name in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
                        (200, "Unhealthy"), (300, "Very unhealthy")]:
        if v <= limit:
            return name
    return "Hazardous"


def fmt(v, unit="", digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}{unit}"


@st.fragment(run_every=600)  # re-checks every 10 minutes without reloading the page
def render_live_now():
    st.header("Live conditions across Bengaluru")
    if st.button("Refresh now"):
        fetch_live.clear()
        st.rerun(scope="fragment")
    try:
        live = fetch_live()
    except Exception as exc:
        st.warning(f"The live weather / air-quality service could not be reached just now ({exc}). "
                   "This tab retries automatically.")
        return
    wx, aq = live["weather"], live["air"]
    now = pd.to_datetime(wx[0]["current"]["time"])

    temp = mean_or_none(current_values(wx, "temperature_2m"))
    feels = mean_or_none(current_values(wx, "apparent_temperature"))
    hum = mean_or_none(current_values(wx, "relative_humidity_2m"))
    wind = mean_or_none(current_values(wx, "wind_speed_10m"))
    rain = mean_or_none(current_values(wx, "precipitation"))
    pm25 = mean_or_none(current_values(aq, "pm2_5"))
    aqi = mean_or_none(current_values(aq, "us_aqi"))
    uv = mean_or_none(current_values(aq, "uv_index"))

    r1 = st.columns(4)
    r1[0].metric("Temperature", fmt(temp, " °C"))
    r1[1].metric("Feels like", fmt(feels, " °C"))
    r1[2].metric("Humidity", fmt(hum, " %", 0))
    r1[3].metric("Rain (last hour)", fmt(rain, " mm"))
    r2 = st.columns(4)
    r2[0].metric("PM2.5", fmt(pm25, " µg/m³"))
    r2[1].metric("US AQI", fmt(aqi, "", 0), help=aqi_category(aqi) if aqi is not None else None)
    r2[2].metric("Wind", fmt(wind, " km/h"))
    r2[3].metric("UV index", fmt(uv))
    st.caption(f"Model time {now:%d %b %Y, %H:%M} IST · fetched {live['fetched_utc']} UTC · average of 5 sample "
               "points (centre, north, south, east, west). Refreshes itself every 10 minutes.")

    if aqi is not None:
        text = f"Air quality: US AQI {aqi:.0f} ({aqi_category(aqi)})."
        (st.success if aqi <= 50 else st.info if aqi <= 100 else st.warning)(text)
    if feels is not None and feels >= 35:
        st.warning(
            f"Heat-stress flag: it feels like {feels:.1f} °C. Heat events are when shade and tree cover matter most; "
            "the wards currently ranked highest for new green space are: "
            + ", ".join(ward_data.head(5)["name"]) + ". (Dashboard flag, not an official advisory.)")

    hw = hourly_frame(wx, ["temperature_2m", "apparent_temperature"])
    if not hw.empty:
        st.subheader("Temperature: last 24 hours and next 24 hours")
        st.line_chart(past_and_forecast(hw, now, ["temperature_2m", "apparent_temperature"], ["Temperature", "Feels like"]))
    ha = hourly_frame(aq, ["pm2_5"])
    if not ha.empty and ha["pm2_5"].notna().any():
        st.subheader("PM2.5: last 24 hours and next 24 hours")
        st.line_chart(past_and_forecast(ha, now, ["pm2_5"], ["PM2.5"]))

    with st.expander("Values at each sample point"):
        rows = []
        for (lat, lon), w, a in zip(LIVE_POINTS, wx, aq):
            cw, ca = w.get("current", {}), a.get("current", {})
            rows.append({"lat": lat, "lon": lon, "temp °C": cw.get("temperature_2m"),
                         "feels °C": cw.get("apparent_temperature"), "PM2.5": ca.get("pm2_5"), "US AQI": ca.get("us_aqi")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Source: Open-Meteo weather and air-quality models. These are model analyses on coarse grids "
        "(roughly 10 to 40 km), not street-level sensors, so they describe the city as a whole and cannot "
        "show neighbourhood heat islands. Satellite vegetation cannot be real-time: Sentinel-2 passes over "
        "Bengaluru every few days (see the Satellite Trends tab). Open-Meteo's free tier is for "
        "non-commercial use; a commercial deployment needs a paid plan or station feeds (e.g. CPCB / KSPCB, KSNDMC).")

# ----------------------------------------------------------------------------- page
st.markdown("""
<div class="hero">
<h1>🌳 Bengaluru Urban Green Space Intelligence Platform</h1>
<p>A satellite-verified, data-driven system for green space planning and impact evaluation —
built for Bengaluru, ready for city-scale deployment.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="problem-box">
<b>The problem:</b> Bengaluru's green cover is shrinking, and existing planning tools rely on
outdated, manually-updated records that don't reflect what's actually on the ground. Officials
lack a live, verifiable, ward-level way to see where green space is genuinely missing.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("OSM-Mapped Parks (verified)", "1,767")
c2.metric("Satellite Green Zones Found", "276")
c3.metric("Reliable Wards Analyzed", f"{len(ward_data)}")
c4.metric("Top Priority Ward", ward_data.iloc[0]["name"], help=f"Avg suitability: {ward_data.iloc[0]['avg_suitability']:.2f}")

st.markdown("---")

tabL, tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "⚡ Live Now",
    "🛰️ Satellite Trends",
    "🗺️ Parks & Boundary",
    "🚶 Accessibility",
    "🛰️ NDVI & Validation",
    "📍 3D Suitability Map",
    "🏘️ Ward Analysis",
    "🧠 CNN Image Classifier",
    "📈 Impact Evaluation",
    "🏛️ For Policymakers",
])

with tabL:
    render_live_now()

with tab0:
    render_trends_tab()

with tab1:
    st.header("City Boundary and OpenStreetMap Parks")
    st.image("bengaluru_parks.png", width="stretch")
    st.markdown(
        '<a class="source-link" href="https://www.openstreetmap.org" target="_blank">📡 Data: OpenStreetMap</a>',
        unsafe_allow_html=True,
    )

with tab2:
    st.header("500m Accessibility Coverage")
    st.image("accessibility_coverage.png", width="stretch")
    st.caption("Light green zones are within a comfortable walking distance (500m) of an existing park.")

with tab3:
    st.header("Satellite-Derived NDVI")
    st.image("ndvi_bengaluru.png", width="stretch")
    st.header("Cross-Validated Green Space")
    st.image("merged_green_space.png", width="stretch")
    st.markdown(
        '<a class="source-link" href="https://dataspace.copernicus.eu" target="_blank">🛰️ Data: Copernicus Sentinel-2</a>',
        unsafe_allow_html=True,
    )

with tab4:
    st.header("Interactive 3D Suitability Map")
    st.write("Taller, redder columns indicate higher-priority locations for new green space.")

    view_state = pdk.ViewState(
        latitude=all_candidates["latitude"].mean(),
        longitude=all_candidates["longitude"].mean(),
        zoom=11,
        pitch=45,
    )
    layer = pdk.Layer(
        "ColumnLayer",
        data=all_candidates,
        get_position=["longitude", "latitude"],
        get_elevation="suitability * 1500",
        elevation_scale=1,
        radius=50,
        get_fill_color=["color_r", "color_g", "color_b", 180],
        pickable=True,
        auto_highlight=True,
    )
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Suitability score: {suitability}"},
    ))

    st.subheader("Top 10 Recommended Sites")
    st.dataframe(top_sites, width="stretch")

    st.subheader("Site Locations")
    m = folium.Map(location=[top_sites["latitude"].mean(), top_sites["longitude"].mean()], zoom_start=11)
    for _, row in top_sites.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=f"Suitability: {row['suitability']:.2f}",
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(m)
    st_folium(m, width=1200, height=450)

with tab5:
    st.header("Ward-Level Priority Analysis")
    st.write(
        "Green space suitability aggregated across 424 OpenStreetMap administrative units, "
        "filtered to only units with at least 10 sample points for statistical reliability."
    )
    st.image("ward_choropleth.png", width="stretch")

    st.subheader("Top 10 Most Underserved Wards")
    st.dataframe(
        ward_data.head(10).rename(columns={
            "name": "Ward",
            "candidate_points": "Sample Points",
            "avg_suitability": "Avg. Priority Score",
            "max_suitability": "Max Priority Score",
        }),
        width="stretch",
    )
    st.caption(
        f"{len(ward_data)} units met the minimum sample size threshold out of all units "
        "intersecting the analyzed satellite tile. Units with fewer than 10 sample points "
        "were excluded to avoid unreliable rankings from small samples."
    )

with tab6:
    st.header("CNN Image Classifier: Green Space Detection")
    st.write(
        "A Convolutional Neural Network trained directly on raw satellite imagery patches "
        "(Red and Near-Infrared bands only — no NDVI given to the model) to classify "
        "32×32 pixel patches (320m × 320m on the ground) as green space or not."
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Training Patches", "46,369")
    mc2.metric("Test Patches (same date)", "11,593")
    mc3.metric("Test Accuracy (Aug 2026)", "99.97%")
    mc4.metric("Cross-season Accuracy (Jan 2026)", "98.29%")

    st.subheader("Training Curves")
    st.image("cnn_training_history.png", width="stretch")
    st.caption("Training and validation accuracy/loss across epochs, with early stopping to prevent overfitting.")

    st.subheader("Sample Predictions")
    st.image("cnn_sample_predictions.png", width="stretch")
    st.caption("Top row: true green-space patches. Bottom row: true non-green patches. Labels show the model's prediction for each.")

    st.info(
        "**Cross-season test:** the model trained on an August 2026 scene was applied, without retraining, to "
        "58,040 patches from a January 2026 scene: 991 non-green patches were called green, no green patch was "
        "missed. Labels in both cases come from each scene's own NDVI quartiles, so this shows consistency "
        "across seasons, not agreement with surveyed ground truth."
    )

with tab7:
    st.header("Impact Evaluation: Park Proximity vs Vegetation Health")
    st.image("impact_evaluation.png", width="stretch")
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Correlation (r)", "0.203")
    ic2.metric("R-squared", "0.041")
    ic3.metric("P-value", "< 0.001")
    st.info(
        "**Interpretation:** The relationship is statistically real, but formal park proximity "
        "alone explains only ~4% of vegetation variation — suggesting neighborhood greenery is "
        "driven by multiple factors, not park placement alone. A full longitudinal impact study "
        "(with health/temperature data) is the natural next phase."
    )

with tab8:
    st.header("Why This Matters for Bengaluru")
    st.markdown(f"""
    **For BBMP and urban planners, this platform offers:**

    - **Verified, satellite-checked data** — not reliant on outdated manual surveys
    - **Live conditions plus automatic satellite updates** — live weather and air quality refresh every
      10 minutes; new Sentinel-2 scenes are cloud-screened and turned into ward-level time series
    - **Two independently trained ML models** — a Random Forest for tabular suitability analysis and a CNN
      for image-based green space classification (98.29% accuracy on an unseen season)
    - **Ward-level granularity** — {len(ward_data)} statistically reliable units ranked by priority,
      with **{ward_data.iloc[0]['name']}** at the top of the current ranking
    - **Built entirely on free, open data sources** — Sentinel-2 (ESA), OpenStreetMap — meaning
      zero licensing cost to scale city-wide

    **Current status:** working proof-of-concept on a single Sentinel-2 tile that covers the whole city.
    **Next steps to production:** ground-truth validation, official BBMP boundary integration, and
    multi-year trend analysis.
    """)

st.sidebar.title("About This Platform")
st.sidebar.write(
    "Combines Sentinel-2 satellite imagery, OpenStreetMap data, and two trained machine "
    "learning models to support evidence-based, ward-level urban green space planning in Bengaluru."
)
st.sidebar.markdown("---")
st.sidebar.caption("Data sources")
st.sidebar.markdown("[Copernicus Data Space](https://dataspace.copernicus.eu)")
st.sidebar.markdown("[OpenStreetMap](https://www.openstreetmap.org)")
