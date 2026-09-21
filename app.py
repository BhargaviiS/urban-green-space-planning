import streamlit as st
import pandas as pd
import folium
import pydeck as pdk
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

@st.cache_data
def load_top_sites():
    return pd.read_csv("top_10_recommended_sites.csv")

@st.cache_data
def load_all_candidates():
    return pd.read_csv("all_candidates_scored.csv")

@st.cache_data
def load_ward_data():
    return pd.read_csv("ward_level_summary_reliable.csv")

top_sites = load_top_sites()
all_candidates = load_all_candidates()
ward_data = load_ward_data()

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🗺️ Parks & Boundary",
    "🚶 Accessibility",
    "🛰️ NDVI & Validation",
    "📍 3D Suitability Map",
    "🏘️ Ward Analysis",
    "🧠 CNN Image Classifier",
    "📈 Impact Evaluation",
    "🏛️ For Policymakers",
])

with tab1:
    st.header("City Boundary and OpenStreetMap Parks")
    st.image("bengaluru_parks.png", use_container_width=True)
    st.markdown(
        '<a class="source-link" href="https://www.openstreetmap.org" target="_blank">📡 Data: OpenStreetMap</a>',
        unsafe_allow_html=True,
    )

with tab2:
    st.header("500m Accessibility Coverage")
    st.image("accessibility_coverage.png", use_container_width=True)
    st.caption("Light green zones are within a comfortable walking distance (500m) of an existing park.")

with tab3:
    st.header("Satellite-Derived NDVI")
    st.image("ndvi_bengaluru.png", use_container_width=True)
    st.header("Cross-Validated Green Space")
    st.image("merged_green_space.png", use_container_width=True)
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
    st.dataframe(top_sites, use_container_width=True)

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
        "Green space suitability aggregated across 424 real Bengaluru ward boundaries "
        "(sourced from OpenStreetMap), filtered to only wards with at least 10 sample "
        "points for statistical reliability."
    )
    st.image("ward_choropleth.png", use_container_width=True)

    st.subheader("Top 10 Most Underserved Wards")
    st.dataframe(
        ward_data.head(10).rename(columns={
            "name": "Ward",
            "candidate_points": "Sample Points",
            "avg_suitability": "Avg. Priority Score",
            "max_suitability": "Max Priority Score",
        }),
        use_container_width=True,
    )
    st.caption(
        f"{len(ward_data)} wards met the minimum sample size threshold out of all wards "
        "intersecting the analyzed satellite tile. Wards with fewer than 10 sample points "
        "were excluded to avoid unreliable rankings from small samples."
    )

with tab6:
    st.header("CNN Image Classifier: Green Space Detection")
    st.write(
        "A Convolutional Neural Network trained directly on raw satellite imagery patches "
        "(Red and Near-Infrared bands only — no NDVI given to the model) to classify "
        "32×32 pixel patches (320m × 320m on the ground) as green space or not."
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Training Patches", "46,369")
    mc2.metric("Test Patches", "11,593")
    mc3.metric("Test Accuracy", "99.98%")

    st.subheader("Training Curves")
    st.image("cnn_training_history.png", use_container_width=True)
    st.caption("Training and validation accuracy/loss across epochs, with early stopping to prevent overfitting.")

    st.subheader("Sample Predictions")
    st.image("cnn_sample_predictions.png", use_container_width=True)
    st.caption("Top row: true green-space patches. Bottom row: true non-green patches. Labels show the model's prediction for each.")

    st.info(
        "**Methodology note:** Patches were labeled using the top and bottom 25% of this "
        "tile's own NDVI distribution to guarantee balanced, unambiguous classes. The model "
        "was deliberately trained on raw Red/NIR bands only (not NDVI) to ensure it learns "
        "genuine visual/spectral patterns rather than reading a pre-computed answer."
    )

with tab7:
    st.header("Impact Evaluation: Park Proximity vs Vegetation Health")
    st.image("impact_evaluation.png", use_container_width=True)
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
    - **Two independently trained ML models** — a tabular Random Forest for suitability scoring,
      and a CNN achieving 99.98% accuracy for image-based green space classification
    - **Ward-level granularity** — {len(ward_data)} statistically reliable wards ranked by priority,
      with **{ward_data.iloc[0]['name']}** identified as the highest-priority ward
    - **Objective site prioritization** — removes guesswork from where new parks go
    - **A repeatable pipeline** — re-run annually as new satellite data becomes available, to track
      whether green cover is genuinely improving, ward by ward
    - **Built entirely on free, open data sources** — Sentinel-2 (ESA), OpenStreetMap — meaning
      zero licensing cost to scale city-wide

    **Current status:** working proof-of-concept covering a real Bengaluru satellite tile with
    ward-level granularity and two trained ML models. **Next steps to production:** full-city tile
    mosaicking, official BBMP ward-boundary integration, and a longitudinal impact-tracking module.
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