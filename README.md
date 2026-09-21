\# 🌳 Bengaluru Urban Green Space Planning \& Impact Evaluation



A satellite-verified, data-driven platform for urban green space planning and impact evaluation in Bengaluru — built as a final-year B.Tech Data Science project, designed for real-world municipal use.



\*\*🔗 Live Dashboard:\*\* \[bengaluru-green-space-planning.streamlit.app](https://bengaluru-green-space-planning.streamlit.app)



\## The Problem



Bengaluru's green cover is shrinking, and existing planning tools rely on outdated, manually-updated municipal records that don't reflect ground reality. Urban planners lack a live, verifiable, ward-level way to see where green space is genuinely missing.



\## What This Platform Does



\- \*\*Satellite vegetation detection\*\* — computes NDVI from Sentinel-2 imagery to detect real, current green cover

\- \*\*Cross-validation\*\* — compares satellite-detected vegetation against OpenStreetMap park data to catch what either source misses alone

\- \*\*Accessibility analysis\*\* — 500m walking-distance buffer coverage around existing parks

\- \*\*Suitability scoring\*\* — a 300m grid-based system that ranks candidate locations for new green space by underserved need

\- \*\*Ward-level analysis\*\* — aggregates findings across 424 real Bengaluru ward boundaries, filtered to statistically reliable results

\- \*\*Two trained ML models:\*\*

&#x20; - A Random Forest Regressor (tabular) with train/test split, cross-validation, and hyperparameter tuning

&#x20; - A Convolutional Neural Network trained directly on raw satellite image patches (99.98% test accuracy)

\- \*\*Impact evaluation\*\* — statistical regression testing the relationship between park proximity and vegetation health

\- \*\*Interactive dashboard\*\* — a Streamlit app with a live 3D map, dynamic statistics, and a dedicated policy-maker view



\## Key Findings



\- Identified \*\*Hemmigepura\*\* as Bengaluru's highest-priority underserved ward for new green space (avg. suitability score 0.40 across 78 sample points)

\- Found a statistically significant but weak correlation (r=0.20, p<0.001) between park proximity and vegetation health — suggesting neighborhood greenery is driven by more than just formal park placement

\- Cross-validation revealed vegetation patches that OpenStreetMap's crowd-sourced data had not captured



\## Tech Stack



\- \*\*Geospatial processing:\*\* GeoPandas, Rasterio, OSMnx

\- \*\*Machine learning:\*\* Scikit-learn, TensorFlow/Keras

\- \*\*Dashboard:\*\* Streamlit, PyDeck, Folium

\- \*\*Data sources:\*\* \[Copernicus Sentinel-2](https://dataspace.copernicus.eu) (satellite imagery), \[OpenStreetMap](https://www.openstreetmap.org) (boundary \& park data)



\## Running Locally



```bash

git clone https://github.com/BhargaviiS/urban-green-space-planning.git

cd urban-green-space-planning

pip install -r requirements.txt

streamlit run app.py

```



Note: the dashboard reads pre-generated analysis outputs (images and CSVs) included in this repo. The full satellite processing pipeline (NDVI computation, model training) requires downloading Sentinel-2 imagery separately via the Copernicus Data Space Ecosystem.



\## Project Status



Working proof-of-concept covering a real Bengaluru satellite tile with ward-level granularity and two independently trained, validated ML models. Next steps toward production: full-city tile mosaicking, official BBMP ward-boundary integration, and longitudinal impact tracking with historical satellite data.

