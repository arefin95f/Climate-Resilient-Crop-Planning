"""AgriSight-BD Streamlit application."""
import json
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = DATA_PROCESSED / "model_artifacts"

st.set_page_config(
    page_title="AgriSight-BD",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    merged = pd.read_csv(DATA_PROCESSED / "merged_dataset.csv")
    districts = pd.read_csv(DATA_RAW / "districts.csv")
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return merged, districts, metadata


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    features = joblib.load(MODEL_DIR / "feature_names.pkl")
    return model, scaler, features


def get_district_features(merged: pd.DataFrame, district: str, year: int | None = None) -> pd.Series:
    """Get latest or specified-year features for a district."""
    ddf = merged[merged["district"] == district]
    if year:
        ddf = ddf[ddf["year"] == year]
    return ddf.iloc[-1] if len(ddf) > 0 else ddf.mean()


def predict_yield(model, scaler, features, row: pd.Series, overrides: dict) -> float:
    """Predict yield with optional feature overrides."""
    X = pd.DataFrame([{f: overrides.get(f, row.get(f, 0)) for f in features}])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


def tab_prediction_playground(merged, districts, model, scaler, features):
    st.header("Prediction Playground")
    st.markdown(
        "Select a district and adjust climate parameters to predict **Boro rice yield** (t/ha)."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        district = st.selectbox("District", sorted(districts["district"].unique()))
        base = get_district_features(merged, district)

        rainfall = st.slider(
            "Expected Seasonal Rainfall (mm)",
            min_value=800.0, max_value=3500.0,
            value=float(base.get("rainfall_total", 1800)),
            step=50.0,
        )
        temp_min = st.slider(
            "Average Min Temperature (°C)",
            min_value=12.0, max_value=26.0,
            value=float(base.get("temp_min_mean", 18)),
            step=0.5,
        )
        temp_max = st.slider(
            "Average Max Temperature (°C)",
            min_value=26.0, max_value=38.0,
            value=float(base.get("temp_max_mean", 31)),
            step=0.5,
        )
        ndvi = st.slider(
            "Vegetation Index (NDVI peak)",
            min_value=0.3, max_value=0.9,
            value=float(base.get("ndvi_peak", 0.72)),
            step=0.01,
        )

        if st.button("Predict Yield", type="primary"):
            overrides = {
                "rainfall_total": rainfall,
                "rainfall_mar_jun": rainfall * 0.4,
                "temp_min_mean": temp_min,
                "temp_max_mean": temp_max,
                "temp_max_mar_jun": temp_max + 1.5,
                "ndvi_peak": ndvi,
                "ndvi_mean": ndvi - 0.12,
            }
            pred = predict_yield(model, scaler, features, base, overrides)
            hist_avg = merged[merged["district"] == district]["yield_ton_ha"].mean()

            st.success(f"**Predicted Yield: {pred:.2f} t/ha**")
            delta = pred - hist_avg
            st.metric("Historical Average", f"{hist_avg:.2f} t/ha", f"{delta:+.2f} t/ha")

    with col2:
        hist = merged[merged["district"] == district].sort_values("year")
        fig = px.line(
            hist, x="year", y="yield_ton_ha",
            title=f"Historical Boro Yield — {district}",
            labels={"yield_ton_ha": "Yield (t/ha)", "year": "Year"},
            markers=True,
        )
        fig.add_hline(y=hist["yield_ton_ha"].mean(), line_dash="dash",
                      annotation_text="Historical avg")
        st.plotly_chart(fig, use_container_width=True)


def tab_what_if(merged, districts, model, scaler, features):
    st.header("What-If Scenario Analysis")
    st.markdown("Explore how climate shocks affect predicted yield compared to historical averages.")

    district = st.selectbox("Select District", sorted(districts["district"].unique()), key="whatif_district")
    scenario = st.selectbox("Scenario", [
        "Normal (baseline)",
        "Drought: 30% below normal rainfall",
        "Drought: 30% below + heatwave (+3°C max temp)",
        "Flood: 40% above normal rainfall",
        "Early heatwave during grain filling (+4°C in Mar-Jun)",
    ])

    base = get_district_features(merged, district)
    hist_avg = merged[merged["district"] == district]["yield_ton_ha"].mean()

    overrides = {}
    if "30% below normal rainfall" in scenario:
        overrides["rainfall_total"] = base["rainfall_total"] * 0.7
        overrides["rainfall_mar_jun"] = base["rainfall_mar_jun"] * 0.7
    if "heatwave (+3°C" in scenario:
        overrides["temp_max_mean"] = base["temp_max_mean"] + 3
        overrides["temp_max_mar_jun"] = base["temp_max_mar_jun"] + 3
    if "40% above normal" in scenario:
        overrides["rainfall_total"] = base["rainfall_total"] * 1.4
        overrides["rainfall_mar_jun"] = base["rainfall_mar_jun"] * 1.4
    if "grain filling (+4°C" in scenario:
        overrides["temp_max_mar_jun"] = base["temp_max_mar_jun"] + 4

    baseline_pred = predict_yield(model, scaler, features, base, {})
    scenario_pred = predict_yield(model, scaler, features, base, overrides)
    loss_pct = ((scenario_pred - hist_avg) / hist_avg) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Historical Average", f"{hist_avg:.2f} t/ha")
    col2.metric("Scenario Prediction", f"{scenario_pred:.2f} t/ha", f"{scenario_pred - hist_avg:+.2f}")
    col3.metric("Change vs Average", f"{loss_pct:+.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ["Historical\nAverage", "Baseline\nPrediction", "Scenario\nPrediction"]
    values = [hist_avg, baseline_pred, scenario_pred]
    colors = ["#2ecc71", "#3498db", "#e74c3c" if scenario_pred < hist_avg else "#f39c12"]
    ax.bar(bars, values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Yield (t/ha)")
    ax.set_title(f"What-If: {scenario}\n{district}", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
    st.pyplot(fig)
    plt.close()

    st.info(
        "Climate shocks during the Boro grain-filling stage (March–June) disproportionately "
        "impact yield. The model captures these non-linear effects through temperature and "
        "rainfall interaction features."
    )


def tab_crop_suitability(merged, districts):
    st.header("Crop Suitability Zones")
    st.markdown(
        "Districts clustered by **static soil** and **long-term average climate** "
        "to identify innate crop suitability zones."
    )

    n_clusters = st.slider("Number of zones", 3, 6, 4)

    static_features = [
        "soil_ph", "soil_soc", "soil_nitrogen", "soil_clay",
        "temp_max_mean", "rainfall_total", "humidity_mean",
    ]

    district_avg = merged.groupby("district")[static_features].mean().reset_index()
    district_avg = district_avg.merge(districts[["district", "latitude", "longitude", "division"]], on="district")

    scaler = StandardScaler()
    X = scaler.fit_transform(district_avg[static_features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    district_avg["zone"] = kmeans.fit_predict(X)

    zone_labels = {
        0: "High-yield rice belt",
        1: "Maize-suitable upland",
        2: "Coastal adaptive zone",
        3: "Mixed cropping area",
        4: "Stress-prone lowland",
        5: "Diverse agro-zone",
    }
    district_avg["zone_name"] = district_avg["zone"].map(
        lambda z: zone_labels.get(z, f"Zone {z}")
    )

    crop_rec = {
        "High-yield rice belt": "Boro & Aman rice (high input)",
        "Maize-suitable upland": "Maize, Boro rice",
        "Coastal adaptive zone": "Aman rice, salt-tolerant varieties",
        "Mixed cropping area": "Boro rice, Maize rotation",
        "Stress-prone lowland": "Aman rice, flood-tolerant varieties",
        "Diverse agro-zone": "Mixed: Boro, Aman, Maize",
    }

    fig = px.scatter_mapbox(
        district_avg,
        lat="latitude", lon="longitude",
        color="zone_name",
        hover_name="district",
        hover_data={"division": True, "soil_soc": ":.1f", "rainfall_total": ":.0f"},
        zoom=6.5, center={"lat": 23.7, "lon": 90.4},
        mapbox_style="carto-positron",
        title="Crop Suitability Zones (K-Means Clustering)",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Zone Recommendations")
    for zone_id in sorted(district_avg["zone"].unique()):
        zone_name = district_avg[district_avg["zone"] == zone_id]["zone_name"].iloc[0]
        zone_districts = district_avg[district_avg["zone"] == zone_id]["district"].tolist()
        rec = crop_rec.get(zone_name, "Mixed cropping")
        with st.expander(f"{zone_name} ({len(zone_districts)} districts)"):
            st.write(f"**Recommended crops:** {rec}")
            st.write(f"**Districts:** {', '.join(zone_districts)}")


def main():
    st.sidebar.title("AgriSight-BD")
    st.sidebar.markdown(
        "**Geospatial ML Framework**  \n"
        "Climate-Resilient Crop Planning  \n"
        "🇧🇩 Bangladesh"
    )

    try:
        merged, districts, metadata = load_data()
        model, scaler, features = load_model()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run the pipeline first:\n\n"
            "```\npython run_pipeline.py\n```"
        )
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Best Model:** {metadata['best_model']}")
    st.sidebar.markdown(f"**RMSE:** {metadata['results'][metadata['best_model']]['rmse_mean']:.3f}")
    st.sidebar.markdown(f"**Data:** {metadata['n_samples']} records ({metadata['year_range'][0]}–{metadata['year_range'][1]})")

    tab1, tab2, tab3 = st.tabs([
        "Prediction Playground",
        "What-If Analysis",
        "Crop Suitability Map",
    ])

    with tab1:
        tab_prediction_playground(merged, districts, model, scaler, features)
    with tab2:
        tab_what_if(merged, districts, model, scaler, features)
    with tab3:
        tab_crop_suitability(merged, districts)

    st.sidebar.markdown("---")
    st.sidebar.caption("AgriSight-BD | NASA POWER · ISRIC · BBS · MODIS NDVI")


if __name__ == "__main__":
    main()
