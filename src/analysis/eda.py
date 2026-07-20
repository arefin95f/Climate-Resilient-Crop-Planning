"""Exploratory data analysis and geospatial visualizations."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config import CLIMATE_FEATURES, DATA_PROCESSED, DATA_RAW, FEATURE_COLUMNS, REPORTS_FIGURES, SOIL_FEATURES


def load_merged() -> pd.DataFrame:
    path = DATA_PROCESSED / "merged_dataset.csv"
    if not path.exists():
        raise FileNotFoundError("Run the data pipeline first: python -m src.data.pipeline")
    return pd.read_csv(path)


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Correlation heatmap of features vs yield."""
    cols = FEATURE_COLUMNS + ["yield_ton_ha"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation with Boro Rice Yield", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(REPORTS_FIGURES / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    return fig


# Map our district CSV names to geoBoundaries shapeName spellings
DISTRICT_NAME_MAP = {
    "Barishal": "Barisal",
    "Chattogram": "Chittagong",
    "Cumilla": "Comilla",
    "Bogura": "Bogra",
    "Chapainawabganj": "Nawabganj",
    "Moulvibazar": "Maulvibazar",
    "Brahmanbaria": "Brahamanbaria",
    "Netrokona": "Netrakona",
}
GEOJSON_TO_CSV = {v: k for k, v in DISTRICT_NAME_MAP.items()}


def _normalize_district_names(df: pd.DataFrame, col: str = "district") -> pd.DataFrame:
    """Align district names with geojson shapeName values."""
    out = df.copy()
    out[col] = out[col].replace(DISTRICT_NAME_MAP)
    return out


def _load_district_geometries() -> gpd.GeoDataFrame:
    """Load or build district point geometries for choropleth-style maps."""
    geojson_path = DATA_RAW / "bangladesh_districts.geojson"
    districts = pd.read_csv(DATA_RAW / "districts.csv")

    if geojson_path.exists():
        gdf = gpd.read_file(geojson_path)
        if "shapeName" in gdf.columns:
            gdf = gdf.rename(columns={"shapeName": "district"})
        elif "ADM2_EN" in gdf.columns:
            gdf = gdf.rename(columns={"ADM2_EN": "district"})
        gdf["district"] = gdf["district"].replace(GEOJSON_TO_CSV)
        return gdf

    geometry = gpd.points_from_xy(districts.longitude, districts.latitude)
    return gpd.GeoDataFrame(districts, geometry=geometry, crs="EPSG:4326")


def plot_yield_choropleth(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Choropleth map of average Boro yield per district."""
    avg_yield = df.groupby("district")["yield_ton_ha"].mean().reset_index()
    gdf = _load_district_geometries()
    merged = gdf.merge(avg_yield, on="district", how="left")

    fig, ax = plt.subplots(figsize=(10, 12))
    is_point = merged.geometry.geom_type.iloc[0] == "Point"
    merged.plot(
        column="yield_ton_ha", cmap="YlGn", legend=True, ax=ax,
        legend_kwds={"label": "Avg Yield (t/ha)", "shrink": 0.6},
        markersize=200 if is_point else None,
    )
    ax.set_title("Average Boro Rice Yield by District", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()

    if save:
        REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(REPORTS_FIGURES / "yield_choropleth.png", dpi=150, bbox_inches="tight")
    return fig


def plot_rainfall_trend(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Map showing change in pre-monsoon (Mar-Jun) rainfall over 20 years."""
    early = df[df["year"] <= df["year"].median()]
    late = df[df["year"] > df["year"].median()]

    early_rain = early.groupby("district")["rainfall_mar_jun"].mean()
    late_rain = late.groupby("district")["rainfall_mar_jun"].mean()
    change = (late_rain - early_rain).reset_index()
    change.columns = ["district", "rainfall_change"]

    gdf = _load_district_geometries()
    merged = gdf.merge(change, on="district", how="left")

    fig, ax = plt.subplots(figsize=(10, 12))
    merged.plot(
        column="rainfall_change", cmap="RdYlBu", legend=True, ax=ax,
        legend_kwds={"label": "Rainfall Change (mm)", "shrink": 0.6},
        markersize=200 if merged.geometry.geom_type.iloc[0] == "Point" else None,
    )
    ax.set_title("Change in Pre-Monsoon Rainfall (Mar–Jun)\nSecond Half vs First Half of Study Period",
                 fontsize=13, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()

    if save:
        fig.savefig(REPORTS_FIGURES / "rainfall_change_map.png", dpi=150, bbox_inches="tight")
    return fig


def plot_soil_yield_comparison(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Side-by-side soil organic carbon and yield maps."""
    avg = df.groupby("district").agg(
        yield_ton_ha=("yield_ton_ha", "mean"),
        soil_soc=("soil_soc", "first"),
    ).reset_index()

    gdf = _load_district_geometries()
    merged = gdf.merge(avg, on="district", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    merged.plot(column="soil_soc", cmap="YlOrBr", legend=True, ax=axes[0],
                legend_kwds={"label": "SOC (g/kg)", "shrink": 0.6},
                markersize=200 if merged.geometry.geom_type.iloc[0] == "Point" else None)
    axes[0].set_title("Soil Organic Carbon", fontweight="bold")
    axes[0].set_axis_off()

    merged.plot(column="yield_ton_ha", cmap="YlGn", legend=True, ax=axes[1],
                legend_kwds={"label": "Yield (t/ha)", "shrink": 0.6},
                markersize=200 if merged.geometry.geom_type.iloc[0] == "Point" else None)
    axes[1].set_title("Average Boro Rice Yield", fontweight="bold")
    axes[1].set_axis_off()

    fig.suptitle("Soil Quality vs Crop Yield", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(REPORTS_FIGURES / "soil_yield_comparison.png", dpi=150, bbox_inches="tight")
    return fig


def plot_yield_decomposition(df: pd.DataFrame, district: str = "Rajshahi", save: bool = True) -> plt.Figure:
    """Time-series decomposition of yield for a key rice-producing district."""
    ts = df[df["district"] == district].sort_values("year").set_index("year")["yield_ton_ha"]
    if len(ts) < 4:
        district = df.groupby("district")["yield_ton_ha"].mean().idxmax()
        ts = df[df["district"] == district].sort_values("year").set_index("year")["yield_ton_ha"]

    decomp = seasonal_decompose(ts, model="additive", period=2, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], color="black", marker="o")
    axes[0].set_ylabel("Observed")
    axes[0].set_title(f"Yield Decomposition — {district}", fontweight="bold")

    decomp.trend.plot(ax=axes[1], color="steelblue")
    axes[1].set_ylabel("Trend")

    decomp.seasonal.plot(ax=axes[2], color="green")
    axes[2].set_ylabel("Seasonal")

    decomp.resid.plot(ax=axes[3], color="red", marker=".")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Year")

    plt.tight_layout()
    if save:
        fig.savefig(REPORTS_FIGURES / f"yield_decomposition_{district.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    return fig


def run_eda(save: bool = True) -> dict:
    """Run full EDA suite and save figures."""
    print("Running EDA...")
    df = load_merged()
    figures = {
        "correlation": plot_correlation_heatmap(df, save),
        "yield_map": plot_yield_choropleth(df, save),
        "rainfall_map": plot_rainfall_trend(df, save),
        "soil_yield": plot_soil_yield_comparison(df, save),
        "decomposition": plot_yield_decomposition(df, save),
    }
    print(f"  Saved figures to {REPORTS_FIGURES}")
    return {"data": df, "figures": figures}
