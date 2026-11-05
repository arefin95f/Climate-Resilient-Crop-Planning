"""AgriSight-BD Streamlit application."""

import html
import io
import math
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.accounts import (
    ROLE_LABELS,
    admin_update_user,
    authenticate,
    change_password,
    create_account,
    ensure_default_admin,
    delete_note,
    delete_run,
    get_note,
    get_user,
    list_notes,
    list_runs,
    list_users,
    save_note,
    save_run,
    update_profile,
    user_count,
)
from src.config import CROP_CATALOG, PRIMARY_CROP
from src.db import (
    load_catalog_summary,
    load_crop_frame,
    load_districts,
    load_model_metadata,
)
from src.models.features import add_forecast_features, prediction_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = DATA_PROCESSED / "model_artifacts"
FIGURES_DIR = PROJECT_ROOT / "src" / "data" / "assets"

PREDICTION_IMAGE = FIGURES_DIR / "prediction_playground.png"
WHAT_IF_IMAGE = FIGURES_DIR / "what_if_analysis.png"
SUITABILITY_IMAGE = FIGURES_DIR / "crop_suitability.png"


st.set_page_config(
    page_title="AgriSight-BD",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ZONE_COLORS = {
    "Highest-yield zone": "#1B7A4E",
    "Favorable zone": "#2A9D8F",
    "Moderate zone": "#E0A100",
    "Lower-yield zone": "#E07A3D",
    "Stress-prone zone": "#C4533D",
    "Marginal zone": "#6D597A",
}

ZONE_FEATURES = [
    "soil_ph",
    "soil_soc",
    "soil_nitrogen",
    "soil_clay",
    "temp_max_mean",
    "rainfall_total",
    "humidity_mean",
]

ZONE_RANK_NAMES = [
    "Highest-yield zone",
    "Favorable zone",
    "Moderate zone",
    "Lower-yield zone",
    "Stress-prone zone",
    "Marginal zone",
]

SCENARIO_NOTES = {
    "Normal (baseline)": (
        "No extra rainfall or temperature change."
    ),
    "Drought: 30% below normal rainfall": (
        "Rainfall is 30% lower than this district’s latest year."
    ),
    "Drought: 30% below + heatwave (+3°C max temp)": (
        "Rainfall is 30% lower and the hottest temperature rises by 3°C."
    ),
    "Flood: 40% above normal rainfall": (
        "Rainfall is 40% higher than this district’s latest year."
    ),
    "Heatwave during the critical season (+4°C)": (
        "The hottest temperature is 4°C higher through the critical season."
    ),
}

SCENARIO_EFFECTS = {
    "Normal (baseline)": {},
    "Drought: 30% below normal rainfall": {
        "rainfall_scale": 0.7,
    },
    "Drought: 30% below + heatwave (+3°C max temp)": {
        "rainfall_scale": 0.7,
        "temp_delta": 3.0,
    },
    "Flood: 40% above normal rainfall": {
        "rainfall_scale": 1.4,
    },
    "Heatwave during the critical season (+4°C)": {
        "temp_delta": 4.0,
    },
}

PROFILE_RISKS = [
    ("Drought", "Drought: 30% below normal rainfall"),
    (
        "Drought and heatwave",
        "Drought: 30% below + heatwave (+3°C max temp)",
    ),
    ("Flood", "Flood: 40% above normal rainfall"),
    (
        "Heatwave",
        "Heatwave during the critical season (+4°C)",
    ),
]

# A 2030 stress test of the latest year, not a weather forecast.
OUTLOOK_STRESSES = [
    (
        "Warmer",
        {"temp_delta": 1.5},
        "The hottest temperature is 1.5°C higher than the latest year.",
    ),
    (
        "Drier",
        {"rainfall_scale": 0.85},
        "Rainfall is 15% lower than the latest year.",
    ),
    (
        "Wetter",
        {"rainfall_scale": 1.15},
        "Rainfall is 15% higher than the latest year.",
    ),
]

SEASON_GROUPS = [
    (
        "Rabi",
        "November–March",
        [
            "boro_rice",
            "wheat",
            "maize",
            "potato",
            "sweet_potato",
            "mustard",
            "lentil",
            "onion",
            "garlic",
            "chili",
            "groundnut",
            "soybean",
        ],
    ),
    (
        "Kharif",
        "April–December",
        ["aus_rice", "aman_rice", "jute", "sesame"],
    ),
    (
        "Year-round",
        "Stands through the year",
        ["sugarcane"],
    ),
]

STRONG_ZONES = {"Highest-yield zone", "Favorable zone"}
# Peak NDVI this far under the district's own average counts as thin cover.
THIN_COVER_GAP = -0.03

APP_CSS = """
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stHeader"] {
    height: 3.25rem;
    min-height: 3.25rem;
    background: #F4F6F3;
    border-bottom: 1px solid #E4EBE5;
    left: 268px !important;
    width: calc(100% - 268px) !important;
}
[data-testid="stToolbar"],
#MainMenu,
footer,
[data-testid="stDecoration"] {
    display: none;
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E4EBE5;
    transform: none !important;
    visibility: visible !important;
    position: relative !important;
    left: 0 !important;
    margin-left: 0 !important;
    opacity: 1 !important;
    flex: 0 0 268px !important;
    width: 268px !important;
    min-width: 268px !important;
    max-width: 268px !important;
    height: 100vh !important;
    z-index: 100;
}
[data-testid="stMain"] {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: auto !important;
}
[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
}
[data-testid="stColumn"] {
    flex: 1 1 180px !important;
    width: auto !important;
    min-width: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.85rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.15rem;
}
[data-testid="stSidebar"] button {
    min-height: 0;
    min-width: 0;
    padding-top: 0.32rem;
    padding-bottom: 0.32rem;
    white-space: normal;
}
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3.2rem;
    max-width: 1120px;
    width: 100%;
}
.brand {
    padding: 0 2px 8px;
    margin: 0 0 4px 0;
    border-bottom: 1px solid #E4EBE5;
}
.brand .mark {
    display: block;
    color: #0E6B45;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.brand h1 {
    color: #14261C !important;
    font-size: 1.28rem !important;
    line-height: 1.15 !important;
    margin: 0 0 4px 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
}
.brand p {
    color: #5C7266 !important;
    font-size: 0.84rem !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}
.nav-label, .side-label {
    color: #6A7F73;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    line-height: 1.35;
    display: block;
    margin: 0;
    padding: 12px 2px 2px;
}
.nav-label.top {
    padding-top: 0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    margin-bottom: 0 !important;
}
.account-pill {
    background: #F4F7F5;
    border: 1px solid #E4EBE5;
    border-radius: 12px;
    padding: 10px 12px;
    margin: 0 0 8px 0;
}
.account-pill span {
    display: block;
    color: #14261C;
    font-weight: 650;
    font-size: 0.92rem;
}
.account-pill p {
    color: #5C7266;
    font-size: 0.8rem;
    line-height: 1.35;
    margin: 2px 0 0 0;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    background: transparent;
    border: 0;
    color: #24382D;
    text-align: left;
    justify-content: flex-start;
    border-radius: 8px;
    padding: 0.42rem 0.7rem;
    font-weight: 500;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: #E8F5EE;
    border: 0;
    color: #0C5A39;
    text-align: left;
    justify-content: flex-start;
    border-radius: 8px;
    box-shadow: inset 3px 0 0 #0E6B45;
    font-weight: 650;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background: #FFFFFF;
    border: 1px solid #D5E0D8;
    color: #163126;
    text-align: center;
    justify-content: center;
    box-shadow: none;
    font-weight: 600;
}
[data-testid="stSidebarHeader"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
.page-head {
    padding: 4px 0 8px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid #E4EBE5;
}
.page-head .eyebrow {
    color: #0E6B45;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 6px 0;
}
.page-head h1 {
    color: #14261C !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.15 !important;
    margin: 0 0 8px 0 !important;
}
.page-head p {
    color: #3E5648 !important;
    font-size: 1.02rem !important;
    line-height: 1.55 !important;
    margin: 0 0 8px 0 !important;
    max-width: 760px;
}
.stat-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    margin: 8px 0 4px 0;
}
.stat-grid div, .kpi-row div, [data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E4EBE5;
    border-radius: 12px;
}
.stat-grid div {
    padding: 8px 10px;
}
.stat-grid span, .kpi-row span {
    display: block;
    color: #6A7F73;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.stat-grid strong {
    color: #14261C;
    font-size: 0.95rem;
}
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0 20px 0;
}
.kpi-row div {
    padding: 14px 14px 12px;
    border-top: 3px solid #0E6B45;
}
.kpi-row strong {
    display: block;
    color: #14261C;
    font-size: 1.45rem;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin-top: 4px;
}
[data-testid="stMetric"] {
    padding: 10px 14px 4px 14px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E4EBE5 !important;
    border-radius: 14px !important;
    box-shadow: none;
}
[data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
    background: #0E6B45;
    border: 1px solid #0E6B45;
    color: #FFFFFF;
    border-radius: 10px;
    font-weight: 650;
}
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {
    background: #FFFFFF;
    border: 1px solid #D5E0D8;
    color: #163126;
    border-radius: 10px;
    font-weight: 600;
}
[data-testid="stTextInputRootElement"] {
    background: #F7FAF8 !important;
    border: 1px solid #C9D6CC !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: #F7FAF8 !important;
    border-color: #C9D6CC !important;
}
[data-testid="stForm"] [data-testid="stMarkdownContainer"] {
    margin-bottom: 0 !important;
}
[data-testid="stForm"] {
    border: 1px solid #E4EBE5;
    border-radius: 16px;
    padding: 1rem 1.1rem 0.85rem;
    background: #FFFFFF;
    height: auto !important;
}
[data-testid="stColumn"]:has([data-testid="stForm"]) {
    flex: 1 1 320px !important;
    align-self: flex-start !important;
    width: auto !important;
    min-width: min(100%, 280px) !important;
}
[data-testid="stForm"] [data-testid="stVerticalBlock"] {
    gap: 0.45rem !important;
}
[data-testid="stForm"] [data-testid="stElementContainer"] {
    flex: 0 0 auto !important;
    height: auto !important;
}
.auth-head {
    margin: 0 0 10px 0;
}
.auth-head h2 {
    color: #14261C !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin: 0 0 4px 0 !important;
}
.auth-head p {
    color: #5C7266 !important;
    font-size: 0.92rem !important;
    line-height: 1.45 !important;
    margin: 0 !important;
}
[data-testid="stMain"] [data-testid="stVerticalBlock"]:has(.auth-stack) {
    width: min(420px, 100%) !important;
    max-width: 420px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    align-self: center !important;
}
.auth-or {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 2px;
    color: #6A7F73;
    font-size: 0.82rem;
    font-weight: 650;
    letter-spacing: 0.04em;
    text-transform: lowercase;
}
.auth-or::before,
.auth-or::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #D5E0D8;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #E4EBE5;
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
h2, h3 {
    letter-spacing: -0.03em !important;
    color: #14261C !important;
}
[data-testid="stCaptionContainer"] {
    color: #5C7266;
}
hr {
    border: 0;
    border-top: 1px solid #E4EBE5;
    margin: 0.6rem 0;
}
@media (max-width: 960px) {
    .page-head h1 {
        font-size: 1.6rem !important;
    }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <header class="page-head">
            <p class="eyebrow">AgriSight-BD</p>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def label_every_year(figure, years) -> None:
    """Put one tick on each year."""

    years = [int(year) for year in years]
    figure.update_xaxes(
        tickmode="array",
        tickvals=years,
        ticktext=[str(year) for year in years],
        tickangle=-90,
        automargin=True,
        tickfont=dict(size=12),
        range=[years[0] - 0.5, years[-1] + 0.5],
    )


def bangladesh_map_view(height: int) -> dict:
    """Open on all of Bangladesh, including on a wide screen.

    A bounds box is fitted to the map width, which crops the north
    and south when the page is wide. The zoom is taken from the
    height instead, so the whole country stays inside the frame.
    """

    def mercator(latitude: float) -> float:
        phi = math.radians(latitude)
        return math.log(math.tan(math.pi / 4 + phi / 2))

    south = 19.7
    north = 27.3
    span = (mercator(north) - mercator(south)) / (2 * math.pi)
    world_pixels = (height * 0.88) / span
    zoom = math.log2(world_pixels / 512)
    midpoint = (mercator(north) + mercator(south)) / 2
    latitude = math.degrees(
        2 * math.atan(math.exp(midpoint)) - math.pi / 2
    )
    return {
        "center": {"lat": round(latitude, 2), "lon": 90.35},
        "zoom": round(zoom, 2),
    }


def style_chart(figure, cartesian: bool = True):
    """Match Plotly charts to the light app theme."""

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Segoe UI, Helvetica Neue, Arial, sans-serif",
            color="#14261C",
            size=13,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#D7E4D4",
            borderwidth=1,
        ),
    )
    if cartesian:
        figure.update_layout(
            plot_bgcolor="rgba(255,255,255,0.72)",
            margin=dict(l=8, r=8, t=56, b=8),
        )
        figure.update_xaxes(
            gridcolor="#E3EBDF",
            zerolinecolor="#E3EBDF",
        )
        figure.update_yaxes(
            gridcolor="#E3EBDF",
            zerolinecolor="#E3EBDF",
        )
    return figure


def _snap(value: float, low: float, high: float, step: float) -> float:
    value = min(max(float(value), low), high)
    steps = round((value - low) / step)
    snapped = round(low + steps * step, 6)
    return float(min(max(snapped, low), high))


def crop_label(crop_id: str) -> str:
    """Display name for a crop id."""

    return CROP_CATALOG[crop_id]["label"]


@st.cache_data
def load_data(crop_id: str):
    merged = add_forecast_features(
        load_crop_frame(crop_id)
    )

    districts = load_districts()
    metadata = load_model_metadata(crop_id)

    return merged, districts, metadata


@st.cache_resource
def load_model(crop_id: str):
    model_dir = MODEL_DIR / crop_id

    model = joblib.load(
        model_dir / "best_model.pkl"
    )

    scaler = joblib.load(
        model_dir / "scaler.pkl"
    )

    features = joblib.load(
        model_dir / "feature_names.pkl"
    )

    return model, scaler, features


@st.cache_data
def load_crop_catalog() -> pd.DataFrame:
    """Summary row for every downloadable crop dataset."""

    summary = load_catalog_summary().set_index("crop")
    rows = []

    for crop_id, spec in CROP_CATALOG.items():
        if crop_id not in summary.index:
            raise LookupError(
                f"No MySQL rows for crop '{crop_id}'."
            )
        row = summary.loc[crop_id]
        rows.append(
            {
                "crop_id": crop_id,
                "Crop": spec["label"],
                "Group": spec["group"],
                "Season": spec["season"],
                "Rows": int(row["records"]),
                "Districts": int(row["districts"]),
                "Years": (
                    f"{int(row['start_year'])}–"
                    f"{int(row['end_year'])}"
                ),
                "Mean yield (t/ha)": round(
                    float(row["mean_yield"]),
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


def dataset_bytes(crop_id: str) -> bytes:
    """CSV bytes for one crop modeling dataset."""

    return load_crop_frame(crop_id).to_csv(index=False).encode("utf-8")


def all_datasets_zip() -> bytes:
    """Zip every crop dataset for a single download."""

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for crop_id in CROP_CATALOG:
            archive.writestr(
                f"{crop_id}.csv",
                dataset_bytes(crop_id),
            )

    return buffer.getvalue()


def get_district_features(
    merged: pd.DataFrame,
    district: str,
    year: int | None = None,
) -> pd.Series:
    """Get latest or specified-year features for a district."""

    district_data = merged[
        merged["district"] == district
    ].sort_values("year")

    if year:
        district_data = district_data[
            district_data["year"] == year
        ]

    if len(district_data) > 0:
        return district_data.iloc[-1]

    return district_data.mean()


def predict_yield(
    model,
    scaler,
    features,
    row: pd.Series,
    overrides: dict,
) -> float:
    """Predict yield with optional feature overrides."""

    input_data = prediction_frame(row, features, overrides)

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    return float(prediction)


def scenario_overrides(base: pd.Series, scenario: str) -> dict:
    """Feature changes for one named climate shock."""

    effect = SCENARIO_EFFECTS[scenario]
    overrides = {}

    rainfall_scale = effect.get("rainfall_scale")
    if rainfall_scale is not None:
        overrides["rainfall_total"] = (
            base["rainfall_total"] * rainfall_scale
        )
        overrides["rainfall_mar_jun"] = (
            base["rainfall_mar_jun"] * rainfall_scale
        )

    temp_delta = effect.get("temp_delta")
    if temp_delta is not None:
        overrides["temp_max_mean"] = (
            base["temp_max_mean"] + temp_delta
        )
        overrides["temp_max_mar_jun"] = (
            base["temp_max_mar_jun"] + temp_delta
        )

    return overrides


def assign_zones(
    merged: pd.DataFrame,
    districts: pd.DataFrame,
    n_clusters: int,
) -> tuple[pd.DataFrame, float]:
    """Cluster districts and name zones by this crop's mean yield."""

    district_average = (
        merged.groupby("district")[ZONE_FEATURES]
        .mean()
        .reset_index()
    )
    district_average = district_average.merge(
        districts[
            [
                "district",
                "latitude",
                "longitude",
                "division",
            ]
        ],
        on="district",
    )

    scaled_features = StandardScaler().fit_transform(
        district_average[ZONE_FEATURES]
    )
    district_average["zone"] = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    ).fit_predict(scaled_features)

    yield_by_district = merged.groupby("district")[
        "yield_ton_ha"
    ].mean()
    district_average["mean_yield"] = (
        district_average["district"].map(yield_by_district)
    )
    overall_yield = float(
        district_average["mean_yield"].mean()
    )
    zone_yield = (
        district_average.groupby("zone")["mean_yield"]
        .mean()
        .sort_values(ascending=False)
    )
    zone_labels = {
        zone_id: ZONE_RANK_NAMES[rank]
        for rank, zone_id in enumerate(zone_yield.index)
    }
    district_average["zone_name"] = (
        district_average["zone"].map(zone_labels)
    )
    return district_average, overall_yield


def zone_note(
    crop_name: str,
    zone_mean: float,
    overall_yield: float,
) -> str:
    """Plain-language fit note for one suitability zone."""

    if zone_mean >= overall_yield:
        return (
            f"A stronger fit for {crop_name}. "
            f"These districts average {zone_mean:.2f} t/ha, "
            f"above the country figure of {overall_yield:.2f} t/ha."
        )
    return (
        f"A weaker fit for {crop_name}. "
        f"These districts average {zone_mean:.2f} t/ha, "
        f"below the country figure of {overall_yield:.2f} t/ha. "
        "Stress-tolerant varieties are a safer choice."
    )


def _pdf_text(text: str) -> str:
    """Keep briefing text inside the standard PDF font."""

    cleaned = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("·", "|")
        .replace("°", " deg")
        .replace("±", "+/-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    cleaned = cleaned.encode("latin-1", "replace").decode("latin-1")
    return (
        cleaned.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _wrap_text(text: str, width: int = 88) -> list[str]:
    lines = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if len(trial) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def _assemble_pdf(streams: list[bytes]) -> bytes:
    """Write a multi-page PDF from content streams."""

    if not streams:
        streams = [b"BT ET"]
    page_count = len(streams)
    page_ids = [5 + (2 * index) for index in range(page_count)]
    content_ids = [6 + (2 * index) for index in range(page_count)]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    for index, stream in enumerate(streams):
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R "
                "/MediaBox [0 0 595 842] "
                f"/Contents {content_ids[index]} 0 R "
                "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
            ).encode("latin-1")
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("latin-1"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_at = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    output.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(output)


def briefing_pdf(blocks: list[tuple[str, str]]) -> bytes:
    """Build a PDF. Each block is (kind, text). Lists continue on the next page."""

    pages: list[list[str]] = []
    commands: list[str] = []
    y_position = 790

    def start_page() -> None:
        nonlocal commands, y_position
        if commands:
            commands.append("ET")
            pages.append(commands)
        commands = ["BT"]
        y_position = 790

    start_page()
    for kind, text in blocks:
        if kind == "gap":
            y_position -= 8
            continue
        if kind == "title":
            font, size, gap = "F2", 16, 24
        elif kind == "section":
            y_position -= 4
            font, size, gap = "F2", 12, 18
        else:
            font, size, gap = "F1", 11, 15
        for line in _wrap_text(text):
            if y_position < 48:
                start_page()
            commands.append(f"/{font} {size} Tf")
            commands.append(
                f"1 0 0 1 48 {y_position} Tm ({_pdf_text(line)}) Tj"
            )
            y_position -= gap
    if commands:
        commands.append("ET")
        pages.append(commands)

    streams = [
        "\n".join(page).encode("latin-1")
        for page in pages
    ]
    return _assemble_pdf(streams)


def _with_climate(frame: pd.DataFrame, effect: dict) -> pd.DataFrame:
    """Copy a season and apply one rainfall or temperature stress."""

    changed = frame.copy()
    if "rainfall_scale" in effect:
        scale = effect["rainfall_scale"]
        changed["rainfall_total"] = changed["rainfall_total"] * scale
        changed["rainfall_mar_jun"] = (
            changed["rainfall_mar_jun"] * scale
        )
    if "temp_delta" in effect:
        delta = effect["temp_delta"]
        changed["temp_max_mean"] = changed["temp_max_mean"] + delta
        changed["temp_max_mar_jun"] = (
            changed["temp_max_mar_jun"] + delta
        )
    return changed


def predict_frame(model, scaler, features, frame: pd.DataFrame) -> np.ndarray:
    """Predict one yield per row, refreshing climate anomalies from raw values."""

    names = [str(name) for name in features]
    columns = {}
    for name in names:
        if name.endswith("_anom"):
            base = name[: -len("_anom")]
            current = pd.to_numeric(frame[base], errors="coerce")
            normal = pd.to_numeric(
                frame[f"{base}_normal"],
                errors="coerce",
            )
            columns[name] = (current - normal).to_numpy(dtype=float)
        else:
            values = pd.to_numeric(frame[name], errors="coerce")
            columns[name] = values.fillna(0.0).to_numpy(dtype=float)
    matrix = pd.DataFrame(columns, columns=names)
    scaled = scaler.transform(matrix)
    return np.asarray(model.predict(scaled), dtype=float)


def _latest_rows(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.sort_values(["district", "year"])
        .groupby("district", as_index=False)
        .tail(1)
        .set_index("district")
    )


def _district_lookup(merged: pd.DataFrame) -> pd.DataFrame:
    return merged[
        ["district", "division", "latitude", "longitude"]
    ].drop_duplicates("district")


def attention_table(merged, model, scaler, features) -> pd.DataFrame:
    """One row per district: usual harvest, estimate, and worst shock."""

    latest = _latest_rows(merged)
    usual = merged.groupby("district")["yield_ton_ha"].mean()
    estimate = pd.Series(
        predict_frame(model, scaler, features, latest),
        index=latest.index,
    )
    shock_values = {}
    for label, scenario in PROFILE_RISKS:
        changed = latest.copy()
        effect = SCENARIO_EFFECTS[scenario]
        if "rainfall_scale" in effect:
            scale = effect["rainfall_scale"]
            changed["rainfall_total"] = (
                changed["rainfall_total"] * scale
            )
            changed["rainfall_mar_jun"] = (
                changed["rainfall_mar_jun"] * scale
            )
        if "temp_delta" in effect:
            delta = effect["temp_delta"]
            changed["temp_max_mean"] = (
                changed["temp_max_mean"] + delta
            )
            changed["temp_max_mar_jun"] = (
                changed["temp_max_mar_jun"] + delta
            )
        shock_values[label] = pd.Series(
            predict_frame(model, scaler, features, changed),
            index=latest.index,
        )

    shocks = pd.DataFrame(shock_values)
    zones, _overall = assign_zones(
        merged,
        _district_lookup(merged),
        4,
    )
    zone_name = zones.set_index("district")["zone_name"]
    usual_aligned = usual.reindex(latest.index)
    table = pd.DataFrame(
        {
            "District": latest.index,
            "Division": latest["division"].to_numpy(),
            "Zone": zone_name.reindex(latest.index).to_numpy(),
            "Usual harvest (t/ha)": usual_aligned.round(2).to_numpy(),
            "Estimate (t/ha)": estimate.round(2).to_numpy(),
            "Vs usual (t/ha)": (
                estimate - usual_aligned
            ).round(2).to_numpy(),
            "Worst shock": shocks.idxmin(axis=1).to_numpy(),
            "Shock drop (t/ha)": (
                shocks.min(axis=1) - estimate
            ).round(2).to_numpy(),
        }
    )
    return table.sort_values(
        ["Vs usual (t/ha)", "Shock drop (t/ha)", "District"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_attention(crop_id: str) -> pd.DataFrame:
    merged, _districts, _metadata = load_data(crop_id)
    model, scaler, features = load_model(crop_id)
    return attention_table(merged, model, scaler, features)


@st.cache_data(show_spinner=False)
def crop_fit_table() -> pd.DataFrame:
    """Zone and own-yield gap for every crop in every district."""

    rows = []
    for crop_id, spec in CROP_CATALOG.items():
        merged, _districts, _metadata = load_data(crop_id)
        model, scaler, features = load_model(crop_id)
        latest = _latest_rows(merged)
        usual = merged.groupby("district")["yield_ton_ha"].mean()
        estimate = pd.Series(
            predict_frame(model, scaler, features, latest),
            index=latest.index,
        )
        zones, _overall = assign_zones(
            merged,
            _district_lookup(merged),
            4,
        )
        zone_name = zones.set_index("district")["zone_name"]
        for district in latest.index:
            own_usual = float(usual.loc[district])
            own_estimate = float(estimate.loc[district])
            rows.append(
                {
                    "crop_id": crop_id,
                    "District": district,
                    "Crop": spec["label"],
                    "Zone": zone_name.loc[district],
                    "Its usual (t/ha)": round(own_usual, 2),
                    "Its estimate (t/ha)": round(own_estimate, 2),
                    "Versus its usual (t/ha)": round(
                        own_estimate - own_usual,
                        2,
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame["zone_rank"] = frame["Zone"].map(
        {
            name: rank
            for rank, name in enumerate(ZONE_RANK_NAMES)
        }
    )
    return frame


@st.cache_data(show_spinner=False)
def outlook_frame() -> pd.DataFrame:
    """Latest-year estimate under each 2030 stress, for every crop."""

    rows = []
    stress_labels = [label for label, _effect, _note in OUTLOOK_STRESSES]
    for crop_id, spec in CROP_CATALOG.items():
        merged, _districts, _metadata = load_data(crop_id)
        model, scaler, features = load_model(crop_id)
        latest = _latest_rows(merged)
        usual = merged.groupby("district")["yield_ton_ha"].mean()
        estimate = pd.Series(
            predict_frame(model, scaler, features, latest),
            index=latest.index,
        )
        zones, _overall = assign_zones(
            merged,
            _district_lookup(merged),
            4,
        )
        zone_name = zones.set_index("district")["zone_name"]
        stressed = {}
        for label, effect, _note in OUTLOOK_STRESSES:
            stressed[label] = pd.Series(
                predict_frame(
                    model,
                    scaler,
                    features,
                    _with_climate(latest, effect),
                ),
                index=latest.index,
            )
        for district in latest.index:
            own_usual = float(usual.loc[district])
            own_estimate = float(estimate.loc[district])
            row = {
                "crop_id": crop_id,
                "Crop": spec["label"],
                "District": district,
                "Division": latest.loc[district, "division"],
                "Zone": zone_name.loc[district],
                "Usual": own_usual,
                "Estimate": own_estimate,
            }
            for label in stress_labels:
                row[label] = float(stressed[label].loc[district])
            rows.append(row)
    return pd.DataFrame(rows)


def season_watch_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Latest rainfall and greenness against each district's own average."""

    latest = _latest_rows(merged)
    rain_usual = merged.groupby("district")["rainfall_total"].mean()
    green_usual = merged.groupby("district")["ndvi_peak"].mean()
    rain_usual = rain_usual.reindex(latest.index)
    green_usual = green_usual.reindex(latest.index)
    rain_pct = (
        (latest["rainfall_total"] - rain_usual) / rain_usual * 100
    )
    green_gap = latest["ndvi_peak"] - green_usual
    dry = rain_pct < 0
    thin = green_gap <= THIN_COVER_GAP
    status = []
    for is_dry, is_thin in zip(dry.to_numpy(), thin.to_numpy()):
        if is_dry and is_thin:
            status.append("Dry, thin cover")
        elif is_dry:
            status.append("Drier than usual")
        elif is_thin:
            status.append("Thin cover")
        else:
            status.append("Near usual")
    table = pd.DataFrame(
        {
            "District": latest.index,
            "Division": latest["division"].to_numpy(),
            "Rainfall (mm)": latest["rainfall_total"].round(0).to_numpy(),
            "Rain vs usual (%)": rain_pct.round(1).to_numpy(),
            "Greenness": latest["ndvi_peak"].round(2).to_numpy(),
            "Greenness vs usual": green_gap.round(2).to_numpy(),
            "Season": status,
        }
    )
    table["_off"] = table["Season"].ne("Near usual")
    return (
        table.sort_values(
            ["_off", "Greenness vs usual", "Rain vs usual (%)", "District"],
            ascending=[False, True, True, True],
        )
        .drop(columns="_off")
        .reset_index(drop=True)
    )


def _zone_rank(zone_name: str) -> int:
    try:
        return ZONE_RANK_NAMES.index(zone_name)
    except ValueError:
        return len(ZONE_RANK_NAMES)


def season_sections(
    fits: pd.DataFrame,
    district: str,
    crop_id: str,
) -> list[tuple[str, str, pd.DataFrame]]:
    """Rabi, kharif, and year-round crops for one district."""

    local = fits[fits["District"] == district]
    sections = []
    for name, window, crop_ids in SEASON_GROUPS:
        chosen = local[local["crop_id"].isin(crop_ids)].copy()
        chosen["_rank"] = chosen["Zone"].map(_zone_rank)
        chosen = chosen.sort_values(
            ["_rank", "Versus its usual (t/ha)", "Crop"],
            ascending=[True, False, True],
        )
        records = []
        for row in chosen.to_dict("records"):
            spec = CROP_CATALOG[row["crop_id"]]
            label = spec["label"]
            if row["crop_id"] == crop_id:
                label = f"{label} · selected"
            records.append(
                {
                    "Crop": label,
                    "Zone": row["Zone"],
                    "Its usual (t/ha)": row["Its usual (t/ha)"],
                    "Its estimate (t/ha)": row["Its estimate (t/ha)"],
                    "Versus its usual (t/ha)": row[
                        "Versus its usual (t/ha)"
                    ],
                }
            )
        sections.append((name, window, pd.DataFrame(records)))
    return sections


def division_pack_blocks(
    crop_name: str,
    division_label: str,
    table: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Lines for a division or country attention briefing."""

    blocks = [
        ("title", "AgriSight-BD attention briefing"),
        ("body", f"{division_label} · {crop_name}"),
        (
            "body",
            "Smallest gap versus the usual harvest is listed first. "
            "The shock figure is the drop from the no-shock estimate.",
        ),
        ("gap", ""),
        ("section", "Districts"),
    ]
    for record in table.to_dict("records"):
        blocks.append(
            (
                "body",
                (
                    f"{record['District']} ({record['Zone']}): "
                    f"usual {record['Usual harvest (t/ha)']:.2f}, "
                    f"estimate {record['Estimate (t/ha)']:.2f}, "
                    f"vs usual {record['Vs usual (t/ha)']:+.2f}, "
                    f"{record['Worst shock']} "
                    f"{record['Shock drop (t/ha)']:+.2f}"
                ),
            )
        )
    return blocks


def season_plan_blocks(
    district: str,
    division: str,
    sections: list[tuple[str, str, pd.DataFrame]],
) -> list[tuple[str, str]]:
    """Lines for one district's year plan."""

    blocks = [
        ("title", "AgriSight-BD season plan"),
        ("body", f"{district}, {division} · this year"),
        (
            "body",
            "Each crop is judged against its own usual harvest "
            "and its own zone. A higher zone is a stronger fit.",
        ),
    ]
    for name, window, frame in sections:
        blocks.append(("gap", ""))
        blocks.append(("section", f"{name} · {window}"))
        for record in frame.to_dict("records"):
            blocks.append(
                (
                    "body",
                    (
                        f"{record['Crop']} ({record['Zone']}): "
                        f"usual {record['Its usual (t/ha)']:.2f}, "
                        f"estimate {record['Its estimate (t/ha)']:.2f}, "
                        f"vs usual {record['Versus its usual (t/ha)']:+.2f}"
                    ),
                )
            )
    return blocks


def outlook_blocks(
    crop_name: str,
    stress: str,
    note: str,
    districts: pd.DataFrame,
    crops: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Lines for one 2030 stress test."""

    blocks = [
        ("title", "AgriSight-BD 2030 stress test"),
        ("body", f"{stress} · {crop_name}"),
        ("body", note),
        (
            "body",
            "This is a stress test of the current model on the "
            "latest year. It is not a weather forecast.",
        ),
        ("gap", ""),
        ("section", "Districts, largest cushion loss first"),
    ]
    for record in districts.to_dict("records"):
        blocks.append(
            (
                "body",
                (
                    f"{record['District']} ({record['Zone']}): "
                    f"cushion {record['Cushion now (t/ha)']:+.2f}, "
                    f"under stress {record['Cushion under stress (t/ha)']:+.2f}, "
                    f"change {record['Change (t/ha)']:+.2f}"
                ),
            )
        )
    blocks.append(("gap", ""))
    blocks.append(
        ("section", "Crops that keep a strong fit")
    )
    blocks.append(
        (
            "body",
            "A strong zone is Highest-yield or Favorable. "
            "Still holds means the stressed estimate stays at or "
            "above that crop's own usual harvest.",
        )
    )
    for record in crops.to_dict("records"):
        blocks.append(
            (
                "body",
                (
                    f"{record['Crop']}: "
                    f"{int(record['Strong-zone districts'])} strong-zone "
                    f"districts, "
                    f"{int(record['Still at or above usual'])} still hold"
                ),
            )
        )
    return blocks


def season_watch_blocks(
    crop_name: str,
    year: int,
    table: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Lines for districts whose season is already off."""

    blocks = [
        ("title", "AgriSight-BD season watch"),
        ("body", f"{crop_name} · {year}"),
        (
            "body",
            "Rainfall and peak greenness compared with each "
            "district's own average. Off seasons are listed first.",
        ),
        ("gap", ""),
        ("section", "Districts"),
    ]
    for record in table.to_dict("records"):
        blocks.append(
            (
                "body",
                (
                    f"{record['District']} ({record['Season']}): "
                    f"rain {record['Rainfall (mm)']:.0f} mm "
                    f"({record['Rain vs usual (%)']:+.1f}%), "
                    f"greenness {record['Greenness']:.2f} "
                    f"({record['Greenness vs usual']:+.2f})"
                ),
            )
        )
    return blocks


def _file_slug(text: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in text
    ).strip("_")


def open_page(page_name: str) -> None:
    """Switch the sidebar page from a button callback."""

    st.session_state.page = page_name


def open_selected_district(crop_id: str, picker_key: str) -> None:
    """Open the district profile for the attention-list choice."""

    st.session_state.page = "District Profile"
    st.session_state[f"profile_district_{crop_id}"] = (
        st.session_state[picker_key]
    )


def remembered_select(
    label: str,
    options: list[str],
    key: str,
    default: str,
    holder=st,
) -> str:
    """Select box that keeps its choice when the page is left."""

    saved_key = f"{key}_saved"
    saved = st.session_state.get(saved_key, default)
    if saved not in options:
        saved = options[0]
    kwargs = {"key": key}
    if key not in st.session_state:
        kwargs["index"] = options.index(saved)
    choice = holder.selectbox(label, options, **kwargs)
    st.session_state[saved_key] = choice
    return choice


def module_card(
    image_path: Path,
    title: str,
    description: str,
    page_name: str,
) -> None:
    """Render one dashboard card with a real project image."""

    with st.container(border=True):
        if image_path.exists():
            st.image(
                str(image_path),
                use_container_width=True,
            )
        else:
            st.warning(
                f"Image file not found: {image_path.name}"
            )

        st.subheader(title)
        st.write(description)

        st.button(
            f"Open {title}",
            key=f"open_{page_name}",
            type="primary",
            width="stretch",
            on_click=open_page,
            args=(page_name,),
        )


def render_dashboard(
    merged: pd.DataFrame,
    districts: pd.DataFrame,
    crop_name: str,
) -> None:
    """Render the AgriSight-BD landing dashboard."""

    page_header(
        "Planning overview",
        f"Plan {crop_name} with district climate, soil, and "
        "harvest history for Bangladesh. Choose a crop in the "
        "sidebar, then open a tool below.",
    )
    account = signed_in_user()
    if account is None:
        st.caption(
            "Guest access is open. Sign in to save the districts "
            "you cover, visit notes, and scenario runs."
        )
    else:
        covered = len(account["districts"])
        district_word = "district" if covered == 1 else "districts"
        st.caption(
            f"Signed in as {account['name']}. "
            f"{covered} {district_word} saved on this account."
        )

    year_start = int(merged["year"].min())
    year_end = int(merged["year"].max())
    mean_yield = float(merged["yield_ton_ha"].mean())
    district_count = int(merged["district"].nunique())

    st.markdown(
        f"""
        <div class="kpi-row">
            <div><span>Crop</span><strong>{crop_name}</strong></div>
            <div><span>Districts</span><strong>{district_count}</strong></div>
            <div><span>Years</span><strong>{year_start}–{year_end}</strong></div>
            <div>
                <span>Typical yield</span>
                <strong>{mean_yield:.2f} t/ha</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Features")

    module1, module2, module3 = st.columns(3, gap="medium")

    with module1:
        module_card(
            PREDICTION_IMAGE,
            "Prediction Playground",
            f"Estimate {crop_name} yield for one district. "
            "The number updates as you change rainfall, "
            "temperature, or crop greenness.",
            "Prediction Playground",
        )

    with module2:
        module_card(
            WHAT_IF_IMAGE,
            "What-If Analysis",
            f"See how drought, flood, or a heatwave would "
            f"move {crop_name} yield versus the usual harvest.",
            "What-If Analysis",
        )

    with module3:
        module_card(
            SUITABILITY_IMAGE,
            "Crop Suitability",
            "See which districts share similar soil and "
            f"climate, and how {crop_name} has yielded there.",
            "Crop Suitability",
        )

    st.subheader("Field work")

    visit_col, brief_col = st.columns(2, gap="medium")
    with visit_col:
        with st.container(border=True):
            st.markdown("**Where to go**")
            st.write(
                f"See which districts have the smallest gap versus "
                f"the usual {crop_name} harvest, and which shock cuts them most."
            )
            st.button(
                "Open Attention List",
                key="open_attention_dash",
                type="primary",
                width="stretch",
                on_click=open_page,
                args=("Attention List",),
            )
    with brief_col:
        with st.container(border=True):
            st.markdown("**One district**")
            st.write(
                f"Read the {crop_name} briefing for a visit: zone, "
                "soil, shocks, other crops that fit, and a PDF."
            )
            st.button(
                "Open District Profile",
                key="open_profile_dash",
                width="stretch",
                on_click=open_page,
                args=("District Profile",),
            )

    watch_col, plan_col, outlook_col = st.columns(3, gap="medium")
    with watch_col:
        with st.container(border=True):
            st.markdown("**This season**")
            st.write(
                f"See where {crop_name} rainfall or greenness is "
                "already off that district’s own average."
            )
            st.button(
                "Open Season Watch",
                key="open_watch_dash",
                width="stretch",
                on_click=open_page,
                args=("Season Watch",),
            )
    with plan_col:
        with st.container(border=True):
            st.markdown("**The year**")
            st.write(
                "Lay out rabi, kharif, and year-round crops for "
                "one district, each against its own usual harvest."
            )
            st.button(
                "Open Season Plan",
                key="open_plan_dash",
                width="stretch",
                on_click=open_page,
                args=("Season Plan",),
            )
    with outlook_col:
        with st.container(border=True):
            st.markdown("**2030 stress test**")
            st.write(
                f"See which districts lose the most {crop_name} "
                "cushion if the latest year is warmer, drier, or wetter."
            )
            st.button(
                "Open Climate Outlook",
                key="open_outlook_dash",
                width="stretch",
                on_click=open_page,
                args=("Climate Outlook",),
            )

    st.subheader("What the tools use")

    scope1, scope2 = st.columns(2)

    with scope1:
        with st.container(border=True):
            st.markdown("**Inputs**")
            st.markdown(
                """
                - Climate: temperature, rainfall, humidity, sun
                - Vegetation: satellite greenness (NDVI)
                - Soil: pH, organic carbon, nitrogen, clay, sand
                - Harvests for 17 field crops, 64 districts
                """
            )

    with scope2:
        with st.container(border=True):
            st.markdown("**What you get**")
            st.markdown(
                """
                - A one-district briefing you can download
                - A year plan of rabi, kharif, and year-round crops
                - A season watch for rainfall and greenness
                - A 2030 stress test: warmer, drier, or wetter
                - Drought, flood, and heatwave comparisons
                - A country map of suitability zones
                """
            )


def _slider_bounds(
    series: pd.Series,
    pad_low: float,
    pad_high: float,
    step: float,
) -> tuple[float, float]:
    """Pad observed values so the slider can move past them."""

    low = float(series.min()) - pad_low
    high = float(series.max()) + pad_high
    low = round(low / step) * step
    high = round(high / step) * step
    if high <= low:
        high = low + step
    return low, high


def tab_prediction_playground(
    merged,
    districts,
    model,
    scaler,
    features,
    crop_id: str,
    crop_name: str,
):
    page_header(
        f"Prediction Playground · {crop_name}",
        "Choose a district. The estimate uses that district’s "
        "latest year, then follows the sliders.",
    )

    result_slot = st.container()
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        with st.container(border=True):
            district = st.selectbox(
                "District",
                sorted(districts["district"].unique()),
                key=f"predict_district_{crop_id}",
            )

            base = get_district_features(merged, district)

            rain_min, rain_max = _slider_bounds(
                merged["rainfall_total"],
                pad_low=20,
                pad_high=40,
                step=5.0,
            )
            tmin_min, tmin_max = _slider_bounds(
                merged["temp_min_mean"],
                pad_low=2,
                pad_high=2,
                step=0.5,
            )
            tmax_min, tmax_max = _slider_bounds(
                merged["temp_max_mean"],
                pad_low=2,
                pad_high=4,
                step=0.5,
            )
            ndvi_min, ndvi_max = _slider_bounds(
                merged["ndvi_peak"],
                pad_low=0.05,
                pad_high=0.05,
                step=0.01,
            )

            rain_key = f"predict_rain_{crop_id}"
            tmin_key = f"predict_tmin_{crop_id}"
            tmax_key = f"predict_tmax_{crop_id}"
            ndvi_key = f"predict_ndvi_{crop_id}"
            marker = f"predict_for_{crop_id}"

            if st.session_state.get(marker) != district:
                st.session_state[rain_key] = _snap(
                    base.get("rainfall_total", rain_min),
                    rain_min,
                    rain_max,
                    5.0,
                )
                st.session_state[tmin_key] = _snap(
                    base.get("temp_min_mean", tmin_min),
                    tmin_min,
                    tmin_max,
                    0.5,
                )
                st.session_state[tmax_key] = _snap(
                    base.get("temp_max_mean", tmax_min),
                    tmax_min,
                    tmax_max,
                    0.5,
                )
                st.session_state[ndvi_key] = _snap(
                    base.get("ndvi_peak", ndvi_min),
                    ndvi_min,
                    ndvi_max,
                    0.01,
                )
                st.session_state[marker] = district

            rainfall = st.slider(
                "Rainfall (mm)",
                min_value=rain_min,
                max_value=rain_max,
                step=5.0,
                key=rain_key,
                help="Season rainfall. Starts from this district’s latest year.",
            )
            temp_min = st.slider(
                "Coolest temperature (°C)",
                min_value=tmin_min,
                max_value=tmin_max,
                step=0.5,
                key=tmin_key,
                help="Average daily minimum temperature.",
            )
            temp_max = st.slider(
                "Hottest temperature (°C)",
                min_value=tmax_min,
                max_value=tmax_max,
                step=0.5,
                key=tmax_key,
                help="Average daily maximum temperature.",
            )
            ndvi = st.slider(
                "Crop greenness (NDVI)",
                min_value=ndvi_min,
                max_value=ndvi_max,
                step=0.01,
                key=ndvi_key,
                help="Peak satellite greenness. Higher usually means a fuller crop canopy.",
            )

            if st.button(
                "Reset to this district",
                width="stretch",
            ):
                st.session_state[marker] = None
                st.rerun()

        rain_base = float(
            base.get("rainfall_total", rainfall)
        ) or 1.0
        season_share = float(
            base.get("rainfall_mar_jun", rain_base * 0.4)
        ) / rain_base
        temp_gap = float(
            base.get("temp_max_mar_jun", temp_max)
        ) - float(
            base.get("temp_max_mean", temp_max)
        )

        overrides = {
            "rainfall_total": rainfall,
            "rainfall_mar_jun": rainfall * season_share,
            "temp_min_mean": temp_min,
            "temp_max_mean": temp_max,
            "temp_max_mar_jun": temp_max + temp_gap,
            "ndvi_peak": ndvi,
            "ndvi_mean": max(ndvi - 0.12, 0.05),
        }

        prediction = predict_yield(
            model,
            scaler,
            features,
            base,
            overrides,
        )
        historical_average = merged[
            merged["district"] == district
        ]["yield_ton_ha"].mean()
        difference = prediction - historical_average
        offer_save_run(
            kind="playground",
            crop_id=crop_id,
            crop_name=crop_name,
            district=district,
            scenario="Slider estimate",
            predicted_yield=float(prediction),
            usual_yield=float(historical_average),
            details={
                "rainfall_mm": round(float(rainfall), 1),
                "temp_min_c": round(float(temp_min), 1),
                "temp_max_c": round(float(temp_max), 1),
                "ndvi": round(float(ndvi), 2),
            },
            key=f"save_playground_{crop_id}",
        )

    with result_slot:
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric(
            "Predicted yield",
            f"{prediction:.2f} t/ha",
        )
        metric2.metric(
            f"Usual {crop_name} yield",
            f"{historical_average:.2f} t/ha",
        )
        metric3.metric(
            "Compared with usual",
            f"{difference:+.2f} t/ha",
        )

    with col2:
        historical = merged[
            merged["district"] == district
        ].sort_values("year")

        figure = px.line(
            historical,
            x="year",
            y="yield_ton_ha",
            title=f"{crop_name} harvest history · {district}",
            labels={
                "yield_ton_ha": "Yield (t/ha)",
                "year": "Year",
            },
            markers=True,
        )
        years = [
            int(year)
            for year in historical["year"].drop_duplicates()
        ]
        figure.update_traces(
            line_color="#1B7A4E",
            marker_color="#1B7A4E",
            hovertemplate=(
                "Year %{x:.0f}<br>"
                "Yield %{y:.2f} t/ha"
                "<extra></extra>"
            ),
        )
        figure.add_hline(
            y=historical["yield_ton_ha"].mean(),
            line_dash="dash",
            line_color="#C47B2B",
            annotation_text="Usual yield",
            annotation_font_color="#163026",
        )
        style_chart(figure)
        label_every_year(figure, years)
        figure.update_layout(height=460)
        st.plotly_chart(
            figure,
            use_container_width=True,
        )


def tab_what_if(
    merged,
    districts,
    model,
    scaler,
    features,
    crop_id: str,
    crop_name: str,
    season: str,
):
    page_header(
        f"What-If Analysis · {crop_name}",
        f"Compare a climate shock with the usual {crop_name} "
        f"harvest. The critical season is {season}.",
    )

    scenario_names = list(SCENARIO_NOTES)

    pick1, pick2 = st.columns(2)
    district = pick1.selectbox(
        "District",
        sorted(districts["district"].unique()),
        key=f"whatif_district_{crop_id}",
    )
    scenario = pick2.selectbox(
        "Climate scenario",
        scenario_names,
        key=f"whatif_scenario_{crop_id}",
    )
    st.caption(SCENARIO_NOTES[scenario])

    base = get_district_features(
        merged,
        district,
    )

    historical_average = merged[
        merged["district"] == district
    ]["yield_ton_ha"].mean()

    overrides = scenario_overrides(base, scenario)

    baseline_prediction = predict_yield(
        model,
        scaler,
        features,
        base,
        {},
    )

    scenario_prediction = predict_yield(
        model,
        scaler,
        features,
        base,
        overrides,
    )

    change_percentage = (
        (
            scenario_prediction
            - historical_average
        )
        / historical_average
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Usual harvest",
        f"{historical_average:.2f} t/ha",
    )

    col2.metric(
        "This scenario",
        f"{scenario_prediction:.2f} t/ha",
        f"{scenario_prediction - historical_average:+.2f} t/ha",
    )

    col3.metric(
        "Change vs usual",
        f"{change_percentage:+.1f}%",
    )
    offer_save_run(
        kind="what_if",
        crop_id=crop_id,
        crop_name=crop_name,
        district=district,
        scenario=scenario,
        predicted_yield=float(scenario_prediction),
        usual_yield=float(historical_average),
        details={
            "scenario": scenario,
            "baseline_yield": round(float(baseline_prediction), 2),
            "change_percent": round(float(change_percentage), 1),
        },
        key=f"save_whatif_{crop_id}",
    )

    labels = [
        "Usual harvest",
        "Model, no shock",
        "This scenario",
    ]
    values = [
        historical_average,
        baseline_prediction,
        scenario_prediction,
    ]
    scenario_color = (
        "#C4533D"
        if scenario_prediction < historical_average
        else "#1B7A4E"
    )
    figure = px.bar(
        x=labels,
        y=values,
        text=[f"{value:.2f}" for value in values],
        labels={"x": "", "y": "Yield (t/ha)"},
        title=f"{district} · {crop_name}",
        color=labels,
        color_discrete_map={
            "Usual harvest": "#8AA090",
            "Model, no shock": "#2F6F9F",
            "This scenario": scenario_color,
        },
    )
    figure.update_traces(
        textposition="outside",
        textfont_color="#163026",
        marker_line_width=0,
    )
    figure.update_layout(
        showlegend=False,
        yaxis_range=[0, max(values) * 1.25],
    )
    style_chart(figure)
    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.info(
        "Gray is the usual average. Blue is the model "
        "with today’s climate. The last bar is the scenario "
        "you picked: green when it is at or above the usual "
        "harvest, red when it falls short."
    )


def tab_crop_suitability(
    merged,
    districts,
    crop_id: str,
    crop_name: str,
):
    page_header(
        f"Crop Suitability · {crop_name}",
        "Districts in the same color share similar soil and "
        f"climate. Labels follow historical {crop_name} yield. "
        "The map opens on the whole country.",
    )

    n_clusters = st.slider(
        "Number of zones",
        3,
        6,
        4,
        key=f"zones_{crop_id}",
        help="Fewer zones keep the map simple. More zones split similar districts apart.",
    )

    district_average, overall_yield = assign_zones(
        merged,
        districts,
        n_clusters,
    )
    zone_yield = (
        district_average.groupby("zone")["mean_yield"]
        .mean()
        .sort_values(ascending=False)
    )

    zone_order = [
        name
        for name in ZONE_COLORS
        if name in set(district_average["zone_name"])
    ]
    figure = px.scatter_map(
        district_average,
        lat="latitude",
        lon="longitude",
        color="zone_name",
        hover_name="district",
        hover_data={
            "division": True,
            "mean_yield": ":.2f",
            "soil_soc": ":.1f",
            "rainfall_total": ":.0f",
            "latitude": False,
            "longitude": False,
        },
        color_discrete_map=ZONE_COLORS,
        category_orders={"zone_name": zone_order},
        map_style="carto-positron",
        labels={
            "zone_name": "Zone",
            "mean_yield": "Yield (t/ha)",
            "soil_soc": "Soil carbon",
            "rainfall_total": "Rainfall (mm)",
            "division": "Division",
        },
        height=720,
    )
    figure.update_traces(marker=dict(size=12))
    figure.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        legend_title_text="Zone",
        map=bangladesh_map_view(720),
    )
    style_chart(figure, cartesian=False)

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.subheader("What each zone means")

    for zone_id in zone_yield.index:
        zone_rows = district_average[
            district_average["zone"] == zone_id
        ]
        zone_name = zone_rows["zone_name"].iloc[0]
        zone_districts = sorted(zone_rows["district"].tolist())
        zone_mean = float(zone_rows["mean_yield"].mean())
        recommendation = zone_note(
            crop_name,
            zone_mean,
            overall_yield,
        )

        with st.container(border=True):
            st.markdown(
                (
                    "<div style=\"height:6px;border-radius:99px;"
                    f"background:{ZONE_COLORS[zone_name]};"
                    "margin-bottom:8px;\"></div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(f"**{zone_name}**")
            st.caption(
                f"{len(zone_districts)} districts · "
                f"{zone_mean:.2f} t/ha"
            )
            st.write(recommendation)
            with st.expander("Show districts"):
                st.write(", ".join(zone_districts))


def tab_attention_list(
    crop_id: str,
    crop_name: str,
) -> None:
    """Rank districts by the gap versus the usual harvest."""

    page_header(
        f"Attention List · {crop_name}",
        "The smallest gap versus the usual "
        f"{crop_name} harvest is listed first. Open one briefing, "
        "or download the division pack.",
    )

    table = cached_attention(crop_id)
    divisions = ["All divisions"] + sorted(
        table["Division"].unique()
    )
    filter_col, open_col = st.columns([1, 1], gap="medium")
    division_key = f"attention_division_{crop_id}"
    saved_division = st.session_state.get(
        f"attention_division_saved_{crop_id}",
        "All divisions",
    )
    if saved_division not in divisions:
        saved_division = "All divisions"
    division = filter_col.selectbox(
        "Division",
        divisions,
        index=divisions.index(saved_division),
        key=division_key,
    )
    st.session_state[f"attention_division_saved_{crop_id}"] = division
    shown = table
    division_label = "All divisions"
    if division != "All divisions":
        shown = table[table["Division"] == division].reset_index(
            drop=True
        )
        division_label = division

    shown = shown[
        [
            "District",
            "Division",
            "Zone",
            "Vs usual (t/ha)",
            "Worst shock",
            "Shock drop (t/ha)",
            "Usual harvest (t/ha)",
            "Estimate (t/ha)",
        ]
    ]

    account = signed_in_user()
    covered = list(account["districts"]) if account else []
    if covered:
        only_mine = filter_col.checkbox(
            "Only my districts",
            key=f"attention_mine_{crop_id}",
        )
        if only_mine:
            shown = shown[
                shown["District"].isin(set(covered))
            ].reset_index(drop=True)
    elif account is not None:
        filter_col.caption(
            "Save districts in My work to filter this list."
        )

    if shown.empty:
        st.info("No districts match this filter.")
        return

    picker_key = f"attention_open_{crop_id}_{division}"
    open_col.selectbox(
        "Open a briefing",
        shown["District"].tolist(),
        key=picker_key,
    )
    open_col.button(
        "Open District Profile",
        type="primary",
        width="stretch",
        on_click=open_selected_district,
        args=(crop_id, picker_key),
        key=f"attention_open_button_{crop_id}",
    )

    below = int((shown["Vs usual (t/ha)"] < 0).sum())
    st.markdown(
        f"""
        <div class="kpi-row">
            <div><span>Districts</span><strong>{len(shown)}</strong></div>
            <div><span>Below usual harvest</span><strong>{below}</strong></div>
            <div><span>First on the list</span><strong>{shown.iloc[0]['District']}</strong></div>
            <div>
                <span>Its gap</span>
                <strong>{shown.iloc[0]['Vs usual (t/ha)']:+.2f} t/ha</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display = shown.copy()
    display["Zone"] = display["Zone"].str.replace(
        " zone",
        "",
        regex=False,
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Zone": st.column_config.TextColumn(
                "Zone",
                width="small",
            ),
            "Usual harvest (t/ha)": st.column_config.NumberColumn(
                "Usual",
                format="%.2f",
                width="small",
            ),
            "Estimate (t/ha)": st.column_config.NumberColumn(
                "Estimate",
                format="%.2f",
                width="small",
            ),
            "Vs usual (t/ha)": st.column_config.NumberColumn(
                "Vs usual",
                format="%+.2f",
                width="small",
            ),
            "Worst shock": st.column_config.TextColumn(
                "Worst shock",
                width="medium",
            ),
            "Shock drop (t/ha)": st.column_config.NumberColumn(
                "Shock drop",
                format="%+.2f",
                width="small",
            ),
        },
    )
    st.download_button(
        f"Download {division_label} briefing (PDF)",
        data=briefing_pdf(
            division_pack_blocks(
                crop_name,
                division_label,
                shown,
            )
        ),
        file_name=(
            f"{_file_slug(division_label)}_{crop_id}_attention.pdf"
        ),
        mime="application/pdf",
        type="primary",
        key=f"attention_pdf_{crop_id}_{division_label}",
    )


def tab_district_profile(
    merged,
    districts,
    model,
    scaler,
    features,
    crop_id: str,
    crop_name: str,
    season: str,
    metadata: dict,
) -> None:
    """One-district briefing for a visit or a seasonal plan."""

    page_header(
        f"District Profile · {crop_name}",
        "Read one district before a visit or a seasonal plan. "
        "The estimate matches What-If with no climate shock. "
        "The zone matches the suitability map at 4 zones.",
    )

    picker, download_slot = st.columns([2, 1], gap="medium")
    district = picker.selectbox(
        "District",
        sorted(districts["district"].unique()),
        key=f"profile_district_{crop_id}",
    )

    history = (
        merged[merged["district"] == district]
        .sort_values("year")
    )
    latest = history.iloc[-1]
    usual = float(history["yield_ton_ha"].mean())
    estimate = predict_yield(
        model,
        scaler,
        features,
        latest,
        {},
    )
    difference = estimate - usual
    year = int(latest["year"])

    zones, overall_yield = assign_zones(merged, districts, 4)
    place = zones[zones["district"] == district].iloc[0]
    zone_name = place["zone_name"]
    zone_mean = float(
        zones.loc[
            zones["zone"] == place["zone"],
            "mean_yield",
        ].mean()
    )
    note = zone_note(crop_name, zone_mean, overall_yield)

    risks = []
    for label, scenario in PROFILE_RISKS:
        predicted = predict_yield(
            model,
            scaler,
            features,
            latest,
            scenario_overrides(latest, scenario),
        )
        risks.append(
            {
                "Shock": label,
                "Estimated yield (t/ha)": round(predicted, 2),
                "Vs no shock (t/ha)": round(
                    predicted - estimate,
                    2,
                ),
                "Vs usual harvest (t/ha)": round(
                    predicted - usual,
                    2,
                ),
            }
        )
    worst = min(
        risks,
        key=lambda item: item["Vs no shock (t/ha)"],
    )
    shock_drop = worst["Vs no shock (t/ha)"]
    if shock_drop < 0:
        risk_call = (
            f"Largest drop is {worst['Shock'].lower()}: "
            f"{shock_drop:+.2f} t/ha versus the estimate "
            "with no shock."
        )
        if worst["Estimated yield (t/ha)"] < usual:
            risk_call += " That falls short of the usual harvest."
        else:
            risk_call += (
                " It still stays at or above the usual harvest."
            )
    else:
        risk_call = "None of these shocks lowers the estimate."

    country = (
        merged.groupby("district")[
            [
                "soil_ph",
                "soil_soc",
                "soil_nitrogen",
                "soil_clay",
                "ndvi_peak",
                "rainfall_total",
                "temp_max_mean",
                "temp_min_mean",
                "humidity_mean",
                "solar_radiation_mean",
            ]
        ]
        .mean()
        .mean()
    )

    def district_usual(column: str) -> float:
        return float(history[column].mean())

    blocks = [
        ("title", "AgriSight-BD district briefing"),
        ("body", f"{district}, {place['division']}"),
        ("body", f"Crop: {crop_name}"),
        ("body", f"Season: {season}"),
        ("gap", ""),
        ("section", "Harvest"),
        ("body", f"Usual yield: {usual:.2f} t/ha"),
        (
            "body",
            f"Model estimate ({year}): {estimate:.2f} t/ha",
        ),
        ("body", f"Difference: {difference:+.2f} t/ha"),
        ("gap", ""),
        ("section", "Zone"),
        ("body", zone_name),
        ("body", note),
        ("gap", ""),
        ("section", "Soil"),
        (
            "body",
            (
                f"pH {latest['soil_ph']:.1f} "
                f"(country {country['soil_ph']:.1f}), "
                f"organic carbon {latest['soil_soc']:.1f} "
                f"(country {country['soil_soc']:.1f}), "
                f"nitrogen {latest['soil_nitrogen']:.2f} "
                f"(country {country['soil_nitrogen']:.2f}), "
                f"clay {latest['soil_clay']:.1f}% "
                f"(country {country['soil_clay']:.1f}%)"
            ),
        ),
        ("gap", ""),
        (
            "section",
            f"Climate in {year}, versus this district's usual",
        ),
        (
            "body",
            (
                f"Rainfall: {latest['rainfall_total']:.0f} mm "
                f"(usual {district_usual('rainfall_total'):.0f} mm)"
            ),
        ),
        (
            "body",
            (
                f"Hottest temperature: {latest['temp_max_mean']:.1f} C "
                f"(usual {district_usual('temp_max_mean'):.1f} C)"
            ),
        ),
        (
            "body",
            (
                f"Coolest temperature: {latest['temp_min_mean']:.1f} C "
                f"(usual {district_usual('temp_min_mean'):.1f} C)"
            ),
        ),
        (
            "body",
            (
                f"Humidity: {latest['humidity_mean']:.0f}% "
                f"(usual {district_usual('humidity_mean'):.0f}%)"
            ),
        ),
        (
            "body",
            (
                f"Sunlight: {latest['solar_radiation_mean']:.1f} "
                f"(usual {district_usual('solar_radiation_mean'):.1f})"
            ),
        ),
        (
            "body",
            (
                f"Crop greenness: {latest['ndvi_peak']:.2f} "
                f"(usual {district_usual('ndvi_peak'):.2f})"
            ),
        ),
        ("gap", ""),
        (
            "section",
            "Climate shocks versus the usual harvest",
        ),
    ]
    for item in risks:
        blocks.append(
            (
                "body",
                (
                    f"{item['Shock']}: "
                    f"{item['Estimated yield (t/ha)']:.2f} t/ha "
                    f"({item['Vs no shock (t/ha)']:+.2f} vs no shock, "
                    f"{item['Vs usual harvest (t/ha)']:+.2f} vs usual)"
                ),
            )
        )
    blocks.extend(
        [
            ("gap", ""),
            ("body", risk_call),
            ("gap", ""),
            (
                "body",
                (
                    f"Years {int(history['year'].min())}-"
                    f"{int(history['year'].max())}. "
                    f"Model {metadata['best_model']}."
                ),
            ),
        ]
    )
    profile_user = signed_in_user()
    saved_note = ""
    if profile_user is not None:
        saved_note = get_note(profile_user["id"], crop_id, district)
    if saved_note:
        blocks.extend(
            [
                ("gap", ""),
                ("section", "Visit note"),
                ("body", saved_note),
            ]
        )

    file_slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in district
    ).strip("_")
    with download_slot:
        st.write("")
        st.download_button(
            "Download briefing (PDF)",
            data=briefing_pdf(blocks),
            file_name=f"{file_slug}_{crop_id}_briefing.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
            key=f"profile_pdf_{crop_id}_{district}",
        )
        st.caption("One page for a visit or a seasonal plan.")
        division_rows = cached_attention(crop_id)
        division_rows = division_rows[
            division_rows["Division"] == place["division"]
        ]
        st.download_button(
            f"Download {place['division']} pack",
            data=briefing_pdf(
                division_pack_blocks(
                    crop_name,
                    place["division"],
                    division_rows,
                )
            ),
            file_name=(
                f"{_file_slug(place['division'])}_"
                f"{crop_id}_division.pdf"
            ),
            mime="application/pdf",
            width="stretch",
            key=f"division_pdf_{crop_id}_{place['division']}",
        )
        st.caption(
            "Every district in this division, smallest gap first."
        )

    st.markdown(
        f"""
        <div class="kpi-row">
            <div><span>Usual harvest</span><strong>{usual:.2f} t/ha</strong></div>
            <div><span>Model estimate</span><strong>{estimate:.2f} t/ha</strong></div>
            <div><span>Versus usual</span><strong>{difference:+.2f} t/ha</strong></div>
            <div>
                <span>Worst shock</span>
                <strong>{worst['Vs no shock (t/ha)']:+.2f} t/ha</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            (
                "<div style=\"height:6px;border-radius:99px;"
                f"background:{ZONE_COLORS[zone_name]};"
                "margin-bottom:8px;\"></div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{zone_name}** · {place['division']}"
        )
        st.write(note)
        st.write(risk_call)

    render_visit_note(crop_id, crop_name, district)

    st.subheader("Other crops in this district")
    st.caption(
        "Each crop is judged against its own usual harvest and its "
        "own zone. A higher zone is a stronger fit. "
        "Tonnes are not compared across crops."
    )
    with st.spinner("Comparing crops for this district..."):
        fits = crop_fit_table()
    local_fits = fits[fits["District"] == district].sort_values(
        ["zone_rank", "Versus its usual (t/ha)"],
        ascending=[True, False],
    )
    crop_names = local_fits["Crop"].where(
        local_fits["crop_id"] != crop_id,
        local_fits["Crop"] + " · selected",
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Crop": crop_names.to_numpy(),
                "Zone": local_fits["Zone"].to_numpy(),
                "Its usual (t/ha)": local_fits[
                    "Its usual (t/ha)"
                ].to_numpy(),
                "Its estimate (t/ha)": local_fits[
                    "Its estimate (t/ha)"
                ].to_numpy(),
                "Versus its usual (t/ha)": local_fits[
                    "Versus its usual (t/ha)"
                ].to_numpy(),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    map_col, chart_col = st.columns(2, gap="medium")
    with map_col:
        pin = pd.DataFrame(
            [
                {
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "district": district,
                    "zone_name": zone_name,
                    "mean_yield": usual,
                }
            ]
        )
        map_figure = px.scatter_map(
            pin,
            lat="latitude",
            lon="longitude",
            color="zone_name",
            hover_name="district",
            hover_data={
                "mean_yield": ":.2f",
                "latitude": False,
                "longitude": False,
            },
            color_discrete_map=ZONE_COLORS,
            map_style="carto-positron",
            labels={
                "zone_name": "Zone",
                "mean_yield": "Usual yield (t/ha)",
            },
            height=520,
        )
        map_figure.update_traces(marker=dict(size=16))
        map_figure.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            legend_title_text="Zone",
            map=bangladesh_map_view(520),
        )
        style_chart(map_figure, cartesian=False)
        st.plotly_chart(
            map_figure,
            use_container_width=True,
        )

    with chart_col:
        history_figure = px.line(
            history,
            x="year",
            y="yield_ton_ha",
            title=f"{crop_name} harvest history · {district}",
            labels={
                "yield_ton_ha": "Yield (t/ha)",
                "year": "Year",
            },
            markers=True,
        )
        history_years = [
            int(year)
            for year in history["year"].drop_duplicates()
        ]
        history_figure.update_traces(
            line_color="#1B7A4E",
            marker_color="#1B7A4E",
            hovertemplate=(
                "Year %{x:.0f}<br>"
                "Yield %{y:.2f} t/ha"
                "<extra></extra>"
            ),
        )
        history_figure.add_hline(
            y=usual,
            line_dash="dash",
            line_color="#C47B2B",
            annotation_text="Usual yield",
            annotation_font_color="#163026",
        )
        history_figure.update_layout(height=460)
        style_chart(history_figure)
        label_every_year(history_figure, history_years)
        st.plotly_chart(
            history_figure,
            use_container_width=True,
        )

    climate_col, soil_col = st.columns(2, gap="medium")
    with climate_col:
        st.subheader(f"Climate in {year}")
        st.caption(
            "Latest year compared with this district’s own average."
        )
        rain_col, hot_col = st.columns(2)
        cool_col, humid_col = st.columns(2)
        rain_col.metric(
            "Rainfall",
            f"{latest['rainfall_total']:.0f} mm",
            (
                f"{latest['rainfall_total'] - district_usual('rainfall_total'):+.0f}"
                " vs usual"
            ),
            delta_color="off",
        )
        hot_col.metric(
            "Max temp",
            f"{latest['temp_max_mean']:.1f} °C",
            (
                f"{latest['temp_max_mean'] - district_usual('temp_max_mean'):+.1f}"
                " vs usual"
            ),
            delta_color="off",
        )
        cool_col.metric(
            "Min temp",
            f"{latest['temp_min_mean']:.1f} °C",
            (
                f"{latest['temp_min_mean'] - district_usual('temp_min_mean'):+.1f}"
                " vs usual"
            ),
            delta_color="off",
        )
        humid_col.metric(
            "Humidity",
            f"{latest['humidity_mean']:.0f}%",
            (
                f"{latest['humidity_mean'] - district_usual('humidity_mean'):+.0f}"
                " vs usual"
            ),
            delta_color="off",
        )

    with soil_col:
        st.subheader("Soil and crop cover")
        st.caption(
            "Soil is the long-term character of the land. "
            "Greenness is the latest peak NDVI."
        )
        ph_col, carbon_col = st.columns(2)
        nitrogen_col, clay_col = st.columns(2)
        ph_col.metric(
            "Soil pH",
            f"{latest['soil_ph']:.1f}",
            f"country {country['soil_ph']:.1f}",
            delta_color="off",
        )
        carbon_col.metric(
            "Organic carbon",
            f"{latest['soil_soc']:.1f}",
            f"country {country['soil_soc']:.1f}",
            delta_color="off",
        )
        nitrogen_col.metric(
            "Nitrogen",
            f"{latest['soil_nitrogen']:.2f}",
            f"country {country['soil_nitrogen']:.2f}",
            delta_color="off",
        )
        clay_col.metric(
            "Clay",
            f"{latest['soil_clay']:.0f}%",
            f"country {country['soil_clay']:.0f}%",
            delta_color="off",
        )
        st.metric(
            "Crop greenness",
            f"{latest['ndvi_peak']:.2f}",
            (
                f"{latest['ndvi_peak'] - district_usual('ndvi_peak'):+.2f}"
                " vs usual"
            ),
            delta_color="off",
        )

    st.subheader("Climate shocks")
    st.caption(
        "Each shock is applied to the latest year and compared "
        "with the usual harvest."
    )
    st.dataframe(
        pd.DataFrame(risks),
        use_container_width=True,
        hide_index=True,
    )


def _fit_column_config() -> dict:
    return {
        "Zone": st.column_config.TextColumn("Zone", width="small"),
        "Its usual (t/ha)": st.column_config.NumberColumn(
            "Its usual",
            format="%.2f",
            width="small",
        ),
        "Its estimate (t/ha)": st.column_config.NumberColumn(
            "Its estimate",
            format="%.2f",
            width="small",
        ),
        "Versus its usual (t/ha)": st.column_config.NumberColumn(
            "Vs its usual",
            format="%+.2f",
            width="small",
        ),
    }


def tab_season_plan(
    crop_id: str,
    districts: pd.DataFrame,
) -> None:
    """Year plan for one district, crop by crop."""

    page_header(
        "Season Plan",
        "Lay out the year for one district. Rabi, kharif, and "
        "year-round crops are each judged against their own usual "
        "harvest and their own zone. A higher zone is a stronger fit.",
    )
    names = sorted(districts["district"].unique())
    district = remembered_select(
        "District",
        names,
        "season_plan_district",
        "Bagerhat",
    )
    division = districts.loc[
        districts["district"] == district,
        "division",
    ].iloc[0]
    st.caption(
        f"{district}, {division}. "
        "Tonnes are not compared across crops."
    )

    with st.spinner("Building the season plan..."):
        fits = crop_fit_table()
    sections = season_sections(fits, district, crop_id)

    leads = []
    above = 0
    for name, _window, frame in sections:
        if frame.empty:
            continue
        above += int((frame["Versus its usual (t/ha)"] > 0).sum())
        if name == "Year-round":
            continue
        lead = frame.iloc[0]
        leads.append((name, lead["Crop"], lead["Versus its usual (t/ha)"]))
    rabi_name = leads[0][1].replace(" · selected", "")
    rabi_gap = leads[0][2]
    kharif_name = leads[1][1].replace(" · selected", "")
    kharif_gap = leads[1][2]
    st.markdown(
        f"""
        <div class="kpi-row">
            <div><span>Rabi lead</span><strong>{rabi_name}</strong></div>
            <div><span>Its gap</span><strong>{rabi_gap:+.2f} t/ha</strong></div>
            <div><span>Kharif lead</span><strong>{kharif_name}</strong></div>
            <div><span>Its gap</span><strong>{kharif_gap:+.2f} t/ha</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{above} crops in the plan sit above their own usual harvest.")

    for name, window, frame in sections:
        st.subheader(f"{name} · {window}")
        shown = frame.copy()
        shown["Zone"] = shown["Zone"].str.replace(" zone", "", regex=False)
        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True,
            column_config=_fit_column_config(),
        )

    st.download_button(
        f"Download {district} season plan (PDF)",
        data=briefing_pdf(
            season_plan_blocks(district, division, sections)
        ),
        file_name=f"{_file_slug(district)}_season_plan.pdf",
        mime="application/pdf",
        type="primary",
        key=f"season_plan_pdf_{district}",
    )


def _outlook_tables(
    frame: pd.DataFrame,
    crop_id: str,
    stress: str,
    division: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """District cushion losses, and crops that keep a strong zone."""

    crop_rows = frame[frame["crop_id"] == crop_id].copy()
    if division != "All divisions":
        crop_rows = crop_rows[crop_rows["Division"] == division]
    crop_rows["Cushion now (t/ha)"] = (
        crop_rows["Estimate"] - crop_rows["Usual"]
    ).round(2)
    crop_rows["Cushion under stress (t/ha)"] = (
        crop_rows[stress] - crop_rows["Usual"]
    ).round(2)
    crop_rows["Change (t/ha)"] = (
        crop_rows[stress] - crop_rows["Estimate"]
    ).round(2)
    districts = crop_rows.sort_values(
        ["Change (t/ha)", "District"],
        ascending=[True, True],
    ).reset_index(drop=True)
    districts = districts[
        [
            "District",
            "Division",
            "Zone",
            "Cushion now (t/ha)",
            "Cushion under stress (t/ha)",
            "Change (t/ha)",
        ]
    ]

    summary_rows = []
    for crop_name, crop_frame in frame.groupby("Crop", sort=False):
        strong = crop_frame["Zone"].isin(STRONG_ZONES)
        holds = strong & (crop_frame[stress] >= crop_frame["Usual"])
        summary_rows.append(
            {
                "Crop": crop_name,
                "Strong-zone districts": int(strong.sum()),
                "Still at or above usual": int(holds.sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["Still at or above usual", "Strong-zone districts", "Crop"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return districts, summary


def tab_climate_outlook(crop_id: str, crop_name: str) -> None:
    """2030 stress test of the latest year."""

    page_header(
        f"Climate Outlook · {crop_name}",
        "A stress test of the latest year. Warmer, drier, or wetter. "
        "This is not a weather forecast. Districts that lose the most "
        "of their cushion versus the usual harvest are listed first.",
    )
    with st.spinner("Running the 2030 stress test..."):
        frame = outlook_frame()

    stresses = [label for label, _effect, _note in OUTLOOK_STRESSES]
    notes = {label: note for label, _effect, note in OUTLOOK_STRESSES}
    filter_col, stress_col = st.columns([1, 1], gap="medium")
    divisions = ["All divisions"] + sorted(frame["Division"].unique())
    division = remembered_select(
        "Division",
        divisions,
        f"outlook_division_{crop_id}",
        "All divisions",
        holder=filter_col,
    )
    stress = remembered_select(
        "2030 stress",
        stresses,
        f"outlook_stress_{crop_id}",
        "Warmer",
        holder=stress_col,
    )
    st.caption(notes[stress])

    districts, summary = _outlook_tables(
        frame,
        crop_id,
        stress,
        division,
    )
    losing = int((districts["Change (t/ha)"] < 0).sum())
    lead = districts.iloc[0]
    holding = int((summary["Still at or above usual"] > 0).sum())
    st.markdown(
        f"""
        <div class="kpi-row">
            <div><span>Districts</span><strong>{len(districts)}</strong></div>
            <div><span>Cushion shrinks</span><strong>{losing}</strong></div>
            <div><span>Largest loss</span><strong>{lead['District']}</strong></div>
            <div>
                <span>Its change</span>
                <strong>{lead['Change (t/ha)']:+.2f} t/ha</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    shown = districts.copy()
    shown["Zone"] = shown["Zone"].str.replace(" zone", "", regex=False)
    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Zone": st.column_config.TextColumn("Zone", width="small"),
            "Cushion now (t/ha)": st.column_config.NumberColumn(
                "Cushion now",
                format="%+.2f",
                width="small",
            ),
            "Cushion under stress (t/ha)": st.column_config.NumberColumn(
                "Under stress",
                format="%+.2f",
                width="small",
            ),
            "Change (t/ha)": st.column_config.NumberColumn(
                "Change",
                format="%+.2f",
                width="small",
            ),
        },
    )

    st.subheader("Crops that keep a strong fit")
    st.caption(
        f"Whole country, under {stress.lower()}. A strong zone is "
        "Highest-yield or Favorable. Still holds means the stressed "
        "estimate stays at or above that crop’s own usual harvest. "
        f"{holding} crops still hold in at least one strong-zone district."
    )
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )
    division_label = division
    st.download_button(
        f"Download {stress.lower()} outlook (PDF)",
        data=briefing_pdf(
            outlook_blocks(
                crop_name,
                stress,
                notes[stress],
                districts,
                summary,
            )
        ),
        file_name=(
            f"{_file_slug(division_label)}_{crop_id}_"
            f"{_file_slug(stress)}_outlook.pdf"
        ),
        mime="application/pdf",
        type="primary",
        key=f"outlook_pdf_{crop_id}_{stress}_{division_label}",
    )


def tab_season_watch(
    merged: pd.DataFrame,
    crop_name: str,
) -> None:
    """Districts whose latest season is already off their own average."""

    table = season_watch_table(merged)
    year = int(
        merged.sort_values("year").groupby("district").tail(1)["year"].iloc[0]
    )
    page_header(
        f"Season Watch · {crop_name}",
        f"Rainfall and peak greenness in {year}, compared with each "
        "district’s own average. A season is off when rainfall is below "
        "that average, or greenness is at least 0.03 below it. "
        "Off seasons are listed first.",
    )
    off = table[table["Season"] != "Near usual"]
    lead = table.iloc[0]
    st.markdown(
        f"""
        <div class="kpi-row">
            <div><span>Districts off</span><strong>{len(off)}</strong></div>
            <div><span>First on the list</span><strong>{lead['District']}</strong></div>
            <div>
                <span>Its greenness</span>
                <strong>{lead['Greenness vs usual']:+.2f}</strong>
            </div>
            <div>
                <span>Its rainfall</span>
                <strong>{lead['Rain vs usual (%)']:+.1f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rainfall (mm)": st.column_config.NumberColumn(
                "Rainfall (mm)",
                format="%.0f",
                width="small",
            ),
            "Rain vs usual (%)": st.column_config.NumberColumn(
                "Rain vs usual",
                format="%+.1f%%",
                width="small",
            ),
            "Greenness": st.column_config.NumberColumn(
                "Greenness",
                format="%.2f",
                width="small",
            ),
            "Greenness vs usual": st.column_config.NumberColumn(
                "Greenness vs usual",
                format="%+.2f",
                width="small",
            ),
        },
    )
    st.download_button(
        f"Download {year} season watch (PDF)",
        data=briefing_pdf(
            season_watch_blocks(crop_name, year, table)
        ),
        file_name=f"{crop_name.lower().replace(' ', '_')}_{year}_season_watch.pdf",
        mime="application/pdf",
        type="primary",
        key=f"season_watch_pdf_{year}",
    )


def tab_crop_datasets(active_crop_id: str) -> None:
    """List every crop dataset and offer a CSV download."""

    page_header(
        "Crop Datasets",
        "Every crop covers all 64 districts from 2004 through 2026. "
        "The sidebar crop is the one used for prediction. "
        "Download that file, another crop, or all of them.",
    )

    catalog = load_crop_catalog()
    table = catalog.drop(columns=["crop_id"]).copy()
    table["Season"] = table["Season"].str.split(",").str[0]
    table = table.rename(columns={"Mean yield (t/ha)": "Mean t/ha"})

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rows": st.column_config.NumberColumn(width="small"),
            "Districts": st.column_config.NumberColumn(width="small"),
            "Years": st.column_config.TextColumn(width="small"),
            "Mean t/ha": st.column_config.NumberColumn(
                format="%.2f",
                width="small",
            ),
        },
    )

    download_col, zip_col = st.columns(2)
    crop_ids = list(CROP_CATALOG)
    with download_col:
        with st.container(border=True):
            choice = st.selectbox(
                "Download one crop",
                crop_ids,
                index=crop_ids.index(active_crop_id),
                format_func=crop_label,
                key="dataset_download_choice",
            )
            summary = catalog.loc[
                catalog["crop_id"] == choice
            ].iloc[0]
            spec = CROP_CATALOG[choice]
            st.caption(
                f"{spec['group']} · {spec['season']} · "
                f"{int(summary['Rows'])} rows · "
                f"{summary['Years']} · "
                f"mean {summary['Mean yield (t/ha)']:.2f} t/ha"
            )
            st.download_button(
                f"Download {spec['label']} CSV",
                data=dataset_bytes(choice),
                file_name=f"{choice}_dataset.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )

    with zip_col:
        with st.container(border=True):
            st.markdown("**All crops**")
            st.caption(
                "One ZIP with a CSV for each of the 17 crops."
            )
            st.download_button(
                "Download all datasets (ZIP)",
                data=all_datasets_zip(),
                file_name="agrisight_crop_datasets.zip",
                mime="application/zip",
                width="stretch",
            )


KIND_LABELS = {
    "playground": "Prediction",
    "what_if": "What-if",
}


def signed_in_user():
    """Return the active account for this browser session."""

    user_id = st.session_state.get("user_id")
    if not user_id:
        return None
    user = get_user(int(user_id))
    if user is None or not user["active"]:
        st.session_state.pop("user_id", None)
        return None
    return user


def sign_out() -> None:
    """End the session and return to the public dashboard."""

    st.session_state.pop("user_id", None)
    st.session_state.page = "Dashboard"
    st.session_state.admin_console = False


def _enter_console() -> None:
    """Leave the field planner and open the operations console."""

    st.session_state.admin_console = True


def _stamp(value: str) -> str:
    return value[:16].replace("T", " ") + " UTC"


def _flash() -> None:
    message = st.session_state.pop("account_flash", None)
    if message:
        st.success(message)


def _run_sentence(run: dict) -> str:
    details = run["details"]
    if run["kind"] == "playground":
        return (
            f"Rain {details.get('rainfall_mm')} mm, "
            f"coolest {details.get('temp_min_c')} C, "
            f"hottest {details.get('temp_max_c')} C, "
            f"greenness {details.get('ndvi')}."
        )
    baseline = details.get("baseline_yield")
    if baseline is None:
        return run["scenario"]
    return (
        f"{run['scenario']}. "
        f"Model with no shock: {baseline} t/ha."
    )


def offer_save_run(
    kind: str,
    crop_id: str,
    crop_name: str,
    district: str,
    scenario: str,
    predicted_yield: float,
    usual_yield: float,
    details: dict,
    key: str,
) -> None:
    """Save a playground or what-if result for the signed-in person."""

    user = signed_in_user()
    if user is None:
        st.caption("Sign in to keep this run in My work.")
        return
    if st.button("Save this run", key=key, width="stretch"):
        error = save_run(
            user["id"],
            kind,
            crop_id,
            crop_name,
            district,
            scenario,
            predicted_yield,
            usual_yield,
            details,
        )
        if error:
            st.error(error)
        else:
            st.success("Saved to My work.")


def render_visit_note(
    crop_id: str,
    crop_name: str,
    district: str,
) -> None:
    """Personal note on one district. Guests still see the briefing."""

    st.subheader("Visit note")
    user = signed_in_user()
    if user is None:
        st.caption(
            "Sign in to keep a note for this district. "
            "The briefing and PDF stay available either way."
        )
        return
    _flash()
    note_nonce = int(
        st.session_state.get(f"note_form_{crop_id}_{district}", 0)
    )
    with st.form(f"visit_note_{user['id']}_{crop_id}_{district}_{note_nonce}"):
        body = st.text_area(
            f"Note for {district} · {crop_name}",
            value=get_note(user["id"], crop_id, district),
            height=120,
            placeholder="What you saw on the visit, and what to follow up.",
        )
        st.caption(
            "Save the note, then download the briefing again to include it."
        )
        submitted = st.form_submit_button("Save note")
    if submitted:
        save_note(user["id"], crop_id, district, body)
        st.session_state[f"note_form_{crop_id}_{district}"] = (
            note_nonce + 1
        )
        if str(body).strip():
            st.session_state.account_flash = "Note saved to My work."
        else:
            st.session_state.account_flash = "Note removed."
        st.rerun()


def render_account_sidebar(user) -> None:
    """Identity block at the top of the sidebar."""

    st.sidebar.markdown(
        '<div class="nav-label top">Account</div>',
        unsafe_allow_html=True,
    )
    if user is None:
        st.sidebar.button(
            "Sign in or create account",
            width="stretch",
            key="goto_sign_in",
            on_click=open_page,
            args=("Sign in",),
        )
        st.sidebar.markdown(
            """
            <div class="account-pill">
                <span>Guest</span>
                <p>Maps and estimates stay open.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    organization = user["organization"] or "No organization"
    role = ROLE_LABELS.get(user["role"], user["role"])
    st.sidebar.markdown(
        f"""
        <div class="account-pill">
            <span>{html.escape(user["name"])}</span>
            <p>{html.escape(role)} · {html.escape(organization)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if user["role"] == "admin":
        st.sidebar.button(
            "Operations console",
            width="stretch",
            key="goto_console",
            on_click=_enter_console,
        )
    st.sidebar.button(
        "My work",
        width="stretch",
        key="goto_my_work",
        on_click=open_page,
        args=("My work",),
    )
    st.sidebar.button(
        "Sign out",
        width="stretch",
        key="sign_out",
        on_click=sign_out,
    )


NAV_GROUPS = [
    ("Overview", ["Dashboard"]),
    (
        "Field work",
        [
            "Attention List",
            "Season Watch",
            "Season Plan",
            "Climate Outlook",
            "District Profile",
        ],
    ),
    (
        "Analysis",
        [
            "Prediction Playground",
            "What-If Analysis",
            "Crop Suitability",
            "Crop Datasets",
        ],
    ),
]


def render_nav(pages: list[str]) -> str:
    """Sidebar navigation shared by every page."""

    current = st.session_state.get("page", "Dashboard")
    if current not in pages:
        current = "Dashboard"
        st.session_state.page = current

    for label, items in NAV_GROUPS:
        visible = [item for item in items if item in pages]
        if not visible:
            continue
        st.sidebar.markdown(
            f'<div class="nav-label">{label}</div>',
            unsafe_allow_html=True,
        )
        for item in visible:
            if st.sidebar.button(
                item,
                key=f"nav_{item}",
                type="primary" if item == current else "secondary",
                width="stretch",
            ):
                st.session_state.page = item
                st.rerun()
    return st.session_state.get("page", "Dashboard")


def render_sign_in(district_names: list[str]) -> None:
    """Sign in first. Create account opens underneath, like a login page."""

    st.markdown('<div class="auth-stack"></div>', unsafe_allow_html=True)
    page_header(
        "Account",
        "Sign in to save the districts you cover, visit notes, and "
        "scenario runs. Maps and estimates stay open without an account.",
    )
    _flash()
    if st.session_state.get("auth_panel") == "create":
        _render_create_account(district_names)
    else:
        _render_sign_in_panel()


def _render_sign_in_panel() -> None:
    with st.form("sign_in_form"):
        st.markdown(
            """
            <div class="auth-head">
                <h2>Sign in</h2>
                <p>Field user: user / user1234. Admin: admin / admin123.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        email = st.text_input(
            "Email or username",
            key="sign_in_email",
        )
        password = st.text_input(
            "Password",
            type="password",
            key="sign_in_password",
        )
        sign_submitted = st.form_submit_button(
            "Sign in",
            type="primary",
            width="stretch",
        )
    if sign_submitted:
        user, error = authenticate(email, password)
        if error:
            st.error(error)
        else:
            st.session_state.user_id = user["id"]
            if user["role"] == "admin":
                st.session_state.admin_console = True
            else:
                st.session_state.admin_console = False
                st.session_state.pending_page = "My work"
            st.rerun()

    st.markdown(
        '<div class="auth-or"><span>or</span></div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Create new account",
        key="open_create_account",
        width="stretch",
    ):
        st.session_state.auth_panel = "create"
        st.rerun()


def _render_create_account(district_names: list[str]) -> None:
    first_account = user_count() == 0
    if first_account:
        st.info("The first account becomes the admin.")
    with st.form("create_account_form"):
        st.markdown(
            """
            <div class="auth-head">
                <h2>Create account</h2>
                <p>An account keeps your districts, visit notes, and scenario runs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        name = st.text_input("Name")
        email = st.text_input("Email", key="create_email")
        password = st.text_input(
            "Password",
            type="password",
            key="create_password",
        )
        confirm = st.text_input(
            "Confirm password",
            type="password",
            key="create_password_confirm",
        )
        organization = st.text_input(
            "Organization",
            placeholder="DAE, an NGO, or a university",
        )
        role = "admin" if first_account else "user"
        if first_account:
            st.caption("This account will be the admin.")
        districts = st.multiselect(
            "Districts you cover",
            district_names,
            help=(
                "Leave this empty to keep the full country list. "
                "Attention List can filter to this set."
            ),
        )
        create_submitted = st.form_submit_button(
            "Create account",
            type="primary",
            width="stretch",
        )
    if create_submitted:
        if password != confirm:
            st.error("Type the same password in both boxes.")
        else:
            created, error = create_account(
                name,
                email,
                password,
                organization,
                role,
                districts,
                district_names,
            )
            if error:
                st.error(error)
            else:
                st.session_state.user_id = created["id"]
                st.session_state.auth_panel = "sign_in"
                if created["role"] == "admin":
                    st.session_state.admin_console = True
                    st.session_state.account_flash = (
                        "Account created. You are signed in."
                    )
                else:
                    st.session_state.pending_page = "My work"
                    st.session_state.account_flash = (
                        "Account created. You are signed in."
                    )
                st.rerun()
    if st.button(
        "Already have an account? Sign in",
        key="back_to_sign_in",
        width="stretch",
    ):
        st.session_state.auth_panel = "sign_in"
        st.rerun()


def render_my_work(user: dict, district_names: list[str]) -> None:
    """Districts, notes, and saved runs for the signed-in person."""

    page_header(
        "My work",
        "Districts you cover, visit notes, and saved scenario runs. "
        "The yield estimate itself stays the same for every account.",
    )
    _flash()
    st.caption(
        f"{user['email']} · {ROLE_LABELS.get(user['role'], user['role'])}"
    )

    if st.button("Edit profile", key="open_profile_form"):
        st.session_state.show_profile_form = True
    if st.session_state.get("show_profile_form"):
        st.caption("Update your name, organization, or the districts you cover.")
        profile_nonce = int(st.session_state.get("profile_nonce", 0))
        with st.form(f"profile_form_{user['id']}_{profile_nonce}"):
            name = st.text_input("Name", value=user["name"])
            organization = st.text_input(
                "Organization",
                value=user["organization"],
            )
            saved_districts = [
                district
                for district in user["districts"]
                if district in district_names
            ]
            districts = st.multiselect(
                "Districts you cover",
                district_names,
                default=saved_districts,
            )
            profile_submitted = st.form_submit_button(
                "Save profile",
                type="primary",
            )
        if profile_submitted:
            error = update_profile(
                user["id"],
                name,
                organization,
                districts,
                district_names,
            )
            if error:
                st.error(error)
            else:
                st.session_state.profile_nonce = profile_nonce + 1
                st.session_state.show_profile_form = False
                st.session_state.account_flash = "Profile saved."
                st.rerun()

    if st.button("Change password", key="open_password_form"):
        st.session_state.show_password_form = True
    if st.session_state.get("show_password_form"):
        st.caption("Use this only when you want a new password.")
        password_nonce = int(st.session_state.get("password_form", 0))
        with st.form(f"password_form_{user['id']}_{password_nonce}"):
            current = st.text_input("Current password", type="password")
            new_password = st.text_input("New password", type="password")
            confirm = st.text_input("Confirm new password", type="password")
            password_submitted = st.form_submit_button("Update password")
        if password_submitted:
            if new_password != confirm:
                st.error("Type the same password in both boxes.")
            else:
                error = change_password(user["id"], current, new_password)
                if error:
                    st.error(error)
                else:
                    st.session_state.password_form = password_nonce + 1
                    st.session_state.show_password_form = False
                    st.session_state.account_flash = "Password updated."
                    st.rerun()

    st.subheader("Saved runs")
    runs = list_runs(user["id"])
    if not runs:
        st.caption(
            "Save a run from Prediction Playground or What-If Analysis."
        )
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "When": _stamp(run["created_at"]),
                        "Tool": KIND_LABELS.get(run["kind"], run["kind"]),
                        "Crop": run["crop_name"],
                        "District": run["district"],
                        "Scenario": run["scenario"],
                        "Predicted (t/ha)": round(
                            run["predicted_yield"],
                            2,
                        ),
                        "Usual (t/ha)": round(run["usual_yield"], 2),
                    }
                    for run in runs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        run_nonce = int(st.session_state.get("run_list_nonce", 0))
        labels = [
            (
                f"{_stamp(run['created_at'])} · {run['district']} · "
                f"{run['crop_name']} · {run['scenario']} · #{run['id']}"
            )
            for run in runs
        ]
        choice = st.selectbox(
            "Saved run",
            labels,
            key=f"saved_run_choice_{run_nonce}",
        )
        st.write(_run_sentence(runs[labels.index(choice)]))
        if st.button("Delete this run", key="delete_saved_run"):
            delete_run(user["id"], runs[labels.index(choice)]["id"])
            st.session_state.run_list_nonce = run_nonce + 1
            st.session_state.account_flash = "Saved run deleted."
            st.rerun()

    st.subheader("Visit notes")
    notes = list_notes(user["id"])
    if not notes:
        st.caption("Notes you save on a district profile show up here.")
    for note in notes:
        crop_name = CROP_CATALOG.get(note["crop_id"], {}).get(
            "label",
            note["crop_id"],
        )
        with st.container(border=True):
            st.markdown(f"**{note['district']}** · {crop_name}")
            st.caption(_stamp(note["updated_at"]))
            st.write(note["body"])
            if st.button(
                "Delete note",
                key=f"delete_note_{note['id']}",
            ):
                delete_note(
                    user["id"],
                    note["crop_id"],
                    note["district"],
                )
                st.session_state.pop(
                    (
                        f"visit_note_{user['id']}_{note['crop_id']}"
                        f"_{note['district']}"
                    ),
                    None,
                )
                st.session_state.account_flash = "Note deleted."
                st.rerun()


def render_users(actor: dict, district_names: list[str]) -> None:
    """Admin screen for roles, districts, and turning an account off."""

    page_header(
        "Users",
        "Set an account as a user or an admin, assign districts, "
        "or turn an account off.",
    )
    _flash()
    people = list_users()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Name": person["name"],
                    "Email": person["email"],
                    "Role": ROLE_LABELS.get(
                        person["role"],
                        person["role"],
                    ),
                    "Organization": person["organization"],
                    "Districts": len(person["districts"]),
                    "Active": "Yes" if person["active"] else "No",
                    "Created": _stamp(person["created_at"]),
                }
                for person in people
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    labels = [
        (
            f"{person['name']} · {person['email']} · "
            f"{ROLE_LABELS.get(person['role'], person['role'])}"
        )
        for person in people
    ]
    choice = st.selectbox("Account", labels, key="admin_account_choice")
    target = people[labels.index(choice)]
    nonce = int(st.session_state.get(f"admin_form_{target['id']}", 0))
    with st.form(f"admin_edit_{target['id']}_{nonce}"):
        role_options = ["user", "admin"]
        role = st.selectbox(
            "Role",
            role_options,
            index=(
                role_options.index(target["role"])
                if target["role"] in role_options
                else 0
            ),
            format_func=lambda item: ROLE_LABELS[item],
        )
        active = st.checkbox(
            "Account is active",
            value=bool(target["active"]),
            disabled=target["id"] == actor["id"],
        )
        current_districts = [
            district
            for district in target["districts"]
            if district in district_names
        ]
        districts = st.multiselect(
            "Districts",
            district_names,
            default=current_districts,
        )
        new_password = st.text_input(
            "New password",
            type="password",
            help="Leave blank to keep the current password.",
        )
        admin_submitted = st.form_submit_button(
            "Save account",
            type="primary",
        )
    if admin_submitted:
        if target["id"] == actor["id"]:
            active = True
        error = admin_update_user(
            actor["id"],
            target["id"],
            role,
            bool(active),
            districts,
            district_names,
            new_password,
        )
        if error:
            st.error(error)
        else:
            st.session_state[f"admin_form_{target['id']}"] = nonce + 1
            st.session_state.account_flash = "Account updated."
            if target["id"] == actor["id"] and role != "admin":
                st.session_state.pending_page = "My work"
            st.rerun()


def main():
    """Run the AgriSight-BD application."""

    ensure_default_admin()
    inject_styles()
    user = signed_in_user()
    if (
        user is not None
        and user["role"] == "admin"
        and st.session_state.get("admin_console", True)
    ):
        from src.app.admin_console import render_admin_app

        render_admin_app(user)
        return

    st.sidebar.markdown(
        """
        <div class="brand">
            <span class="mark">Bangladesh</span>
            <h1>AgriSight</h1>
            <p>Climate-resilient crop planning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_account_sidebar(user)

    st.sidebar.markdown(
        '<div class="side-label">Dataset</div>',
        unsafe_allow_html=True,
    )

    crop_ids = list(CROP_CATALOG)
    crop_id = st.sidebar.selectbox(
        "Crop",
        crop_ids,
        index=crop_ids.index(PRIMARY_CROP),
        format_func=crop_label,
        help="Charts, estimates, and the map all follow this crop.",
    )
    spec = CROP_CATALOG[crop_id]
    name = spec["label"]

    st.sidebar.download_button(
        f"Download {name} data",
        data=dataset_bytes(crop_id),
        file_name=f"{crop_id}_dataset.csv",
        mime="text/csv",
        key=f"sidebar_download_{crop_id}",
        width="stretch",
    )

    try:
        merged, districts, metadata = load_data(crop_id)
        model, scaler, features = load_model(crop_id)

    except FileNotFoundError:
        st.error(
            f"Dataset or model for {name} was not found. "
            "Build the crop datasets and train the models first."
        )
        st.stop()

    district_names = sorted(
        districts["district"].astype(str).unique()
    )
    pages = [
        "Dashboard",
        "Attention List",
        "Season Watch",
        "Season Plan",
        "Climate Outlook",
        "District Profile",
        "Prediction Playground",
        "What-If Analysis",
        "Crop Suitability",
        "Crop Datasets",
    ]
    if user is None:
        pages.append("Sign in")
    else:
        pages.append("My work")

    pending = st.session_state.pop("pending_page", None)
    if pending in pages:
        st.session_state.page = pending
    elif (
        "page" in st.session_state
        and st.session_state.page not in pages
    ):
        st.session_state.page = "Dashboard"

    page = render_nav(pages)

    st.sidebar.markdown(
        '<div class="side-label">This model</div>',
        unsafe_allow_html=True,
    )
    year_start, year_end = metadata["year_range"]
    st.sidebar.markdown(
        f"""
        <div class="stat-grid">
            <div><span>Model</span><strong>{metadata['best_model']}</strong></div>
            <div><span>Rows</span><strong>{metadata['n_samples']}</strong></div>
            <div><span>Years</span><strong>{year_start}–{year_end}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: NASA POWER climate · ISRIC soil · "
        "MODIS greenness · BBS agricultural statistics"
    )

    if page == "Dashboard":
        render_dashboard(
            merged,
            districts,
            name,
        )

    elif page == "Attention List":
        tab_attention_list(crop_id, name)

    elif page == "Season Watch":
        tab_season_watch(merged, name)

    elif page == "Season Plan":
        tab_season_plan(crop_id, districts)

    elif page == "Climate Outlook":
        tab_climate_outlook(crop_id, name)

    elif page == "District Profile":
        tab_district_profile(
            merged,
            districts,
            model,
            scaler,
            features,
            crop_id,
            name,
            spec["season"],
            metadata,
        )

    elif page == "Prediction Playground":
        tab_prediction_playground(
            merged,
            districts,
            model,
            scaler,
            features,
            crop_id,
            name,
        )

    elif page == "What-If Analysis":
        tab_what_if(
            merged,
            districts,
            model,
            scaler,
            features,
            crop_id,
            name,
            spec["season"],
        )

    elif page == "Crop Suitability":
        tab_crop_suitability(
            merged,
            districts,
            crop_id,
            name,
        )

    elif page == "Crop Datasets":
        tab_crop_datasets(crop_id)

    elif page == "Sign in":
        render_sign_in(district_names)

    elif page == "My work" and user is not None:
        render_my_work(user, district_names)

    elif page == "Users" and user is not None and user["role"] == "admin":
        render_users(user, district_names)

    else:
        render_sign_in(district_names)


if __name__ == "__main__":
    main()