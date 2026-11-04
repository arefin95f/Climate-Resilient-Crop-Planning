"""Central configuration for AgriSight-BD."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CROP_DATA_DIR = DATA_PROCESSED / "crops"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = DATA_PROCESSED / "model_artifacts"

# Major Bangladesh field crops. Boro, Aman, and maize keep the
# yield records already shipped in data/raw/bbs_yield.csv.
# Every other crop is a reproducible sample built from the same
# district climate and soil so it can be predicted and downloaded.
CROP_CATALOG = {
    "boro_rice": {
        "label": "Boro rice",
        "group": "Cereal",
        "season": "November–May, grain filling March–June",
        "yield_source": "bbs",
    },
    "aman_rice": {
        "label": "Aman rice",
        "group": "Cereal",
        "season": "July–December, grain filling July–October",
        "yield_source": "bbs",
        "season_rain_share": 0.62,
        "season_temp_offset": -1.2,
        "ndvi_shift": 0.02,
    },
    "aus_rice": {
        "label": "Aus rice",
        "group": "Cereal",
        "season": "April–August",
        "yield_source": "sample",
        "base": 2.35,
        "rain_coef": 0.18,
        "temp_coef": -0.07,
        "soil_coef": 0.10,
        "trend": 0.016,
        "noise": 0.10,
        "floor": 1.15,
        "season_rain_share": 0.50,
        "season_temp_offset": 0.5,
        "ndvi_shift": -0.04,
    },
    "wheat": {
        "label": "Wheat",
        "group": "Cereal",
        "season": "November–March",
        "yield_source": "sample",
        "base": 3.15,
        "rain_coef": 0.06,
        "temp_coef": -0.22,
        "soil_coef": 0.08,
        "trend": 0.02,
        "noise": 0.12,
        "floor": 1.40,
        "season_rain_share": 0.14,
        "season_temp_offset": -4.2,
        "ndvi_shift": -0.06,
    },
    "maize": {
        "label": "Maize",
        "group": "Cereal",
        "season": "November–February",
        "yield_source": "bbs",
        "season_rain_share": 0.18,
        "season_temp_offset": -3.4,
        "ndvi_shift": 0.04,
    },
    "potato": {
        "label": "Potato",
        "group": "Tuber",
        "season": "November–February",
        "yield_source": "sample",
        "base": 20.8,
        "rain_coef": -0.55,
        "temp_coef": -1.15,
        "soil_coef": 0.35,
        "trend": 0.11,
        "noise": 0.70,
        "floor": 8.0,
        "season_rain_share": 0.15,
        "season_temp_offset": -4.8,
        "ndvi_shift": -0.08,
    },
    "sweet_potato": {
        "label": "Sweet potato",
        "group": "Tuber",
        "season": "November–March",
        "yield_source": "sample",
        "base": 10.4,
        "rain_coef": 0.28,
        "temp_coef": -0.18,
        "soil_coef": 0.12,
        "trend": 0.04,
        "noise": 0.35,
        "floor": 4.0,
        "season_rain_share": 0.18,
        "season_temp_offset": -3.0,
        "ndvi_shift": -0.03,
    },
    "jute": {
        "label": "Jute",
        "group": "Fiber",
        "season": "March–August",
        "yield_source": "sample",
        "base": 2.42,
        "rain_coef": 0.14,
        "temp_coef": 0.03,
        "soil_coef": 0.04,
        "trend": 0.008,
        "noise": 0.08,
        "floor": 1.05,
        "season_rain_share": 0.70,
        "season_temp_offset": 0.4,
        "ndvi_shift": 0.06,
    },
    "sugarcane": {
        "label": "Sugarcane",
        "group": "Cash crop",
        "season": "Year-round",
        "yield_source": "sample",
        "base": 46.0,
        "rain_coef": 1.60,
        "temp_coef": 0.35,
        "soil_coef": 0.70,
        "trend": 0.12,
        "noise": 1.40,
        "floor": 24.0,
        "season_rain_share": 0.85,
        "season_temp_offset": 0.2,
        "ndvi_shift": 0.05,
    },
    "mustard": {
        "label": "Mustard",
        "group": "Oilseed",
        "season": "November–February",
        "yield_source": "sample",
        "base": 1.12,
        "rain_coef": -0.04,
        "temp_coef": -0.08,
        "soil_coef": 0.03,
        "trend": 0.006,
        "noise": 0.05,
        "floor": 0.45,
        "season_rain_share": 0.12,
        "season_temp_offset": -4.0,
        "ndvi_shift": -0.07,
    },
    "lentil": {
        "label": "Lentil",
        "group": "Pulse",
        "season": "November–March",
        "yield_source": "sample",
        "base": 1.22,
        "rain_coef": -0.03,
        "temp_coef": -0.07,
        "soil_coef": 0.03,
        "trend": 0.005,
        "noise": 0.045,
        "floor": 0.40,
        "season_rain_share": 0.13,
        "season_temp_offset": -3.6,
        "ndvi_shift": -0.05,
    },
    "onion": {
        "label": "Onion",
        "group": "Spice",
        "season": "October–March",
        "yield_source": "sample",
        "base": 11.4,
        "rain_coef": -0.32,
        "temp_coef": -0.22,
        "soil_coef": 0.16,
        "trend": 0.05,
        "noise": 0.40,
        "floor": 4.0,
        "season_rain_share": 0.20,
        "season_temp_offset": -2.4,
        "ndvi_shift": -0.06,
    },
    "garlic": {
        "label": "Garlic",
        "group": "Spice",
        "season": "November–March",
        "yield_source": "sample",
        "base": 6.3,
        "rain_coef": -0.14,
        "temp_coef": -0.16,
        "soil_coef": 0.08,
        "trend": 0.025,
        "noise": 0.22,
        "floor": 2.4,
        "season_rain_share": 0.16,
        "season_temp_offset": -3.2,
        "ndvi_shift": -0.05,
    },
    "chili": {
        "label": "Chili",
        "group": "Spice",
        "season": "October–March",
        "yield_source": "sample",
        "base": 1.68,
        "rain_coef": 0.04,
        "temp_coef": -0.04,
        "soil_coef": 0.03,
        "trend": 0.01,
        "noise": 0.07,
        "floor": 0.55,
        "season_rain_share": 0.22,
        "season_temp_offset": -1.6,
        "ndvi_shift": -0.02,
    },
    "groundnut": {
        "label": "Groundnut",
        "group": "Oilseed",
        "season": "November–March",
        "yield_source": "sample",
        "base": 1.82,
        "rain_coef": 0.05,
        "temp_coef": -0.03,
        "soil_coef": -0.02,
        "trend": 0.008,
        "noise": 0.07,
        "floor": 0.65,
        "season_rain_share": 0.17,
        "season_temp_offset": -2.8,
        "ndvi_shift": -0.03,
    },
    "sesame": {
        "label": "Sesame",
        "group": "Oilseed",
        "season": "March–June",
        "yield_source": "sample",
        "base": 0.92,
        "rain_coef": -0.02,
        "temp_coef": 0.015,
        "soil_coef": 0.01,
        "trend": 0.003,
        "noise": 0.035,
        "floor": 0.30,
        "season_rain_share": 0.28,
        "season_temp_offset": 1.1,
        "ndvi_shift": -0.08,
    },
    "soybean": {
        "label": "Soybean",
        "group": "Oilseed",
        "season": "January–April",
        "yield_source": "sample",
        "base": 1.72,
        "rain_coef": 0.06,
        "temp_coef": -0.05,
        "soil_coef": 0.03,
        "trend": 0.01,
        "noise": 0.06,
        "floor": 0.55,
        "season_rain_share": 0.19,
        "season_temp_offset": -1.8,
        "ndvi_shift": 0.01,
    },
}
CROPS = list(CROP_CATALOG)
PRIMARY_CROP = "boro_rice"

# Boro growing season (Nov transplant – May harvest); climate window for features
BORO_SEASON_MONTHS = [3, 4, 5, 6]  # critical grain-filling: Mar–Jun
AMAN_SEASON_MONTHS = [7, 8, 9, 10]

# NASA POWER API
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/monthly/point"
NASA_POWER_PARAMS = [
    "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"
]

# ISRIC SoilGrids API
SOILGRIDS_BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOIL_PROPERTIES = ["phh2o", "soc", "nitrogen", "cec", "clay", "sand"]
SOIL_DEPTH = "0-5cm"

# Time range for historical analysis
START_YEAR = 2004
END_YEAR = 2023
# Crop files continue through the latest season in the app.
DATASET_END_YEAR = 2026

# Bangladesh bounding box (approximate)
BD_LAT_MIN, BD_LAT_MAX = 20.5, 26.7
BD_LON_MIN, BD_LON_MAX = 88.0, 92.7

# Model features (dynamic + static)
CLIMATE_FEATURES = [
    "temp_max_mean", "temp_min_mean", "rainfall_total",
    "humidity_mean", "solar_radiation_mean",
    "rainfall_mar_jun", "temp_max_mar_jun",
]
SOIL_FEATURES = ["soil_ph", "soil_soc", "soil_nitrogen", "soil_cec", "soil_clay", "soil_sand"]
NDVI_FEATURES = ["ndvi_mean", "ndvi_peak", "ndvi_std"]
FEATURE_COLUMNS = CLIMATE_FEATURES + SOIL_FEATURES + NDVI_FEATURES

# Google Earth Engine (MODIS NDVI)
GEE_PROJECT = os.environ.get("GEE_PROJECT", "curious-ocean-502913-j0")
# Cross-validation
N_SPLITS = 5
RANDOM_STATE = 42


def crop_dataset_path(crop_id: str) -> Path:
    """Modeling dataset for one crop."""

    return CROP_DATA_DIR / f"{crop_id}.csv"


def crop_model_dir(crop_id: str) -> Path:
    """Trained model directory for one crop."""

    return MODEL_DIR / crop_id
