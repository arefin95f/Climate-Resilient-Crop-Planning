"""Central configuration for AgriSight-BD."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = DATA_PROCESSED / "model_artifacts"

# Crop configuration
CROPS = ["boro_rice", "aman_rice", "maize"]
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
