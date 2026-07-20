"""Fetch MODIS NDVI features via Google Earth Engine."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BORO_SEASON_MONTHS, DATA_RAW, END_YEAR, GEE_PROJECT, START_YEAR
from src.data.bbs_loader import load_districts

CACHE_DIR = DATA_RAW / "ndvi_cache"


def _initialize_gee():
    """Initialize the Earth Engine client."""
    try:
        import ee
    except ImportError as e:
        raise ImportError(
            "earthengine-api is required. Install with: pip install earthengine-api"
        ) from e

    if not GEE_PROJECT:
        raise RuntimeError(
            "GEE_PROJECT is not set. Add it to src/config.py or set the GEE_PROJECT environment variable."
        )

    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        raise RuntimeError(
            f"Google Earth Engine initialization failed for project '{GEE_PROJECT}'. "
            "Run: python scripts/authenticate_gee.py"
        ) from e

    return ee


def fetch_ndvi_data(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch district-year NDVI features from MODIS via Google Earth Engine."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"ndvi_{start_year}_{end_year}.csv"

    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file)

    ee = _initialize_gee()
    districts = load_districts()
    records = []
    total = len(districts) * (end_year - start_year + 1)
    done = 0

    print(f"  Using Google Earth Engine (MODIS MOD13Q1, project={GEE_PROJECT})...")

    for _, row in districts.iterrows():
        district = row["district"]
        point = ee.Geometry.Point([row["longitude"], row["latitude"]])

        for year in range(start_year, end_year + 1):
            done += 1
            print(f"  NDVI (GEE): {district} {year} ({done}/{total})")

            start_date = f"{year}-{min(BORO_SEASON_MONTHS):02d}-01"
            end_date = f"{year}-{max(BORO_SEASON_MONTHS):02d}-28"

            collection = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterBounds(point)
                .filterDate(start_date, end_date)
                .select("NDVI")
            )

            def extract_val(image):
                val = image.reduceRegion(ee.Reducer.mean(), point, 500).get("NDVI")
                return ee.Feature(None, {"ndvi": val, "date": image.date().format("YYYY-MM-dd")})

            features = collection.map(extract_val).getInfo()
            ndvi_vals = [
                f["properties"]["ndvi"] / 10000
                for f in features["features"]
                if f["properties"]["ndvi"] is not None
            ]

            if not ndvi_vals:
                raise RuntimeError(f"No MODIS NDVI data for {district} in {year}")

            records.append({
                "district": district,
                "year": year,
                "ndvi_mean": round(np.mean(ndvi_vals), 3),
                "ndvi_peak": round(np.max(ndvi_vals), 3),
                "ndvi_std": round(np.std(ndvi_vals), 3),
            })

    df = pd.DataFrame(records)
    df.to_csv(cache_file, index=False)
    print(f"  GEE fetched {len(records)} district-year NDVI records.")
    return df
