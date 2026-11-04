"""Fetch crop-aware MODIS NDVI features via Google Earth Engine."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AMAN_SEASON_MONTHS,
    BORO_SEASON_MONTHS,
    DATA_RAW,
    END_YEAR,
    GEE_PROJECT,
    START_YEAR,
)
from src.data.bbs_loader import load_districts


CACHE_DIR = DATA_RAW / "ndvi_cache"


CROP_SEASON_MONTHS = {
    "boro_rice": BORO_SEASON_MONTHS,
    "aman_rice": AMAN_SEASON_MONTHS,
    "maize": [11, 12, 1, 2],
}


def _initialize_gee():
    """Initialize the Earth Engine client."""

    try:
        import ee

    except ImportError as error:
        raise ImportError(
            "earthengine-api is required. "
            "Install with: pip install earthengine-api"
        ) from error

    if not GEE_PROJECT:
        raise RuntimeError(
            "GEE_PROJECT is not set. "
            "Add it to src/config.py or set the "
            "GEE_PROJECT environment variable."
        )

    try:
        ee.Initialize(
            project=GEE_PROJECT
        )

    except Exception as error:
        raise RuntimeError(
            f"Google Earth Engine initialization failed "
            f"for project '{GEE_PROJECT}'. "
            "Run: python scripts/authenticate_gee.py"
        ) from error

    return ee


def _get_season_dates(
    year: int,
    crop: str,
) -> tuple[str, str]:
    """Return the MODIS filtering dates for a crop season."""

    if crop not in CROP_SEASON_MONTHS:
        supported = ", ".join(
            CROP_SEASON_MONTHS.keys()
        )

        raise ValueError(
            f"Unsupported crop '{crop}'. "
            f"Supported crops: {supported}"
        )

    months = CROP_SEASON_MONTHS[crop]

    start_month = min(months)
    end_month = max(months)

    start_date = (
        f"{year}-{start_month:02d}-01"
    )

    # Earth Engine's filterDate end date is exclusive.
    # Using the first day of the following month ensures
    # the complete final month is included.
    if end_month == 12:
        end_year = year + 1
        end_month_next = 1
    else:
        end_year = year
        end_month_next = end_month + 1

    end_date = (
        f"{end_year}-{end_month_next:02d}-01"
    )

    return start_date, end_date


def _fetch_crop_ndvi(
    ee,
    point,
    year: int,
    crop: str,
) -> dict:
    """Fetch NDVI statistics for one district, year, and crop."""

    start_date, end_date = _get_season_dates(
        year,
        crop,
    )

    collection = (
        ee.ImageCollection(
            "MODIS/061/MOD13Q1"
        )
        .filterBounds(point)
        .filterDate(
            start_date,
            end_date,
        )
        .select("NDVI")
    )

    def extract_val(image):
        value = image.reduceRegion(
            ee.Reducer.mean(),
            point,
            500,
        ).get("NDVI")

        return ee.Feature(
            None,
            {
                "ndvi": value,
                "date": image.date().format(
                    "YYYY-MM-dd"
                ),
            },
        )

    features = collection.map(
        extract_val
    ).getInfo()

    ndvi_values = [
        feature["properties"]["ndvi"] / 10000
        for feature in features["features"]
        if feature["properties"]["ndvi"] is not None
    ]

    if not ndvi_values:
        raise RuntimeError(
            f"No MODIS NDVI data for "
            f"{crop}, {year}"
        )

    return {
        "ndvi_mean": round(
            float(np.mean(ndvi_values)),
            3,
        ),
        "ndvi_peak": round(
            float(np.max(ndvi_values)),
            3,
        ),
        "ndvi_std": round(
            float(np.std(ndvi_values)),
            3,
        ),
    }


def fetch_ndvi_data(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch crop-aware district-year NDVI features.

    The returned dataset contains one row per district,
    year, and supported crop.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = (
        CACHE_DIR
        / f"ndvi_{start_year}_{end_year}_multicrop.csv"
    )

    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file)

    ee = _initialize_gee()

    districts = load_districts()

    records = []

    total = (
        len(districts)
        * (end_year - start_year + 1)
        * len(CROP_SEASON_MONTHS)
    )

    done = 0

    print(
        "  Using Google Earth Engine "
        f"(MODIS MOD13Q1, project={GEE_PROJECT})..."
    )

    for _, row in districts.iterrows():

        district = row["district"]

        point = ee.Geometry.Point(
            [
                row["longitude"],
                row["latitude"],
            ]
        )

        for year in range(
            start_year,
            end_year + 1,
        ):

            for crop in CROP_SEASON_MONTHS:

                done += 1

                print(
                    f"  NDVI (GEE): "
                    f"{district} {year} "
                    f"{crop} "
                    f"({done}/{total})"
                )

                features = _fetch_crop_ndvi(
                    ee,
                    point,
                    year,
                    crop,
                )

                records.append(
                    {
                        "district": district,
                        "year": year,
                        "crop": crop,
                        **features,
                    }
                )

    df = pd.DataFrame(records)

    df.to_csv(
        cache_file,
        index=False,
    )

    print(
        f"  GEE fetched {len(records)} "
        "district-year-crop NDVI records."
    )

    return df  