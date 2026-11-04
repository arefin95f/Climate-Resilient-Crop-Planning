"""Fetch crop-aware climate data from the NASA POWER API."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    AMAN_SEASON_MONTHS,
    BORO_SEASON_MONTHS,
    DATA_RAW,
    END_YEAR,
    NASA_POWER_BASE,
    NASA_POWER_PARAMS,
    START_YEAR,
)
from src.data.bbs_loader import load_districts


CACHE_DIR = DATA_RAW / "climate_cache"


# Crop-specific climate windows.
#
# Boro: March-June
# Aman: July-October
# Maize: November-February
#
# The current maize window uses November-February within the
# available calendar year. Cross-year season handling can be
# added later if the BBS crop-year definition requires it.
CROP_SEASON_MONTHS = {
    "boro_rice": BORO_SEASON_MONTHS,
    "aman_rice": AMAN_SEASON_MONTHS,
    "maize": [11, 12, 1, 2],
}


def _fetch_nasa_power(
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> dict:
    """Query NASA POWER monthly point API for a single location."""

    params = {
        "parameters": ",".join(NASA_POWER_PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }

    response = requests.get(
        NASA_POWER_BASE,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def _monthly_values(
    params_data: dict,
    parameter: str,
    year: int,
) -> dict[int, float]:
    """Extract monthly values for one NASA POWER parameter."""

    values = {}

    parameter_data = params_data.get(
        parameter,
        {},
    )

    for month in range(1, 13):
        key = f"{year}{month:02d}"

        value = parameter_data.get(
            key,
            -999,
        )

        if value != -999:
            values[month] = value

    return values


def _seasonal_sum(
    monthly_data: dict[int, float],
    months: list[int],
) -> float:
    """Calculate the sum across the requested months."""

    values = [
        monthly_data[month]
        for month in months
        if month in monthly_data
    ]

    if not values:
        raise ValueError(
            "No valid monthly values were available "
            "for the requested seasonal period."
        )

    return float(sum(values))


def _seasonal_mean(
    monthly_data: dict[int, float],
    months: list[int],
) -> float:
    """Calculate the mean across the requested months."""

    values = [
        monthly_data[month]
        for month in months
        if month in monthly_data
    ]

    if not values:
        raise ValueError(
            "No valid monthly values were available "
            "for the requested seasonal period."
        )

    return float(np.mean(values))


def _parse_nasa_response(
    data: dict,
    district: str,
    year: int,
) -> dict:
    """Extract annual and crop-aware seasonal climate features."""

    params_data = data["properties"]["parameter"]

    years_in_data = sorted(
        {
            int(key[:4])
            for key in params_data["T2M_MAX"].keys()
        }
    )

    if year not in years_in_data:
        raise ValueError(
            f"No NASA POWER data for {district} in {year}"
        )

    tmax = _monthly_values(
        params_data,
        "T2M_MAX",
        year,
    )

    tmin = _monthly_values(
        params_data,
        "T2M_MIN",
        year,
    )

    rainfall = _monthly_values(
        params_data,
        "PRECTOTCORR",
        year,
    )

    humidity = _monthly_values(
        params_data,
        "RH2M",
        year,
    )

    solar = _monthly_values(
        params_data,
        "ALLSKY_SFC_SW_DWN",
        year,
    )

    if not tmax:
        raise ValueError(
            f"No valid temperature data for {district} in {year}"
        )

    if not tmin:
        raise ValueError(
            f"No valid minimum temperature data "
            f"for {district} in {year}"
        )

    if not rainfall:
        raise ValueError(
            f"No valid rainfall data for {district} in {year}"
        )

    if not humidity:
        raise ValueError(
            f"No valid humidity data for {district} in {year}"
        )

    if not solar:
        raise ValueError(
            f"No valid solar radiation data "
            f"for {district} in {year}"
        )

    boro_months = CROP_SEASON_MONTHS["boro_rice"]
    aman_months = CROP_SEASON_MONTHS["aman_rice"]
    maize_months = CROP_SEASON_MONTHS["maize"]

    return {
        "district": district,
        "year": year,

        # Annual climate features.
        "temp_max_mean": float(np.mean(list(tmax.values()))),
        "temp_min_mean": float(np.mean(list(tmin.values()))),
        "rainfall_total": float(sum(rainfall.values())),
        "humidity_mean": float(np.mean(list(humidity.values()))),
        "solar_radiation_mean": float(
            np.mean(list(solar.values()))
        ),

        # Boro seasonal features.
        "rainfall_mar_jun": _seasonal_sum(
            rainfall,
            boro_months,
        ),
        "temp_max_mar_jun": _seasonal_mean(
            tmax,
            boro_months,
        ),

        # Aman seasonal features.
        "rainfall_jul_oct": _seasonal_sum(
            rainfall,
            aman_months,
        ),
        "temp_max_jul_oct": _seasonal_mean(
            tmax,
            aman_months,
        ),

        # Maize seasonal features.
        "rainfall_nov_feb": _seasonal_sum(
            rainfall,
            maize_months,
        ),
        "temp_max_nov_feb": _seasonal_mean(
            tmax,
            maize_months,
        ),
    }


def fetch_climate_data(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch district-year climate features from NASA POWER."""

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = (
        CACHE_DIR
        / f"climate_{start_year}_{end_year}.csv"
    )

    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file)

    districts = load_districts()

    records = []

    api_start = str(start_year)
    api_end = str(end_year)

    for index, row in districts.iterrows():

        district = row["district"]
        latitude = row["latitude"]
        longitude = row["longitude"]

        print(
            f"  Climate: {district} "
            f"({index + 1}/{len(districts)})"
        )

        try:
            api_data = _fetch_nasa_power(
                latitude,
                longitude,
                api_start,
                api_end,
            )

        except requests.RequestException as error:
            raise RuntimeError(
                f"NASA POWER API failed for "
                f"{district}: {error}"
            ) from error

        for year in range(
            start_year,
            end_year + 1,
        ):
            records.append(
                _parse_nasa_response(
                    api_data,
                    district,
                    year,
                )
            )

        time.sleep(0.5)

    climate_df = pd.DataFrame(records)

    climate_df.to_csv(
        cache_file,
        index=False,
    )

    return climate_df  