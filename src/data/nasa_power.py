"""Fetch climate data from NASA POWER API."""

import time

from pathlib import Path



import numpy as np

import pandas as pd

import requests



from src.config import (

    BORO_SEASON_MONTHS,

    DATA_RAW,

    END_YEAR,

    NASA_POWER_BASE,

    NASA_POWER_PARAMS,

    START_YEAR,

)

from src.data.bbs_loader import load_districts



CACHE_DIR = DATA_RAW / "climate_cache"





def _fetch_nasa_power(lat: float, lon: float, start: str, end: str) -> dict:

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

    resp = requests.get(NASA_POWER_BASE, params=params, timeout=60)

    resp.raise_for_status()

    return resp.json()





def _parse_nasa_response(data: dict, district: str, year: int) -> dict:

    """Extract monthly climate features from NASA POWER JSON response."""

    params_data = data["properties"]["parameter"]

    years_in_data = sorted(set(int(k[:4]) for k in params_data["T2M_MAX"].keys()))



    if year not in years_in_data:

        raise ValueError(f"No NASA POWER data for {district} in {year}")



    def monthly_values(param: str, yr: int) -> list[float]:

        vals = []

        for m in range(1, 13):

            key = f"{yr}{m:02d}"

            v = params_data[param].get(key, -999)

            if v != -999:

                vals.append(v)

        return vals



    tmax = monthly_values("T2M_MAX", year)

    tmin = monthly_values("T2M_MIN", year)

    rain = monthly_values("PRECTOTCORR", year)

    rh = monthly_values("RH2M", year)

    solar = monthly_values("ALLSKY_SFC_SW_DWN", year)



    if not tmax:

        raise ValueError(f"No valid temperature data for {district} in {year}")



    boro_rain = sum(rain[m - 1] for m in BORO_SEASON_MONTHS if m - 1 < len(rain))

    boro_tmax = np.mean([tmax[m - 1] for m in BORO_SEASON_MONTHS if m - 1 < len(tmax)])



    return {

        "district": district,

        "year": year,

        "temp_max_mean": np.mean(tmax),

        "temp_min_mean": np.mean(tmin),

        "rainfall_total": sum(rain),

        "humidity_mean": np.mean(rh),

        "solar_radiation_mean": np.mean(solar),

        "rainfall_mar_jun": boro_rain,

        "temp_max_mar_jun": boro_tmax,

    }





def fetch_climate_data(

    start_year: int = START_YEAR,

    end_year: int = END_YEAR,

    refresh: bool = False,

) -> pd.DataFrame:

    """Fetch district-year climate features from NASA POWER."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE_DIR / f"climate_{start_year}_{end_year}.csv"



    if cache_file.exists() and not refresh:

        return pd.read_csv(cache_file)



    districts = load_districts()

    records = []

    api_start = str(start_year)
    api_end = str(end_year)



    for i, row in districts.iterrows():

        district, lat, lon = row["district"], row["latitude"], row["longitude"]

        print(f"  Climate: {district} ({i + 1}/{len(districts)})")



        try:

            api_data = _fetch_nasa_power(lat, lon, api_start, api_end)

        except requests.RequestException as e:

            raise RuntimeError(f"NASA POWER API failed for {district}: {e}") from e



        for year in range(start_year, end_year + 1):

            records.append(_parse_nasa_response(api_data, district, year))



        time.sleep(0.5)



    df = pd.DataFrame(records)

    df.to_csv(cache_file, index=False)

    return df


