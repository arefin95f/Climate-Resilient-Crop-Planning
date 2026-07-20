"""Fetch soil properties from ISRIC SoilGrids API."""

import time

from pathlib import Path



import pandas as pd

import requests



from src.config import DATA_RAW, SOIL_DEPTH, SOIL_PROPERTIES, SOILGRIDS_BASE

from src.data.bbs_loader import load_districts



CACHE_DIR = DATA_RAW / "soil_cache"



PROPERTY_MAP = {

    "phh2o": "soil_ph",

    "soc": "soil_soc",

    "nitrogen": "soil_nitrogen",

    "cec": "soil_cec",

    "clay": "soil_clay",

    "sand": "soil_sand",

}



# Some district centroids fall on water or data gaps; try nearby land points.

NEARBY_OFFSETS_DEG = [

    (0.0, 0.0),

    (0.03, 0.0),

    (-0.03, 0.0),

    (0.0, 0.03),

    (0.0, -0.03),

    (0.03, 0.03),

    (-0.03, -0.03),

]



SOIL_DEPTH_FALLBACKS = ["0-5cm", "5-15cm", "15-30cm"]





def _fetch_soilgrids(lat: float, lon: float, depth: str) -> dict:

    """Query ISRIC SoilGrids REST API for a point."""

    params = {

        "lon": lon,

        "lat": lat,

        "property": SOIL_PROPERTIES,

        "depth": depth,

        "value": "mean",

    }

    resp = requests.get(SOILGRIDS_BASE, params=params, timeout=30)

    resp.raise_for_status()

    data = resp.json()

    if "properties" not in data:

        raise ValueError(f"Unexpected SoilGrids response: {data}")

    return data





def _parse_soilgrids(data: dict) -> dict:

    """Parse ISRIC SoilGrids JSON response into feature columns."""

    result = {}

    for layer in data.get("properties", {}).get("layers", []):

        prop_name = layer["name"]

        if prop_name not in PROPERTY_MAP:

            continue



        depths = layer.get("depths", [])

        if not depths:

            continue



        val = depths[0]["values"].get("mean")

        if val is None:

            continue



        d_factor = layer.get("unit_measure", {}).get("d_factor", 1)

        result[PROPERTY_MAP[prop_name]] = round(val / d_factor, 3)



    return result





def _fetch_soil_for_location(lat: float, lon: float) -> dict:

    """

    Resolve soil properties for a district coordinate.



    SoilGrids often returns null at exact centroids over rivers/coast;

    try small nearby offsets and alternate depth layers before failing.

    """

    required = set(PROPERTY_MAP.values())

    last_error = "no data returned"



    for dlat, dlon in NEARBY_OFFSETS_DEG:

        sample_lat = lat + dlat

        sample_lon = lon + dlon



        for depth in SOIL_DEPTH_FALLBACKS:

            try:

                api_data = _fetch_soilgrids(sample_lat, sample_lon, depth)

                parsed = _parse_soilgrids(api_data)

            except (requests.RequestException, ValueError) as e:

                last_error = str(e)

                time.sleep(0.5)

                continue



            missing = required - set(parsed)

            if not missing:

                if dlat or dlon or depth != SOIL_DEPTH:

                    print(f"    used nearby sample ({sample_lat:.4f}, {sample_lon:.4f}) depth={depth}")

                return parsed



            last_error = f"missing {sorted(missing)} at depth {depth}"

            time.sleep(0.3)



    raise ValueError(last_error)





def fetch_soil_data(refresh: bool = False) -> pd.DataFrame:

    """Fetch static soil properties per district from ISRIC SoilGrids."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = CACHE_DIR / "soil_districts.csv"



    if cache_file.exists() and not refresh:

        return pd.read_csv(cache_file)



    districts = load_districts()

    records = []



    for i, row in districts.iterrows():

        district, lat, lon = row["district"], row["latitude"], row["longitude"]

        print(f"  Soil: {district} ({i + 1}/{len(districts)})")



        try:

            parsed = _fetch_soil_for_location(lat, lon)

            parsed["district"] = district

            records.append(parsed)

        except ValueError as e:

            raise RuntimeError(f"SoilGrids API failed for {district}: {e}") from e



        time.sleep(0.3)



    df = pd.DataFrame(records)

    df.to_csv(cache_file, index=False)

    return df


