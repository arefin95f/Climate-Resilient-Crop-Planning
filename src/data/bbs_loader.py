"""Load BBS Yearbook of Agricultural Statistics yield data."""

from pathlib import Path



import pandas as pd



from src.config import DATA_RAW, PRIMARY_CROP





def load_districts() -> pd.DataFrame:

    """Load district metadata (name, division, coordinates)."""

    path = DATA_RAW / "districts.csv"

    if not path.exists():

        raise FileNotFoundError(f"District metadata not found: {path}")

    return pd.read_csv(path)





def load_bbs_data() -> pd.DataFrame:

    """Load BBS yield data from the raw CSV."""

    path = DATA_RAW / "bbs_yield.csv"

    if not path.exists():

        raise FileNotFoundError(

            f"BBS yield data not found: {path}. "

            "Place bbs_yield.csv in data/raw/ before running the pipeline."

        )

    return pd.read_csv(path)





def get_yield_for_crop(df: pd.DataFrame, crop: str = PRIMARY_CROP) -> pd.DataFrame:

    """Filter yield data to a single crop."""

    return df[df["crop"] == crop].copy()


