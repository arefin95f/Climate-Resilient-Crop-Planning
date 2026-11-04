"""Load and prepare BBS agricultural yield data."""

from pathlib import Path

import pandas as pd

from src.config import CROPS, DATA_RAW, PRIMARY_CROP


REQUIRED_COLUMNS = [
    "year",
    "district",
    "division",
    "latitude",
    "longitude",
    "crop",
    "yield_ton_ha",
]


def load_districts() -> pd.DataFrame:
    """Load district metadata."""

    path = DATA_RAW / "districts.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"District metadata not found: {path}"
        )

    return pd.read_csv(path)


def load_bbs_data() -> pd.DataFrame:
    """Load the complete BBS crop-yield dataset."""

    path = DATA_RAW / "bbs_yield.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"BBS yield data not found: {path}. "
            "Place bbs_yield.csv in data/raw/ before "
            "running the pipeline."
        )

    df = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "BBS dataset is missing required columns: "
            f"{missing_columns}"
        )

    df = df.copy()

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["yield_ton_ha"] = pd.to_numeric(
        df["yield_ton_ha"],
        errors="coerce",
    )

    df["crop"] = (
        df["crop"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "year",
            "district",
            "crop",
            "yield_ton_ha",
        ]
    )

    return df


def validate_crop(crop: str) -> str:
    """Validate and normalize a crop name."""

    normalized_crop = (
        str(crop)
        .strip()
        .lower()
    )

    if normalized_crop not in CROPS:
        supported = ", ".join(CROPS)

        raise ValueError(
            f"Unsupported crop '{crop}'. "
            f"Supported crops: {supported}"
        )

    return normalized_crop


def get_yield_for_crop(
    df: pd.DataFrame,
    crop: str = PRIMARY_CROP,
) -> pd.DataFrame:
    """Return yield records for one supported crop."""

    crop = validate_crop(crop)

    crop_data = df[
        df["crop"] == crop
    ].copy()

    if crop_data.empty:
        raise ValueError(
            f"No BBS yield records were found "
            f"for crop '{crop}'."
        )

    return crop_data.reset_index(drop=True)


def get_all_crop_yields(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return BBS yield data separately for every supported crop."""

    crop_data = {}

    for crop in CROPS:
        crop_data[crop] = get_yield_for_crop(
            df,
            crop,
        )

    return crop_data


def validate_crop_coverage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize district and year coverage for each crop."""

    records = []

    for crop in CROPS:
        crop_df = get_yield_for_crop(
            df,
            crop,
        )

        records.append(
            {
                "crop": crop,
                "records": len(crop_df),
                "districts": crop_df["district"].nunique(),
                "start_year": int(
                    crop_df["year"].min()
                ),
                "end_year": int(
                    crop_df["year"].max()
                ),
                "missing_yield": int(
                    crop_df["yield_ton_ha"].isna().sum()
                ),
            }
        )

    return pd.DataFrame(records)