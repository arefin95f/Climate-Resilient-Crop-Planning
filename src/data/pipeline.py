"""Merge all data sources into a multi-crop modeling dataset."""

import pandas as pd

from src.config import DATA_PROCESSED, PRIMARY_CROP
from src.data.bbs_loader import (
    get_yield_for_crop,
    load_bbs_data,
)
from src.data.ndvi import fetch_ndvi_data
from src.data.nasa_power import fetch_climate_data
from src.data.soilgrids import fetch_soil_data


def clean_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove invalid rows."""

    df = df.dropna(
        subset=["yield_ton_ha"]
    )

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    df = df[df["yield_ton_ha"] > 0]

    return df.reset_index(drop=True)


def run_pipeline(
    refresh: bool = False,
    crop: str | None = PRIMARY_CROP,
) -> pd.DataFrame:
    """
    Execute the full multi-crop data pipeline.

    When crop is provided, only that crop is processed.

    When crop is None, all supported crops are processed.

    Steps:
    1. Load BBS yield data.
    2. Fetch NASA POWER climate data.
    3. Fetch ISRIC SoilGrids soil data.
    4. Fetch crop-aware MODIS NDVI data.
    5. Merge the datasets using crop-aware keys.
    6. Clean the final modeling dataset.
    7. Save the processed dataset.
    """

    print("=" * 60)
    print("AgriSight-BD Data Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading BBS yield data...")

    bbs = load_bbs_data()

    if crop is None:
        yield_df = bbs.copy()

        print(
            f"  {len(yield_df)} yield records "
            "(all supported crops)"
        )

    else:
        yield_df = get_yield_for_crop(
            bbs,
            crop,
        )

        print(
            f"  {len(yield_df)} yield records "
            f"({crop})"
        )

    print(
        "\n[2/4] Fetching climate data "
        "(NASA POWER)..."
    )

    climate = fetch_climate_data(
        refresh=refresh
    )

    print(
        f"  {len(climate)} climate records"
    )

    print(
        "\n[3/4] Fetching soil data "
        "(ISRIC SoilGrids)..."
    )

    soil = fetch_soil_data(
        refresh=refresh
    )

    print(
        f"  {len(soil)} soil records"
    )

    print(
        "\n[4/4] Fetching NDVI features "
        "(Google Earth Engine)..."
    )

    ndvi = fetch_ndvi_data(
        refresh=refresh
    )

    print(
        f"  {len(ndvi)} NDVI records"
    )

    print("\nMerging datasets...")

    # Yield + climate.
    merged = yield_df.merge(
        climate,
        on=["district", "year"],
        how="inner",
    )

    # Soil is district-level, so crop/year are not needed.
    merged = merged.merge(
        soil,
        on="district",
        how="left",
    )

    # NDVI is now crop-aware.
    #
    # This prevents Boro yield records from being
    # matched with Aman or Maize NDVI records.
    merged = merged.merge(
        ndvi,
        on=["district", "year", "crop"],
        how="left",
    )

    merged = clean_merged(
        merged
    )

    out_path = (
        DATA_PROCESSED
        / "merged_dataset.csv"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        out_path,
        index=False,
    )

    print(
        f"\nSaved: {out_path}"
    )

    print(
        f"Final dataset: "
        f"{len(merged)} rows x "
        f"{len(merged.columns)} columns"
    )

    if "crop" in merged.columns:

        print("\nCrop distribution:")

        crop_counts = (
            merged["crop"]
            .value_counts()
            .sort_index()
        )

        for crop_name, count in crop_counts.items():
            print(
                f"  {crop_name}: {count}"
            )

    print("=" * 60)

    return merged


if __name__ == "__main__":
    run_pipeline()  