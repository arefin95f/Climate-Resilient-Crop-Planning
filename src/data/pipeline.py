"""Merge all data sources into a modeling-ready dataset."""

import pandas as pd



from src.config import DATA_PROCESSED, PRIMARY_CROP

from src.data.bbs_loader import get_yield_for_crop, load_bbs_data

from src.data.ndvi import fetch_ndvi_data

from src.data.nasa_power import fetch_climate_data

from src.data.soilgrids import fetch_soil_data





def clean_merged(df: pd.DataFrame) -> pd.DataFrame:

    """Handle missing values and remove invalid rows."""

    df = df.dropna(subset=["yield_ton_ha"])

    numeric_cols = df.select_dtypes(include="number").columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = df[df["yield_ton_ha"] > 0]

    return df.reset_index(drop=True)





def run_pipeline(refresh: bool = False, crop: str = PRIMARY_CROP) -> pd.DataFrame:

    """

    Execute the full data pipeline:

    1. Load BBS yield data

    2. Fetch NASA POWER climate

    3. Fetch ISRIC soil properties

    4. Fetch MODIS NDVI via Google Earth Engine

    5. Merge and clean

    """

    print("=" * 60)

    print("AgriSight-BD Data Pipeline")

    print("=" * 60)



    print("\n[1/4] Loading BBS yield data...")

    bbs = load_bbs_data()

    yield_df = get_yield_for_crop(bbs, crop)

    print(f"  {len(yield_df)} yield records ({crop})")



    print("\n[2/4] Fetching climate data (NASA POWER)...")

    climate = fetch_climate_data(refresh=refresh)

    print(f"  {len(climate)} climate records")



    print("\n[3/4] Fetching soil data (ISRIC SoilGrids)...")

    soil = fetch_soil_data(refresh=refresh)

    print(f"  {len(soil)} soil records")



    print("\n[4/4] Fetching NDVI features (Google Earth Engine)...")

    ndvi = fetch_ndvi_data(refresh=refresh)

    print(f"  {len(ndvi)} NDVI records")



    print("\nMerging datasets...")

    merged = yield_df.merge(climate, on=["district", "year"], how="inner")

    merged = merged.merge(soil, on="district", how="left")

    merged = merged.merge(ndvi, on=["district", "year"], how="left")



    merged = clean_merged(merged)



    out_path = DATA_PROCESSED / "merged_dataset.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")

    print(f"Final dataset: {len(merged)} rows x {len(merged.columns)} columns")

    print("=" * 60)



    return merged





if __name__ == "__main__":

    run_pipeline()


