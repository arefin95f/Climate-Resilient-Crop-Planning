"""Build one modeling dataset per agricultural crop.

Boro, Aman, and maize keep the yield values in bbs_yield.csv.
The other crops get reproducible sample yields driven by the
district climate and soil already stored with the Boro dataset,
so each crop can be modeled and downloaded on its own.
"""

import zlib

import numpy as np
import pandas as pd

from src.config import (
    CROP_CATALOG,
    CROP_DATA_DIR,
    DATA_PROCESSED,
    DATA_RAW,
    DATASET_END_YEAR,
)
from src.db import publish_datasets, publish_trained_models


YIELD_COLUMNS = [
    "year",
    "district",
    "division",
    "latitude",
    "longitude",
    "crop",
    "yield_ton_ha",
]


def _noise(crop_id: str, district: str, year: int, width: float) -> float:
    """Stable noise in [-width, width] for one district-year."""

    key = f"{crop_id}|{district}|{int(year)}".encode()
    unit = (zlib.crc32(key) % 10001) / 10000
    return (unit - 0.5) * 2 * width


def _apply_season(row: pd.Series, spec: dict) -> pd.Series:
    """Point seasonal climate and NDVI at this crop's growing window."""

    row = row.copy()
    rain_share = spec.get("season_rain_share")
    temp_offset = spec.get("season_temp_offset")

    if rain_share is not None:
        row["rainfall_mar_jun"] = float(row["rainfall_total"]) * rain_share

    if temp_offset is not None:
        row["temp_max_mar_jun"] = float(row["temp_max_mean"]) + temp_offset

    shift = spec.get("ndvi_shift", 0.0)
    if shift:
        row["ndvi_mean"] = float(np.clip(row["ndvi_mean"] + shift, 0.15, 0.92))
        row["ndvi_peak"] = float(np.clip(row["ndvi_peak"] + shift, 0.2, 0.95))
        row["ndvi_std"] = float(np.clip(row["ndvi_std"] + abs(shift) * 0.15, 0.01, 0.2))

    return row


def _sample_yield(
    row: pd.Series,
    crop_id: str,
    spec: dict,
    rain_mean: float,
    rain_std: float,
    temp_mean: float,
    temp_std: float,
    district_factor: float,
) -> float:
    """Sample yield that responds to rainfall, heat, and soil."""

    rain_z = (float(row["rainfall_total"]) - rain_mean) / rain_std
    temp_z = (float(row["temp_max_mean"]) - temp_mean) / temp_std
    soil_z = float(row["soil_ph"]) - 5.8
    year_term = spec["trend"] * (int(row["year"]) - 2004)
    district_term = (district_factor - 1.0) * spec["base"] * 0.22
    noise = _noise(crop_id, row["district"], int(row["year"]), spec["noise"])

    value = (
        spec["base"]
        + spec["rain_coef"] * rain_z
        + spec["temp_coef"] * temp_z
        + spec["soil_coef"] * soil_z
        + year_term
        + district_term
        + noise
    )
    return round(max(spec["floor"], value), 2)


def _extend_climate(base: pd.DataFrame) -> pd.DataFrame:
    """Add 2024–2026 by continuing each district's latest climate."""

    latest_year = int(base["year"].max())
    if latest_year >= DATASET_END_YEAR:
        return base

    template = base[base["year"] == latest_year]
    extra_rows = []

    for year in range(latest_year + 1, DATASET_END_YEAR + 1):
        step = year - latest_year
        for source in template.itertuples(index=False):
            row = pd.Series(source._asdict())
            district = row["district"]
            row["year"] = year

            row["temp_max_mean"] = (
                float(row["temp_max_mean"])
                + 0.06 * step
                + _noise("climate-tmax", district, year, 0.15)
            )
            row["temp_min_mean"] = (
                float(row["temp_min_mean"])
                + 0.04 * step
                + _noise("climate-tmin", district, year, 0.12)
            )
            row["temp_max_mar_jun"] = (
                float(row["temp_max_mar_jun"])
                + 0.07 * step
                + _noise("climate-tmax-season", district, year, 0.18)
            )

            rain_factor = (
                1
                + 0.01 * step
                + _noise("climate-rain", district, year, 0.08)
            )
            row["rainfall_total"] = max(
                5.0,
                float(row["rainfall_total"]) * rain_factor,
            )
            row["rainfall_mar_jun"] = max(
                1.0,
                float(row["rainfall_mar_jun"]) * rain_factor,
            )
            row["humidity_mean"] = float(
                np.clip(
                    float(row["humidity_mean"])
                    + _noise("climate-humidity", district, year, 1.2),
                    55,
                    95,
                )
            )
            row["solar_radiation_mean"] = max(
                1.0,
                float(row["solar_radiation_mean"])
                + _noise("climate-solar", district, year, 0.2),
            )
            row["ndvi_mean"] = float(
                np.clip(
                    float(row["ndvi_mean"])
                    + _noise("climate-ndvi", district, year, 0.02),
                    0.15,
                    0.92,
                )
            )
            row["ndvi_peak"] = float(
                np.clip(
                    float(row["ndvi_peak"])
                    + _noise("climate-ndvi-peak", district, year, 0.02),
                    0.2,
                    0.95,
                )
            )
            row["ndvi_std"] = float(
                np.clip(
                    float(row["ndvi_std"])
                    + _noise("climate-ndvi-std", district, year, 0.005),
                    0.01,
                    0.2,
                )
            )
            extra_rows.append(row)

    return pd.concat(
        [base, pd.DataFrame(extra_rows)],
        ignore_index=True,
    )


def _project_recorded_yields(
    yield_lookup: dict[tuple[str, int], float],
    crop_id: str,
) -> dict[tuple[str, int], float]:
    """Continue a recorded yield series through the dataset end year."""

    by_district: dict[str, list[tuple[int, float]]] = {}
    for (district, year), yield_value in yield_lookup.items():
        by_district.setdefault(district, []).append(
            (year, yield_value)
        )

    for district, series in by_district.items():
        series.sort()
        last_year, last_yield = series[-1]
        recent = series[-6:]
        if len(recent) >= 2 and recent[-1][0] != recent[0][0]:
            slope = (recent[-1][1] - recent[0][1]) / (
                recent[-1][0] - recent[0][0]
            )
        else:
            slope = 0.0

        for year in range(last_year + 1, DATASET_END_YEAR + 1):
            projected = (
                last_yield
                + slope * (year - last_year)
                + _noise(crop_id, district, year, 0.08)
            )
            yield_lookup[(district, year)] = round(max(0.3, projected), 2)

    return yield_lookup


def build_crop_datasets() -> pd.DataFrame:
    """Write per-crop CSVs and add sample yields to the BBS file."""

    base_path = DATA_PROCESSED / "merged_dataset.csv"
    bbs_path = DATA_RAW / "bbs_yield.csv"

    base = pd.read_csv(base_path)
    base = base[base["crop"] == "boro_rice"].copy()
    base = base[base["year"] <= 2023].copy()
    base = _extend_climate(base)
    bbs = pd.read_csv(bbs_path)
    bbs = bbs[bbs["year"] <= 2023].copy()

    history = base[base["year"] <= 2023]
    rain_mean = float(history["rainfall_total"].mean())
    rain_std = float(history["rainfall_total"].std())
    temp_mean = float(history["temp_max_mean"].mean())
    temp_std = float(history["temp_max_mean"].std())

    district_boro = history.groupby("district")["yield_ton_ha"].mean()
    national_boro = float(district_boro.mean())

    sample_frames = []
    future_frames = []
    summaries = []

    CROP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for crop_id, spec in CROP_CATALOG.items():
        yield_lookup = None
        if spec["yield_source"] == "bbs":
            crop_yields = bbs[bbs["crop"] == crop_id][
                ["district", "year", "yield_ton_ha"]
            ]
            yield_lookup = {
                (row.district, int(row.year)): float(row.yield_ton_ha)
                for row in crop_yields.itertuples(index=False)
                if int(row.year) <= 2023
            }
            yield_lookup = _project_recorded_yields(
                yield_lookup,
                crop_id,
            )

        records = []
        for source in base.itertuples(index=False):
            row = pd.Series(source._asdict())
            row = _apply_season(row, spec)
            row["crop"] = crop_id

            if yield_lookup is not None:
                key = (row["district"], int(row["year"]))
                if key not in yield_lookup:
                    continue
                row["yield_ton_ha"] = yield_lookup[key]
            else:
                factor = float(district_boro.get(row["district"], national_boro))
                factor = factor / national_boro
                row["yield_ton_ha"] = _sample_yield(
                    row,
                    crop_id,
                    spec,
                    rain_mean,
                    rain_std,
                    temp_mean,
                    temp_std,
                    factor,
                )

            records.append(row)

        crop_df = pd.DataFrame(records)
        crop_df.to_csv(CROP_DATA_DIR / f"{crop_id}.csv", index=False)

        if spec["yield_source"] == "sample":
            sample_frames.append(crop_df[YIELD_COLUMNS].copy())
        else:
            future_frames.append(
                crop_df.loc[
                    crop_df["year"] >= 2024,
                    YIELD_COLUMNS,
                ].copy()
            )

        summaries.append(
            {
                "crop": crop_id,
                "label": spec["label"],
                "records": len(crop_df),
                "districts": int(crop_df["district"].nunique()),
                "start_year": int(crop_df["year"].min()),
                "end_year": int(crop_df["year"].max()),
                "mean_yield": round(float(crop_df["yield_ton_ha"].mean()), 2),
            }
        )
        print(
            f"  {spec['label']:<16} {len(crop_df):5d} rows  "
            f"mean {crop_df['yield_ton_ha'].mean():6.2f} t/ha"
        )

    sample_ids = [
        crop_id
        for crop_id, spec in CROP_CATALOG.items()
        if spec["yield_source"] == "sample"
    ]
    bbs = bbs[~bbs["crop"].isin(sample_ids)]
    frames = sample_frames + future_frames
    if frames:
        bbs = pd.concat(
            [bbs, *frames],
            ignore_index=True,
        )
    bbs = bbs.sort_values(
        ["year", "district", "crop"]
    ).reset_index(drop=True)
    bbs.to_csv(bbs_path, index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(CROP_DATA_DIR / "catalog.csv", index=False)

    # Seasons opened from the operations console survive a rebuild.
    from src.year_intake import replay_intakes

    replayed = replay_intakes(publish=False, log=False)
    if replayed.get("opened"):
        print(
            "Applied stored season intake through "
            f"{replayed['latest_year']}"
        )

    published = [
        pd.read_csv(CROP_DATA_DIR / f"{crop_id}.csv")
        for crop_id in CROP_CATALOG
    ]
    counts = publish_datasets(published)
    try:
        counts["trained_models"] = publish_trained_models()
    except Exception as error:
        print(f"Model table was not updated: {type(error).__name__}")
    if (CROP_DATA_DIR / "catalog.csv").exists():
        summary = pd.read_csv(CROP_DATA_DIR / "catalog.csv")
    print(f"\nWrote {len(summary)} crop datasets to {CROP_DATA_DIR}")
    print(
        "MySQL observations: "
        f"{counts['observations']} rows"
    )
    return summary


if __name__ == "__main__":
    build_crop_datasets()
