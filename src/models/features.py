"""Leakage-safe features for a district yield forecast.

Every yield feature uses only earlier harvests. Climate and greenness
enter as the departure from that district's own past seasons, so a
slider or a what-if shock changes the forecast the same way it changed
the training matrix.
"""

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, SOIL_FEATURES


HISTORY_FEATURES = [
    "yield_lag1",
    "yield_lag2",
    "yield_roll3",
    "yield_roll5",
    "yield_expanding",
    "yield_slope",
    "yield_trend",
    "year_index",
    "national_lag1",
    "national_lag2",
]

ANOMALY_BASES = [
    "temp_max_mean",
    "temp_min_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "rainfall_mar_jun",
    "temp_max_mar_jun",
    "ndvi_mean",
    "ndvi_peak",
    "ndvi_std",
]


def _past_slopes(years: np.ndarray, values: np.ndarray, window: int = 6) -> np.ndarray:
    """One-year yield trend fitted on earlier seasons only."""

    slopes = np.full(len(values), np.nan)
    for index in range(3, len(values)):
        start = max(0, index - window)
        year_window = years[start:index].astype(float)
        value_window = values[start:index].astype(float)
        year_window = year_window - year_window.mean()
        value_window = value_window - value_window.mean()
        denominator = float(np.dot(year_window, year_window))
        if denominator:
            slopes[index] = float(np.dot(year_window, value_window) / denominator)
        else:
            slopes[index] = 0.0
    return slopes


def add_forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add past-yield, climate-anomaly, and district columns."""

    frame = frame.sort_values(["district", "year"]).reset_index(drop=True)
    grouped = frame.groupby("district", sort=False)

    frame["yield_lag1"] = grouped["yield_ton_ha"].shift(1)
    frame["yield_lag2"] = grouped["yield_ton_ha"].shift(2)
    frame["yield_roll3"] = grouped["yield_ton_ha"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    frame["yield_roll5"] = grouped["yield_ton_ha"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    frame["yield_expanding"] = grouped["yield_ton_ha"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )

    slopes = np.full(len(frame), np.nan)
    for positions in grouped.indices.values():
        positions = np.asarray(positions)
        slopes[positions] = _past_slopes(
            frame.loc[positions, "year"].to_numpy(),
            frame.loc[positions, "yield_ton_ha"].to_numpy(),
        )
    frame["yield_slope"] = slopes
    frame["yield_trend"] = frame["yield_lag1"] + frame["yield_slope"]
    frame["year_index"] = frame["year"] - int(frame["year"].min())

    national = frame.groupby("year")["yield_ton_ha"].mean().sort_index()
    frame["national_lag1"] = frame["year"].map(national.shift(1))
    frame["national_lag2"] = frame["year"].map(national.shift(2))

    grouped = frame.groupby("district", sort=False)
    for column in ANOMALY_BASES:
        if column not in frame.columns:
            continue
        normal = grouped[column].transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean()
        )
        frame[f"{column}_normal"] = normal
        frame[f"{column}_anom"] = frame[column] - normal

    for district in sorted(frame["district"].dropna().unique()):
        frame[f"dist_{district}"] = (frame["district"] == district).astype(float)

    return frame.sort_values(["year", "district"]).reset_index(drop=True)


def candidate_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Feature lists for the two models compared in training."""

    history = [name for name in HISTORY_FEATURES if name in frame.columns]
    anomalies = [
        f"{column}_anom"
        for column in ANOMALY_BASES
        if f"{column}_anom" in frame.columns
    ]
    soil = [name for name in SOIL_FEATURES if name in frame.columns]
    raw = [name for name in FEATURE_COLUMNS if name in frame.columns]
    districts = [name for name in frame.columns if name.startswith("dist_")]
    return {
        "XGBoost": history + anomalies + soil,
        "Ridge": history + anomalies + raw + districts,
    }


def prediction_frame(
    row: pd.Series,
    feature_names,
    overrides: dict | None = None,
) -> pd.DataFrame:
    """One model row. Climate overrides refresh the matching anomalies."""

    overrides = overrides or {}
    names = [str(name) for name in feature_names]
    source = row.to_dict()
    source.update(overrides)

    values = []
    for name in names:
        if name.endswith("_anom"):
            base = name[: -len("_anom")]
            current = source.get(base, np.nan)
            normal = source.get(f"{base}_normal", np.nan)
            if pd.notna(current) and pd.notna(normal):
                values.append(float(current) - float(normal))
                continue
        value = source.get(name, np.nan)
        values.append(0.0 if pd.isna(value) else float(value))

    return pd.DataFrame([values], columns=names)
