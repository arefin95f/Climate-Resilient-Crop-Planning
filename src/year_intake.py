"""Open a new season and keep every crop, map, and estimate on it.

Official yields replace a district-crop-year. Cells left blank are
continued: climate follows the same 2023-anchor rules already used for
2024–2026, and a recorded crop follows that district's recent harvests.
Sample crops follow their climate formula. The planner reads the newest
year in these files, so a season opened here becomes the current season.
"""

from __future__ import annotations

import difflib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.accounts import log_admin_event
from src.config import (
    CROP_CATALOG,
    CROP_DATA_DIR,
    DATA_RAW,
    crop_model_dir,
)
from src.data.build_crop_datasets import _apply_season, _noise, _sample_yield
from src.db import (
    DATASET_COLUMNS,
    mysql_configured,
    publish_datasets,
    publish_trained_models,
    use_mysql,
)
from src.models.features import add_forecast_features, prediction_frame


ANCHOR_YEAR = 2023
MAX_AHEAD = 8
MAX_OPEN = 5
YIELD_CAP = 150.0

HEADER_ALIASES = {
    "yr": "year",
    "district_name": "district",
    "crop_name": "crop",
    "crop_id": "crop",
    "yield": "yield_ton_ha",
    "yield_t_ha": "yield_ton_ha",
    "yield_ton_per_ha": "yield_ton_ha",
    "yield_(t/ha)": "yield_ton_ha",
    "yield_t/ha": "yield_ton_ha",
    "rainfall": "rainfall_total",
    "rainfall_mm": "rainfall_total",
    "maximum_temperature": "temp_max_mean",
    "temp_max": "temp_max_mean",
    "max_temperature": "temp_max_mean",
    "minimum_temperature": "temp_min_mean",
    "temp_min": "temp_min_mean",
    "min_temperature": "temp_min_mean",
    "humidity": "humidity_mean",
    "sunlight": "solar_radiation_mean",
    "solar": "solar_radiation_mean",
    "solar_radiation": "solar_radiation_mean",
    "ph": "soil_ph",
    "soil_ph": "soil_ph",
    "organic_carbon": "soil_soc",
    "soc": "soil_soc",
    "nitrogen": "soil_nitrogen",
    "clay": "soil_clay",
    "sand": "soil_sand",
    "cec": "soil_cec",
    "soil_cec": "soil_cec",
    "greenness": "ndvi_mean",
    "ndvi": "ndvi_mean",
    "ndvi_mean": "ndvi_mean",
    "ndvi_peak": "ndvi_peak",
    "ndvi_std": "ndvi_std",
    "average_greenness": "ndvi_mean",
    "peak_greenness": "ndvi_peak",
    "greenness_variation": "ndvi_std",
}

WEATHER_FIELDS = [
    "rainfall_total",
    "temp_max_mean",
    "temp_min_mean",
    "humidity_mean",
    "solar_radiation_mean",
]
SOIL_FIELDS = [
    "soil_ph",
    "soil_soc",
    "soil_nitrogen",
    "soil_clay",
    "soil_sand",
    "soil_cec",
]
GREENNESS_FIELDS = ["ndvi_mean", "ndvi_peak", "ndvi_std"]
WEATHER_LIMITS = {
    "rainfall_total": (5, 800, "rainfall"),
    "temp_max_mean": (20, 45, "maximum temperature"),
    "temp_min_mean": (5, 32, "minimum temperature"),
    "humidity_mean": (40, 100, "humidity"),
    "solar_radiation_mean": (5, 30, "sunlight"),
}
SOIL_LIMITS = {
    "soil_ph": (4, 9, "pH"),
    "soil_soc": (1, 120, "organic carbon"),
    "soil_nitrogen": (0.2, 15, "nitrogen"),
    "soil_clay": (5, 80, "clay"),
    "soil_sand": (5, 80, "sand"),
    "soil_cec": (1, 80, "CEC"),
}
GREENNESS_LIMITS = {
    "ndvi_mean": (0, 1, "average greenness"),
    "ndvi_peak": (0, 1, "peak greenness"),
    "ndvi_std": (0, 0.5, "greenness variation"),
}
ROW_FIELDS = [
    "yield_ton_ha",
    "temp_max_mean",
    "temp_min_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
    "rainfall_mar_jun",
    "temp_max_mar_jun",
    "soil_cec",
    "soil_clay",
    "soil_nitrogen",
    "soil_ph",
    "soil_sand",
    "soil_soc",
    "ndvi_mean",
    "ndvi_peak",
    "ndvi_std",
]
ROW_LIMITS = {
    **WEATHER_LIMITS,
    **SOIL_LIMITS,
    "yield_ton_ha": (0.01, YIELD_CAP, "yield"),
    "rainfall_mar_jun": (0, 800, "season rainfall"),
    "temp_max_mar_jun": (15, 50, "season maximum temperature"),
    "soil_cec": (1, 80, "CEC"),
    "ndvi_mean": (0, 1, "greenness"),
    "ndvi_peak": (0, 1, "peak greenness"),
    "ndvi_std": (0, 0.5, "greenness variation"),
}


@dataclass
class Store:
    """Folders the season registry reads and writes."""

    crop_dir: Path
    intake_dir: Path

    @property
    def catalog_file(self) -> Path:
        return self.crop_dir / "catalog.csv"

    @property
    def provenance_file(self) -> Path:
        return self.crop_dir / "provenance.csv"

    @property
    def official_file(self) -> Path:
        return self.intake_dir / "official.csv"

    @property
    def horizon_file(self) -> Path:
        return self.intake_dir / "horizon.json"

    @property
    def weather_file(self) -> Path:
        return self.intake_dir / "weather.csv"

    @property
    def soil_file(self) -> Path:
        return self.intake_dir / "soil.csv"

    @property
    def greenness_file(self) -> Path:
        return self.intake_dir / "greenness.csv"

    @property
    def crop_rows_file(self) -> Path:
        return self.intake_dir / "crop_rows.csv"


@dataclass
class UploadReport:
    """Checked official-yield file."""

    rows: pd.DataFrame
    file_errors: list[str]
    row_errors: list[str]
    warnings: list[str]
    blank_rows: int
    read_rows: int

    @property
    def years(self) -> list[int]:
        if self.rows.empty:
            return []
        return sorted(int(year) for year in self.rows["year"].unique())


def default_store() -> Store:
    return Store(
        crop_dir=CROP_DATA_DIR,
        intake_dir=DATA_RAW / "intakes",
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm(value: object) -> str:
    text = " ".join(str(value).split()).casefold()
    return text.replace("’", "'").replace("`", "'")


def _crop_lookup() -> dict[str, str]:
    lookup = {}
    for crop_id, spec in CROP_CATALOG.items():
        lookup[_norm(crop_id)] = crop_id
        lookup[_norm(spec["label"])] = crop_id
    return lookup


def _read_crop(store: Store, crop_id: str) -> pd.DataFrame:
    path = store.crop_dir / f"{crop_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Crop dataset not found: {path.name}")
    frame = pd.read_csv(path)
    frame["year"] = frame["year"].astype(int)
    return frame


def _load_frames(store: Store) -> dict[str, pd.DataFrame]:
    return {
        crop_id: _read_crop(store, crop_id)
        for crop_id in CROP_CATALOG
    }


def _max_year(frames: dict[str, pd.DataFrame]) -> int:
    return max(int(frame["year"].max()) for frame in frames.values())


def _min_year(frames: dict[str, pd.DataFrame]) -> int:
    return min(int(frame["year"].min()) for frame in frames.values())


def _anchor_year(frame: pd.DataFrame) -> int:
    years = {int(year) for year in frame["year"]}
    if ANCHOR_YEAR in years:
        return ANCHOR_YEAR
    earlier = [year for year in years if year <= ANCHOR_YEAR]
    if earlier:
        return max(earlier)
    return min(years)


def _load_provenance(store: Store) -> pd.DataFrame:
    columns = ["year", "district", "crop", "source"]
    if not store.provenance_file.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(store.provenance_file)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["year"] = frame["year"].astype(int)
    return frame[columns]


def _read_official(store: Store) -> pd.DataFrame:
    columns = ["year", "district", "crop", "yield_ton_ha", "recorded_at"]
    if not store.official_file.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(store.official_file)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["year"] = frame["year"].astype(int)
    frame["yield_ton_ha"] = frame["yield_ton_ha"].astype(float)
    if "recorded_at" not in frame.columns:
        frame["recorded_at"] = ""
    return frame[columns]


def _read_horizon(store: Store) -> int | None:
    if not store.horizon_file.exists():
        return None
    try:
        payload = json.loads(store.horizon_file.read_text(encoding="utf-8"))
        return int(payload["through_year"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _history_map(frame: pd.DataFrame) -> dict[str, list[tuple[int, float]]]:
    ordered = frame.sort_values(["district", "year"])
    grouped: dict[str, list[tuple[int, float]]] = {}
    for district, year, yield_value in zip(
        ordered["district"],
        ordered["year"],
        ordered["yield_ton_ha"],
    ):
        grouped.setdefault(str(district), []).append(
            (int(year), float(yield_value))
        )
    return grouped


def _drift(anchor: pd.Series, year: int, anchor_year: int) -> pd.Series:
    """One future climate year from the 2023 district row, not from the year before."""

    step = int(year) - int(anchor_year)
    row = anchor.copy()
    district = str(row["district"])
    row["year"] = int(year)
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
    row["rainfall_total"] = max(5.0, float(row["rainfall_total"]) * rain_factor)
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
    return row


def _climate_context(boro: pd.DataFrame) -> dict:
    anchor = _anchor_year(boro)
    history = boro[boro["year"] <= anchor]
    district_mean = history.groupby("district")["yield_ton_ha"].mean()
    national = float(district_mean.mean())
    factors = {
        str(district): float(value) / national
        for district, value in district_mean.items()
    }
    anchor_rows = {
        str(row["district"]): row
        for _, row in boro[boro["year"] == anchor].iterrows()
    }
    return {
        "anchor": anchor,
        "anchor_rows": anchor_rows,
        "factors": factors,
        "rain_mean": float(history["rainfall_total"].mean()),
        "rain_std": float(history["rainfall_total"].std()),
        "temp_mean": float(history["temp_max_mean"].mean()),
        "temp_std": float(history["temp_max_mean"].std()),
    }


def _continued_yield(
    history: list[tuple[int, float]],
    crop_id: str,
    district: str,
    year: int,
) -> float:
    """Next harvest from this district's own recent seasons."""

    history = sorted(history)
    last_year, last_yield = history[-1]
    recent = history[-6:]
    if len(recent) >= 2 and recent[-1][0] != recent[0][0]:
        slope = (recent[-1][1] - recent[0][1]) / (
            recent[-1][0] - recent[0][0]
        )
    else:
        slope = 0.0
    projected = (
        last_yield
        + slope * (int(year) - int(last_year))
        + _noise(crop_id, district, year, 0.08)
    )
    floor = float(CROP_CATALOG[crop_id].get("floor", 0.3))
    ceiling = max(floor + 1.0, last_yield * 1.5, last_yield + 2.0)
    return round(float(min(max(projected, floor), ceiling)), 2)


def _rows_for_year(
    crop_id: str,
    existing: pd.DataFrame,
    year: int,
    context: dict,
) -> pd.DataFrame:
    spec = CROP_CATALOG[crop_id]
    history = _history_map(existing)
    records = []
    for district, anchor_row in context["anchor_rows"].items():
        if district not in history:
            raise ValueError(
                f"{spec['label']} has no harvest history for {district}."
            )
        drifted = _drift(anchor_row, year, context["anchor"])
        seasoned = _apply_season(drifted, spec)
        seasoned["crop"] = crop_id
        seasoned["year"] = int(year)
        if spec["yield_source"] == "bbs":
            seasoned["yield_ton_ha"] = _continued_yield(
                history[district],
                crop_id,
                district,
                year,
            )
        else:
            seasoned["yield_ton_ha"] = _sample_yield(
                seasoned,
                crop_id,
                spec,
                context["rain_mean"],
                context["rain_std"],
                context["temp_mean"],
                context["temp_std"],
                context["factors"].get(district, 1.0),
            )
        records.append(seasoned)
    return pd.DataFrame(records)


def _provenance_map(frame: pd.DataFrame) -> dict[tuple[int, str, str], str]:
    if frame.empty:
        return {}
    return {
        (int(row.year), str(row.district), str(row.crop)): str(row.source)
        for row in frame.itertuples(index=False)
    }


def _provenance_frame(
    mapping: dict[tuple[int, str, str], str],
) -> pd.DataFrame:
    rows = [
        {
            "year": year,
            "district": district,
            "crop": crop,
            "source": source,
        }
        for (year, district, crop), source in mapping.items()
    ]
    frame = pd.DataFrame(
        rows,
        columns=["year", "district", "crop", "source"],
    )
    if frame.empty:
        return frame
    return frame.sort_values(["year", "district", "crop"])


def _overlay(
    frames: dict[str, pd.DataFrame],
    provenance: dict[tuple[int, str, str], str],
    official: pd.DataFrame,
) -> int:
    """Replace stored yields with official numbers for years already on file."""

    applied = 0
    if official.empty:
        return applied
    for row in official.itertuples(index=False):
        crop_id = str(row.crop)
        frame = frames.get(crop_id)
        if frame is None:
            continue
        year = int(row.year)
        district = str(row.district)
        mask = (frame["district"] == district) & (frame["year"] == year)
        if not bool(mask.any()):
            continue
        frame.loc[mask, "yield_ton_ha"] = round(float(row.yield_ton_ha), 2)
        provenance[(year, district, crop_id)] = "official"
        applied += 1
    return applied


def _provided_fields(record, fields: list[str]) -> dict[str, float]:
    provided = {}
    for field in fields:
        value = getattr(record, field, None)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if pd.isna(value):
            continue
        provided[field] = float(value)
    return provided


def _overlay_weather(
    frames: dict[str, pd.DataFrame],
    weather: pd.DataFrame,
) -> int:
    """Write measured weather onto every crop for that district and year."""

    if weather.empty:
        return 0
    touched = 0
    for record in weather.itertuples(index=False):
        provided = _provided_fields(record, WEATHER_FIELDS)
        if not provided:
            continue
        year = int(record.year)
        district = str(record.district)
        for crop_id, frame in frames.items():
            mask = (frame["district"].astype(str) == district) & (
                frame["year"] == year
            )
            indexes = frame.index[mask]
            if len(indexes) == 0:
                continue
            spec = CROP_CATALOG[crop_id]
            for idx in indexes:
                old_rain = float(frame.at[idx, "rainfall_total"])
                old_temp = float(frame.at[idx, "temp_max_mean"])
                for field, value in provided.items():
                    frame.at[idx, field] = value
                seasoned = _apply_season(frame.loc[idx], spec)
                if "rainfall_total" in provided:
                    if spec.get("season_rain_share") is not None:
                        frame.at[idx, "rainfall_mar_jun"] = float(
                            seasoned["rainfall_mar_jun"]
                        )
                    elif old_rain > 0:
                        frame.at[idx, "rainfall_mar_jun"] = max(
                            1.0,
                            float(frame.at[idx, "rainfall_mar_jun"])
                            * (provided["rainfall_total"] / old_rain),
                        )
                if "temp_max_mean" in provided:
                    if spec.get("season_temp_offset") is not None:
                        frame.at[idx, "temp_max_mar_jun"] = float(
                            seasoned["temp_max_mar_jun"]
                        )
                    else:
                        frame.at[idx, "temp_max_mar_jun"] = float(
                            frame.at[idx, "temp_max_mar_jun"]
                        ) + (provided["temp_max_mean"] - old_temp)
            touched += len(indexes)
    return touched


def _overlay_soil(
    frames: dict[str, pd.DataFrame],
    soil: pd.DataFrame,
) -> int:
    """Apply each soil test from its year forward. Older years stay put."""

    if soil.empty:
        return 0
    touched = 0
    ordered = soil.sort_values(["year", "district"])
    for record in ordered.itertuples(index=False):
        provided = _provided_fields(record, SOIL_FIELDS)
        if not provided:
            continue
        start = int(record.year)
        district = str(record.district)
        for frame in frames.values():
            mask = (frame["district"].astype(str) == district) & (
                frame["year"] >= start
            )
            if not bool(mask.any()):
                continue
            for field, value in provided.items():
                frame.loc[mask, field] = value
            touched += int(mask.sum())
    return touched


def _overlay_greenness(
    frames: dict[str, pd.DataFrame],
    greenness: pd.DataFrame,
) -> int:
    """Write measured greenness onto every crop for that district and year."""

    if greenness.empty:
        return 0
    touched = 0
    for record in greenness.itertuples(index=False):
        provided = _provided_fields(record, GREENNESS_FIELDS)
        if not provided:
            continue
        year = int(record.year)
        district = str(record.district)
        for frame in frames.values():
            mask = (frame["district"].astype(str) == district) & (
                frame["year"] == year
            )
            if not bool(mask.any()):
                continue
            for field, value in provided.items():
                frame.loc[mask, field] = value
            touched += int(mask.sum())
    return touched


def _refresh_season_fields(
    frame: pd.DataFrame,
    idx: int,
    spec: dict,
    provided: dict[str, float],
    old_rain: float,
    old_temp: float,
) -> None:
    """Keep the growing-season columns in step when only the year total changed."""

    if "rainfall_total" in provided and "rainfall_mar_jun" not in provided:
        seasoned = _apply_season(frame.loc[idx], spec)
        if spec.get("season_rain_share") is not None:
            frame.at[idx, "rainfall_mar_jun"] = float(seasoned["rainfall_mar_jun"])
        elif old_rain > 0:
            frame.at[idx, "rainfall_mar_jun"] = max(
                1.0,
                float(frame.at[idx, "rainfall_mar_jun"])
                * (provided["rainfall_total"] / old_rain),
            )
    if "temp_max_mean" in provided and "temp_max_mar_jun" not in provided:
        seasoned = _apply_season(frame.loc[idx], spec)
        if spec.get("season_temp_offset") is not None:
            frame.at[idx, "temp_max_mar_jun"] = float(seasoned["temp_max_mar_jun"])
        else:
            frame.at[idx, "temp_max_mar_jun"] = float(
                frame.at[idx, "temp_max_mar_jun"]
            ) + (provided["temp_max_mean"] - old_temp)


def _overlay_crop_rows(
    frames: dict[str, pd.DataFrame],
    updates: pd.DataFrame,
) -> int:
    """Write filled cells onto that crop, district, and year only."""

    if updates.empty:
        return 0
    touched = 0
    for record in updates.itertuples(index=False):
        provided = _provided_fields(record, ROW_FIELDS)
        if not provided:
            continue
        crop_id = str(record.crop)
        frame = frames.get(crop_id)
        if frame is None:
            continue
        mask = (frame["district"].astype(str) == str(record.district)) & (
            frame["year"] == int(record.year)
        )
        indexes = frame.index[mask]
        if len(indexes) == 0:
            continue
        spec = CROP_CATALOG[crop_id]
        for idx in indexes:
            old_rain = float(frame.at[idx, "rainfall_total"])
            old_temp = float(frame.at[idx, "temp_max_mean"])
            for field, value in provided.items():
                frame.at[idx, field] = value
            _refresh_season_fields(
                frame,
                idx,
                spec,
                provided,
                old_rain,
                old_temp,
            )
        touched += len(indexes)
    return touched


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _save_frames(store: Store, frames: dict[str, pd.DataFrame]) -> None:
    written: list[tuple[Path, Path]] = []
    for crop_id, frame in frames.items():
        ordered = (
            frame[DATASET_COLUMNS]
            .drop_duplicates(["district", "year"], keep="last")
            .sort_values(["year", "district"])
        )
        ordered["year"] = ordered["year"].astype(int)
        target = store.crop_dir / f"{crop_id}.csv"
        temporary = store.crop_dir / f"{crop_id}.csv.tmp"
        ordered.to_csv(temporary, index=False)
        written.append((temporary, target))
    for temporary, target in written:
        temporary.replace(target)


def _save_catalog(store: Store, frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for crop_id, spec in CROP_CATALOG.items():
        frame = frames[crop_id]
        rows.append(
            {
                "crop": crop_id,
                "label": spec["label"],
                "records": int(len(frame)),
                "districts": int(frame["district"].nunique()),
                "start_year": int(frame["year"].min()),
                "end_year": int(frame["year"].max()),
                "mean_yield": round(float(frame["yield_ton_ha"].mean()), 2),
            }
        )
    _write_table(pd.DataFrame(rows), store.catalog_file)


def _save_horizon(store: Store, year: int) -> None:
    store.intake_dir.mkdir(parents=True, exist_ok=True)
    payload = {"through_year": int(year), "updated_at": _now()}
    store.horizon_file.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _publish(frames: dict[str, pd.DataFrame]) -> str:
    if not mysql_configured():
        return "Project files are current. MySQL is not configured."
    try:
        if not use_mysql():
            return (
                "Project files are current. MySQL did not accept a connection, "
                "so the planner will keep using project files."
            )
    except Exception:
        return (
            "Project files are current. MySQL did not accept a connection, "
            "so the planner will keep using project files."
        )
    try:
        publish_datasets(list(frames.values()))
        publish_trained_models()
    except Exception:
        return (
            "Project files are current. MySQL was not updated, "
            "so a planner that reads the database still has the previous season."
        )
    return "Project files and MySQL are current."


def _sentence(
    opened: list[int],
    official_applied: int,
    continued_rows: int,
    latest: int,
    storage_message: str,
) -> str:
    parts = []
    if opened:
        labels = ", ".join(str(year) for year in opened)
        parts.append(f"Opened {labels}.")
    if official_applied:
        noun = "yield" if official_applied == 1 else "yields"
        parts.append(f"Stored {official_applied} official {noun}.")
    if continued_rows:
        parts.append(
            f"Continued {continued_rows} district-crop rows from their own record."
        )
    if opened or continued_rows:
        parts.append(f"The planner now uses {latest} as the latest season.")
    elif official_applied:
        parts.append(f"The latest season remains {latest}.")
    parts.append(storage_message)
    return " ".join(parts)


def _sync(
    store: Store,
    target_year: int,
    official: pd.DataFrame,
    *,
    publish: bool,
    log: bool,
    actor_id: int,
    actor_name: str,
    action: str,
    official_applied_limit: pd.DataFrame | None = None,
) -> dict:
    """Extend the registry through target_year and stamp official yields."""

    frames = _load_frames(store)
    current = _max_year(frames)
    if target_year > current + MAX_AHEAD:
        raise ValueError(
            f"The registry ends at {current}. "
            f"Open at most {MAX_AHEAD} new seasons at a time."
        )
    if target_year > 2100:
        raise ValueError("Choose a season before 2101.")

    provenance = _provenance_map(_load_provenance(store))
    context = _climate_context(frames["boro_rice"])
    opened: list[int] = []
    continued_rows = 0

    _overlay(frames, provenance, official)

    year = current
    while year < target_year:
        year += 1
        fresh: list[tuple[int, str, str]] = []
        for crop_id in CROP_CATALOG:
            addition = _rows_for_year(
                crop_id,
                frames[crop_id],
                year,
                context,
            )
            frames[crop_id] = pd.concat(
                [frames[crop_id], addition],
                ignore_index=True,
            )
            for district in addition["district"].astype(str):
                key = (year, district, crop_id)
                provenance[key] = "continued"
                fresh.append(key)
        if not official.empty:
            _overlay(
                frames,
                provenance,
                official[official["year"] == year],
            )
        continued_rows += sum(
            1 for key in fresh if provenance.get(key) == "continued"
        )
        opened.append(year)

    counted = official if official_applied_limit is None else official_applied_limit
    official_applied = 0
    if not counted.empty:
        for row in counted.itertuples(index=False):
            crop_id = str(row.crop)
            frame = frames.get(crop_id)
            if frame is None:
                continue
            district = str(row.district)
            row_year = int(row.year)
            mask = (frame["district"] == district) & (frame["year"] == row_year)
            if bool(mask.any()):
                official_applied += 1
                provenance[(row_year, district, crop_id)] = "official"

    weather_rows = _overlay_weather(frames, _read_weather(store))
    soil_rows = _overlay_soil(frames, _read_soil(store))
    greenness_rows = _overlay_greenness(frames, _read_greenness(store))
    crop_rows = _overlay_crop_rows(frames, _read_crop_rows(store))

    latest = _max_year(frames)
    storage_message = "Project files are current."
    if (
        opened
        or official_applied
        or not counted.empty
        or weather_rows
        or soil_rows
        or greenness_rows
        or crop_rows
    ):
        _save_frames(store, frames)
        _save_catalog(store, frames)
        _write_table(_provenance_frame(provenance), store.provenance_file)
        if not official.empty:
            _write_table(
                official.sort_values(["year", "district", "crop"]),
                store.official_file,
            )
        _save_horizon(store, latest)
        if publish:
            storage_message = _publish(frames)

    message = _sentence(
        opened,
        official_applied,
        continued_rows,
        latest,
        storage_message,
    )
    if log and (opened or official_applied):
        log_admin_event(actor_id, actor_name, action, message)
    return {
        "changed": bool(opened or official_applied),
        "opened": opened,
        "official_applied": official_applied,
        "continued_rows": continued_rows,
        "latest_year": latest,
        "storage_message": storage_message,
        "message": message,
    }


def _merge_official(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    if incoming.empty:
        return existing
    stamped = incoming.copy()
    stamped["recorded_at"] = _now()
    if existing.empty:
        return stamped.reset_index(drop=True)
    frame = pd.concat([existing, stamped], ignore_index=True)
    return frame.drop_duplicates(
        ["year", "district", "crop"],
        keep="last",
    )


def apply_upload(
    rows: pd.DataFrame,
    actor_id: int,
    actor_name: str,
    store: Store | None = None,
    *,
    publish: bool = True,
    log: bool = True,
) -> dict:
    """Store an official file and open any season it needs."""

    store = store or default_store()
    if rows is None or rows.empty:
        raise ValueError("The file has no yield numbers to store.")
    frames = _load_frames(store)
    current = _max_year(frames)
    earliest = _min_year(frames)
    years = sorted(int(year) for year in rows["year"].unique())
    if years[0] < earliest:
        raise ValueError(
            f"The registry starts in {earliest}. "
            f"{years[0]} is earlier than that."
        )
    target = max(current, years[-1])
    archive = _merge_official(_read_official(store), rows)
    return _sync(
        store,
        target,
        archive,
        publish=publish,
        log=log,
        actor_id=actor_id,
        actor_name=actor_name,
        action="results_uploaded",
        official_applied_limit=rows,
    )


def open_seasons(
    count: int,
    actor_id: int,
    actor_name: str,
    store: Store | None = None,
    *,
    publish: bool = True,
    log: bool = True,
) -> dict:
    """Continue the registry by one or more seasons with no new official file."""

    count = int(count)
    if count < 1 or count > MAX_OPEN:
        raise ValueError(f"Open between 1 and {MAX_OPEN} seasons at a time.")
    store = store or default_store()
    frames = _load_frames(store)
    target = _max_year(frames) + count
    return _sync(
        store,
        target,
        _read_official(store),
        publish=publish,
        log=log,
        actor_id=actor_id,
        actor_name=actor_name,
        action="season_opened",
        official_applied_limit=pd.DataFrame(
            columns=["year", "district", "crop", "yield_ton_ha"]
        ),
    )


def replay_intakes(
    publish: bool = False,
    log: bool = False,
    store: Store | None = None,
) -> dict:
    """Reapply seasons saved by the console after a dataset rebuild."""

    store = store or default_store()
    if not any(
        path.exists()
        for path in (
            store.horizon_file,
            store.official_file,
            store.weather_file,
            store.soil_file,
            store.greenness_file,
            store.crop_rows_file,
        )
    ):
        return {
            "changed": False,
            "opened": [],
            "latest_year": None,
            "message": "",
        }
    frames = _load_frames(store)
    current = _max_year(frames)
    official = _read_official(store)
    horizon = _read_horizon(store)
    target = current if horizon is None else max(current, horizon)
    if not official.empty:
        target = max(target, int(official["year"].max()))
    if (
        target <= current
        and official.empty
        and not store.weather_file.exists()
        and not store.soil_file.exists()
        and not store.greenness_file.exists()
        and not store.crop_rows_file.exists()
    ):
        return {
            "changed": False,
            "opened": [],
            "latest_year": current,
            "message": "",
        }
    return _sync(
        store,
        target,
        official,
        publish=publish,
        log=log,
        actor_id=0,
        actor_name="Dataset build",
        action="season_opened",
    )


def registry_stamp(store: Store | None = None) -> str:
    """Changes when a season file or its provenance changes."""

    store = store or default_store()
    stamps = []
    for name in ("boro_rice.csv", "provenance.csv", "catalog.csv"):
        path = store.crop_dir / name
        if path.exists():
            stamps.append(path.stat().st_mtime)
    for path in (
        store.weather_file,
        store.soil_file,
        store.greenness_file,
        store.crop_rows_file,
    ):
        if path.exists():
            stamps.append(path.stat().st_mtime)
    if not stamps:
        return "0"
    return str(max(stamps))


def season_status(store: Store | None = None) -> dict:
    """Coverage the operations overview shows."""

    store = store or default_store()
    frames = _load_frames(store)
    provenance = _load_provenance(store)
    latest = _max_year(frames)
    earliest = _min_year(frames)
    districts = int(frames["boro_rice"]["district"].nunique())
    crops = len(frames)
    year_rows = []
    for year in range(earliest, latest + 1):
        official = 0
        continued = 0
        if not provenance.empty:
            subset = provenance[provenance["year"] == year]
            official = int((subset["source"] == "official").sum())
            continued = int((subset["source"] == "continued").sum())
        records = sum(int((frame["year"] == year).sum()) for frame in frames.values())
        if year == latest:
            role = "Current"
        elif official or continued:
            role = "Opened here"
        else:
            role = "On file"
        year_rows.append(
            {
                "Year": year,
                "Records": records,
                "Official yields": official,
                "Continued yields": continued,
                "Role": role,
            }
        )
    ready = sum(
        1
        for crop_id in CROP_CATALOG
        if (crop_model_dir(crop_id) / "best_model.pkl").exists()
    )
    window = ""
    meta_path = crop_model_dir("boro_rice") / "model_metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        start, end = metadata.get("year_range", [None, None])
        if start is not None:
            window = f"{int(start)}–{int(end)}"
    if mysql_configured():
        try:
            storage = "MySQL" if use_mysql() else "Project files"
        except Exception:
            storage = "Project files"
    else:
        storage = "Project files"
    return {
        "start_year": earliest,
        "latest_year": latest,
        "next_year": latest + 1,
        "districts": districts,
        "crops": crops,
        "years": year_rows,
        "models_ready": ready,
        "model_window": window,
        "storage": storage,
    }


def national_means(crop_id: str, store: Store | None = None) -> pd.DataFrame:
    store = store or default_store()
    frame = _read_crop(store, crop_id)
    grouped = (
        frame.groupby("year", as_index=False)["yield_ton_ha"]
        .mean()
        .sort_values("year")
    )
    grouped["year"] = grouped["year"].astype(int)
    grouped["mean_yield"] = grouped["yield_ton_ha"].round(2)
    return grouped[["year", "mean_yield"]]


def catalog_frame(store: Store | None = None) -> pd.DataFrame:
    store = store or default_store()
    rows = []
    for crop_id, spec in CROP_CATALOG.items():
        frame = _read_crop(store, crop_id)
        rows.append(
            {
                "Crop": spec["label"],
                "Group": spec["group"],
                "Season": spec["season"],
                "Rows": int(len(frame)),
                "Districts": int(frame["district"].nunique()),
                "Years": (
                    f"{int(frame['year'].min())}–{int(frame['year'].max())}"
                ),
                "Mean yield (t/ha)": round(float(frame["yield_ton_ha"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def _fields_for(kind: str) -> list[str]:
    return {
        "weather": WEATHER_FIELDS,
        "soil": SOIL_FIELDS,
        "greenness": GREENNESS_FIELDS,
    }[kind]


def _limits_for(kind: str) -> dict[str, tuple[float, float, str]]:
    return {
        "weather": WEATHER_LIMITS,
        "soil": SOIL_LIMITS,
        "greenness": GREENNESS_LIMITS,
    }[kind]


def _measurement_columns(kind: str) -> list[str]:
    return ["year", "district", *_fields_for(kind), "recorded_at"]


def _read_measurement(path: Path, kind: str) -> pd.DataFrame:
    columns = _measurement_columns(kind)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["year"] = frame["year"].astype(int)
    frame["district"] = frame["district"].astype(str)
    fields = _fields_for(kind)
    for field in fields:
        if field not in frame.columns:
            frame[field] = pd.NA
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "recorded_at" not in frame.columns:
        frame["recorded_at"] = ""
    return frame[columns]


def _read_weather(store: Store) -> pd.DataFrame:
    return _read_measurement(store.weather_file, "weather")


def _read_soil(store: Store) -> pd.DataFrame:
    return _read_measurement(store.soil_file, "soil")


def _read_greenness(store: Store) -> pd.DataFrame:
    return _read_measurement(store.greenness_file, "greenness")


def _merge_partial(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    fields: list[str],
) -> pd.DataFrame:
    """Newer numbers replace only the cells that were filled in."""

    if incoming.empty:
        return existing
    stamped = incoming.copy()
    stamped["recorded_at"] = _now()
    if existing.empty:
        return stamped.reset_index(drop=True)
    base = existing.drop_duplicates(
        ["year", "district"],
        keep="last",
    ).set_index(["year", "district"])
    fresh = stamped.drop_duplicates(
        ["year", "district"],
        keep="last",
    ).set_index(["year", "district"])
    for key, row in fresh.iterrows():
        if key not in base.index:
            base.loc[key] = row
            continue
        for field in fields:
            if pd.notna(row[field]):
                base.loc[key, field] = row[field]
        base.loc[key, "recorded_at"] = row["recorded_at"]
    return base.reset_index()


def _optional_number(
    value: object,
    line: int,
    label: str,
    low: float,
    high: float,
    errors: list[str],
) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip().replace(" ", "")
    if "," in text and "." not in text:
        errors.append(f"Line {line}: use a dot for decimals, not a comma.")
        return None
    try:
        number = float(text)
    except ValueError:
        errors.append(f"Line {line}: {label} is not a number.")
        return None
    if not (low <= number <= high):
        errors.append(
            f"Line {line}: {label} must be between {low:g} and {high:g}."
        )
        return None
    return number


def _districts(store: Store) -> dict[str, str]:
    boro = _read_crop(store, "boro_rice")
    return {
        _norm(name): str(name)
        for name in boro["district"].astype(str).unique()
    }


def _parse_measurement(
    payload: bytes,
    store: Store,
    kind: str,
) -> UploadReport:
    fields = _fields_for(kind)
    limits = _limits_for(kind)
    empty = pd.DataFrame(columns=["year", "district", *fields])
    if not payload or not payload.strip():
        return UploadReport(empty, ["The file is empty."], [], [], 0, 0)
    try:
        raw = _canonical_headers(_read_upload_frame(payload))
    except ValueError as error:
        return UploadReport(empty, [str(error)], [], [], 0, 0)
    missing = [name for name in ("year", "district") if name not in raw.columns]
    present = [field for field in fields if field in raw.columns]
    if missing or not present:
        return UploadReport(
            empty,
            [
                "The CSV needs year, district, and at least one "
                "measurement column."
            ],
            [],
            [],
            0,
            int(len(raw)),
        )
    frames = _load_frames(store)
    earliest = _min_year(frames)
    current = _max_year(frames)
    known = _districts(store)
    kept = []
    row_errors = []
    warnings = []
    blank_rows = 0
    seen: set[tuple[int, str]] = set()
    for index, row in raw.iterrows():
        line = int(index) + 2
        try:
            year = int(float(row["year"]))
        except (TypeError, ValueError):
            row_errors.append(f"Line {line}: year is not a number.")
            continue
        if year < earliest or year > current + MAX_AHEAD:
            row_errors.append(
                f"Line {line}: {year} is outside "
                f"{earliest}–{current + MAX_AHEAD}."
            )
            continue
        district = known.get(_norm(row["district"]))
        if district is None:
            row_errors.append(
                f"Line {line}: unknown district '{row['district']}'."
            )
            continue
        values = {}
        failed = False
        for field in present:
            low, high, label = limits[field]
            number = _optional_number(
                row[field],
                line,
                label,
                low,
                high,
                row_errors,
            )
            if (
                number is None
                and not pd.isna(row[field])
                and str(row[field]).strip() != ""
            ):
                failed = True
            values[field] = number
        if failed:
            continue
        if all(number is None for number in values.values()):
            blank_rows += 1
            continue
        if "temp_max_mean" in values and "temp_min_mean" in values:
            high_temp = values["temp_max_mean"]
            low_temp = values["temp_min_mean"]
            if (
                high_temp is not None
                and low_temp is not None
                and low_temp > high_temp
            ):
                row_errors.append(
                    f"Line {line}: minimum temperature is above "
                    "the maximum."
                )
                continue
        if "ndvi_mean" in values and "ndvi_peak" in values:
            average = values["ndvi_mean"]
            peak = values["ndvi_peak"]
            if average is not None and peak is not None and peak < average:
                row_errors.append(
                    f"Line {line}: peak greenness is below the average."
                )
                continue
        key = (year, district)
        if key in seen:
            warnings.append(
                f"Line {line}: {district} {year} repeats an earlier row. "
                "The last one is kept."
            )
        seen.add(key)
        item = {"year": year, "district": district}
        for field in fields:
            item[field] = values.get(field)
        kept.append(item)
    rows = pd.DataFrame(kept, columns=["year", "district", *fields])
    if not rows.empty:
        rows = rows.drop_duplicates(["year", "district"], keep="last")
    return UploadReport(
        rows,
        [],
        row_errors,
        warnings,
        blank_rows,
        int(len(raw)),
    )


def parse_weather(payload: bytes, store: Store | None = None) -> UploadReport:
    """Check a district weather file without writing it."""

    return _parse_measurement(payload, store or default_store(), "weather")


def parse_soil(payload: bytes, store: Store | None = None) -> UploadReport:
    """Check a district soil file without writing it."""

    return _parse_measurement(payload, store or default_store(), "soil")


def parse_greenness(payload: bytes, store: Store | None = None) -> UploadReport:
    """Check a district greenness file without writing it."""

    return _parse_measurement(payload, store or default_store(), "greenness")


def _apply_measurement(
    rows: pd.DataFrame,
    actor_id: int,
    actor_name: str,
    store: Store,
    kind: str,
    *,
    publish: bool,
    log: bool,
) -> dict:
    if rows is None or rows.empty:
        raise ValueError("The file has no numbers to store.")
    fields = _fields_for(kind)
    path = {
        "weather": store.weather_file,
        "soil": store.soil_file,
        "greenness": store.greenness_file,
    }[kind]
    existing = {
        "weather": _read_weather(store),
        "soil": _read_soil(store),
        "greenness": _read_greenness(store),
    }[kind]
    merged = _merge_partial(existing, rows, fields)
    _write_table(
        merged.sort_values(["year", "district"]),
        path,
    )
    frames = _load_frames(store)
    current = _max_year(frames)
    incoming_max = int(rows["year"].max())
    if incoming_max > current:
        summary = _sync(
            store,
            incoming_max,
            _read_official(store),
            publish=publish,
            log=False,
            actor_id=actor_id,
            actor_name=actor_name,
            action="season_opened",
        )
    else:
        if kind == "weather":
            touched = _overlay_weather(frames, merged)
        elif kind == "soil":
            touched = _overlay_soil(frames, merged)
        else:
            touched = _overlay_greenness(frames, merged)
        _save_frames(store, frames)
        _save_catalog(store, frames)
        storage = _publish(frames) if publish else "Project files are current."
        noun = {
            "weather": "weather rows",
            "soil": "soil tests",
            "greenness": "greenness rows",
        }[kind]
        summary = {
            "changed": touched > 0,
            "opened": [],
            "official_applied": 0,
            "continued_rows": 0,
            "latest_year": current,
            "storage_message": storage,
            "message": (
                f"Stored {len(rows)} {noun}. "
                "Blank cells stayed carried forward. "
                f"The model was not retrained. {storage}"
            ),
        }
    if kind == "weather" and incoming_max <= current:
        pass
    elif incoming_max > current:
        label = {"weather": "Weather", "soil": "Soil", "greenness": "Greenness"}[kind]
        summary["message"] = (
            f"{summary['message']} {label} measurements were applied. "
            "The model was not retrained."
        )
    if log:
        log_admin_event(
            actor_id,
            actor_name,
            {
                "weather": "weather_uploaded",
                "soil": "soil_uploaded",
                "greenness": "greenness_uploaded",
            }[kind],
            summary["message"],
        )
    return summary


def apply_weather(
    rows: pd.DataFrame,
    actor_id: int,
    actor_name: str,
    store: Store | None = None,
    *,
    publish: bool = True,
    log: bool = True,
) -> dict:
    """Store district weather and copy it onto every crop for that year."""

    return _apply_measurement(
        rows,
        actor_id,
        actor_name,
        store or default_store(),
        "weather",
        publish=publish,
        log=log,
    )


def apply_soil(
    rows: pd.DataFrame,
    actor_id: int,
    actor_name: str,
    store: Store | None = None,
    *,
    publish: bool = True,
    log: bool = True,
) -> dict:
    """Store a soil test and use it from that year forward."""

    return _apply_measurement(
        rows,
        actor_id,
        actor_name,
        store or default_store(),
        "soil",
        publish=publish,
        log=log,
    )


def apply_greenness(
    rows: pd.DataFrame,
    actor_id: int,
    actor_name: str,
    store: Store | None = None,
    *,
    publish: bool = True,
    log: bool = True,
) -> dict:
    """Store district greenness and copy it onto every crop for that year."""

    return _apply_measurement(
        rows,
        actor_id,
        actor_name,
        store or default_store(),
        "greenness",
        publish=publish,
        log=log,
    )


def _crop_row_columns() -> list[str]:
    return ["year", "district", "crop", *ROW_FIELDS, "recorded_at"]


def _read_crop_rows(store: Store) -> pd.DataFrame:
    columns = _crop_row_columns()
    path = store.crop_rows_file
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["year"] = frame["year"].astype(int)
    frame["district"] = frame["district"].astype(str)
    frame["crop"] = frame["crop"].astype(str)
    for field in ROW_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "recorded_at" not in frame.columns:
        frame["recorded_at"] = ""
    return frame[columns]


def _merge_crop_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """A later file replaces only the cells that were filled in."""

    if incoming.empty:
        return existing
    stamped = incoming.copy()
    stamped["recorded_at"] = _now()
    for field in ROW_FIELDS:
        if field not in stamped.columns:
            stamped[field] = pd.NA
    records: dict[tuple[int, str, str], dict] = {}
    if not existing.empty:
        for record in existing.itertuples(index=False):
            key = (int(record.year), str(record.district), str(record.crop))
            records[key] = {
                field: getattr(record, field)
                for field in (*ROW_FIELDS, "recorded_at")
            }
    for record in stamped.itertuples(index=False):
        key = (int(record.year), str(record.district), str(record.crop))
        current = records.get(key)
        if current is None:
            current = {field: pd.NA for field in ROW_FIELDS}
        current["recorded_at"] = record.recorded_at
        for field in ROW_FIELDS:
            value = getattr(record, field)
            if pd.notna(value):
                current[field] = value
        records[key] = current
    rows = []
    for (year, district, crop), values in records.items():
        rows.append(
            {"year": year, "district": district, "crop": crop, **values}
        )
    return pd.DataFrame(rows, columns=_crop_row_columns())


def parse_crop_rows(payload: bytes, store: Store | None = None) -> UploadReport:
    """Check one crop's harvest file. Weather and soil columns are ignored."""

    store = store or default_store()
    empty = pd.DataFrame(columns=["year", "district", "crop", *ROW_FIELDS])
    if not payload or not payload.strip():
        return UploadReport(empty, ["The file is empty."], [], [], 0, 0)
    try:
        raw = _canonical_headers(_read_upload_frame(payload))
    except ValueError as error:
        return UploadReport(empty, [str(error)], [], [], 0, 0)
    missing = [
        name
        for name in ("year", "district", "crop", "yield_ton_ha")
        if name not in raw.columns
    ]
    if missing:
        return UploadReport(
            empty,
            [
                "This upload needs year, district, crop, and yield_ton_ha. "
                "Weather and soil have their own uploads."
            ],
            [],
            [],
            0,
            int(len(raw)),
        )
    frames = _load_frames(store)
    earliest = _min_year(frames)
    current = _max_year(frames)
    known = _districts(store)
    crops = _crop_lookup()
    kept = []
    row_errors = []
    warnings = []
    blank_rows = 0
    seen: set[tuple[int, str, str]] = set()
    for index, row in raw.iterrows():
        line = int(index) + 2
        try:
            year = int(float(row["year"]))
        except (TypeError, ValueError):
            row_errors.append(f"Line {line}: year is not a number.")
            continue
        if year < earliest or year > current + MAX_AHEAD:
            row_errors.append(
                f"Line {line}: {year} is outside "
                f"{earliest}–{current + MAX_AHEAD}."
            )
            continue
        district = known.get(_norm(row["district"]))
        if district is None:
            row_errors.append(
                f"Line {line}: unknown district '{row['district']}'."
            )
            continue
        crop_id = crops.get(_norm(row["crop"]))
        if crop_id is None:
            row_errors.append(f"Line {line}: unknown crop '{row['crop']}'.")
            continue
        low, high, label = ROW_LIMITS["yield_ton_ha"]
        yield_value = _optional_number(
            row["yield_ton_ha"],
            line,
            label,
            low,
            high,
            row_errors,
        )
        if yield_value is None:
            if (
                not pd.isna(row["yield_ton_ha"])
                and str(row["yield_ton_ha"]).strip() != ""
            ):
                continue
            blank_rows += 1
            continue
        key = (year, district, crop_id)
        if key in seen:
            warnings.append(
                f"Line {line}: {district} {year} repeats an earlier row. "
                "The last one is kept."
            )
        seen.add(key)
        item = {"year": year, "district": district, "crop": crop_id}
        for field in ROW_FIELDS:
            item[field] = None
        item["yield_ton_ha"] = yield_value
        kept.append(item)
    if len(row_errors) > 12:
        hidden = len(row_errors) - 12
        row_errors = row_errors[:12]
        row_errors.append(f"{hidden} more rows were skipped.")
    rows = pd.DataFrame(kept, columns=["year", "district", "crop", *ROW_FIELDS])
    if not rows.empty:
        rows = rows.drop_duplicates(["year", "district", "crop"], keep="last")
    return UploadReport(
        rows,
        [],
        row_errors,
        warnings,
        blank_rows,
        int(len(raw)),
    )


def apply_crop_rows(
    rows: pd.DataFrame,
    actor_id: int,
    actor_name: str,
    store: Store | None = None,
    *,
    publish: bool = True,
    log: bool = True,
) -> dict:
    """Write a crop file onto matching district, crop, and year cells."""

    store = store or default_store()
    if rows is None or rows.empty:
        raise ValueError("The file has no numbers to store.")
    merged = _merge_crop_rows(_read_crop_rows(store), rows)
    _write_table(
        merged.sort_values(["year", "district", "crop"]),
        store.crop_rows_file,
    )
    frames = _load_frames(store)
    current = _max_year(frames)
    incoming_max = int(rows["year"].max())
    if incoming_max > current:
        summary = _sync(
            store,
            incoming_max,
            _read_official(store),
            publish=publish,
            log=False,
            actor_id=actor_id,
            actor_name=actor_name,
            action="season_opened",
        )
        summary["message"] = (
            f"{summary['message']} Harvests in the crop file were stored. "
            "Weather, soil, and greenness were left as they are. "
            "The model was not retrained."
        )
    else:
        touched = _overlay_crop_rows(frames, merged)
        _save_frames(store, frames)
        _save_catalog(store, frames)
        storage = _publish(frames) if publish else "Project files are current."
        summary = {
            "changed": touched > 0,
            "opened": [],
            "official_applied": 0,
            "continued_rows": 0,
            "latest_year": current,
            "storage_message": storage,
            "message": (
                f"Updated {len(rows)} harvests. "
                "Weather, soil, and greenness were left as they are. "
                f"The model was not retrained. {storage}"
            ),
        }
    if log:
        log_admin_event(actor_id, actor_name, "crop_file_uploaded", summary["message"])
    return summary


def _harvest_sheet(frame: pd.DataFrame, year: int, crop_id: str) -> bytes:
    """One crop, one year, harvest only."""

    columns = ["year", "district", "crop", "yield_ton_ha"]
    rows = []
    for record in frame.sort_values("district").itertuples(index=False):
        rows.append(
            {
                "year": int(year),
                "district": record.district,
                "crop": crop_id,
                "yield_ton_ha": (
                    ""
                    if pd.isna(record.yield_ton_ha)
                    else record.yield_ton_ha
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8")


def crop_year_csv(
    year: int,
    crop_id: str,
    store: Store | None = None,
) -> tuple[bytes, bool]:
    """Harvest sheet for one crop and year.

    A year that is not on file yet comes back with the yield left blank.
    The second value is True when the bytes are that blank sheet.
    """

    store = store or default_store()
    frame = _read_crop(store, crop_id)
    subset = frame[frame["year"].astype(int) == int(year)]
    if subset.empty:
        identity = frame[frame["year"] == int(frame["year"].max())].copy()
        identity["yield_ton_ha"] = pd.NA
        return _harvest_sheet(identity, year, crop_id), True
    return _harvest_sheet(subset, year, crop_id), False


def crop_row_template_csv(
    year: int,
    crop_id: str,
    store: Store | None = None,
) -> bytes:
    """Blank harvest sheet for one crop. Weather and soil are not included."""

    payload, _blank = crop_year_csv(year, crop_id, store)
    return payload


def condition_status(store: Store | None = None) -> dict:
    """How many districts have a real weather or soil update."""

    store = store or default_store()
    boro = _read_crop(store, "boro_rice")
    latest = int(boro["year"].max())
    district_names = set(boro["district"].astype(str))
    weather = _read_weather(store)
    weather_measured = 0
    if not weather.empty:
        subset = weather[weather["year"] == latest]
        for record in subset.itertuples(index=False):
            if _provided_fields(record, WEATHER_FIELDS):
                weather_measured += 1
    soil = _read_soil(store)
    soil_measured: set[str] = set()
    if not soil.empty:
        for record in soil.itertuples(index=False):
            if int(record.year) > latest:
                continue
            district = str(record.district)
            if district in district_names and _provided_fields(
                record,
                SOIL_FIELDS,
            ):
                soil_measured.add(district)
    return {
        "latest_year": latest,
        "districts": len(district_names),
        "weather_measured": weather_measured,
        "soil_measured": len(soil_measured),
    }


def _blank_template(year: int, store: Store, columns: list[str]) -> bytes:
    districts = sorted(_districts(store).values())
    rows = []
    for district in districts:
        item = {column: "" for column in columns}
        item["year"] = int(year)
        item["district"] = district
        rows.append(item)
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8")


def weather_template_csv(year: int, store: Store | None = None) -> bytes:
    """Blank weather file: one row per district, measurements left empty."""

    return _blank_template(
        year,
        store or default_store(),
        [
            "year",
            "district",
            "rainfall",
            "maximum_temperature",
            "minimum_temperature",
            "humidity",
            "sunlight",
        ],
    )


def soil_template_csv(year: int, store: Store | None = None) -> bytes:
    """Blank soil file: one row per district, the test year in the first column."""

    return _blank_template(
        year,
        store or default_store(),
        [
            "year",
            "district",
            "ph",
            "organic_carbon",
            "nitrogen",
            "clay",
            "sand",
            "cec",
        ],
    )


def greenness_template_csv(year: int, store: Store | None = None) -> bytes:
    """Blank greenness file: one row per district, readings left empty."""

    return _blank_template(
        year,
        store or default_store(),
        [
            "year",
            "district",
            "average_greenness",
            "peak_greenness",
            "greenness_variation",
        ],
    )


def template_csv(year: int, store: Store | None = None) -> bytes:
    """Blank official file for one season, one row per district and crop."""

    store = store or default_store()
    boro = _read_crop(store, "boro_rice")
    districts = sorted(str(name) for name in boro["district"].unique())
    rows = []
    for district in districts:
        for _crop_id, spec in CROP_CATALOG.items():
            rows.append(
                {
                    "year": int(year),
                    "district": district,
                    "crop": spec["label"],
                    "yield_ton_ha": "",
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def crop_file_bytes(crop_id: str, store: Store | None = None) -> bytes:
    store = store or default_store()
    path = store.crop_dir / f"{crop_id}.csv"
    return path.read_bytes()


def datasets_zip_bytes(store: Store | None = None) -> bytes:
    store = store or default_store()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for crop_id in CROP_CATALOG:
            archive.write(
                store.crop_dir / f"{crop_id}.csv",
                arcname=f"{crop_id}.csv",
            )
    return buffer.getvalue()


def _read_upload_frame(payload: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(payload), encoding=encoding)
        except Exception as error:
            last_error = error
    raise ValueError(
        "That file could not be read as a CSV. "
        "Save it as CSV (UTF-8) from Excel."
    ) from last_error


def _canonical_headers(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = _norm(column).replace(" ", "_")
        renamed[column] = HEADER_ALIASES.get(key, key)
    return frame.rename(columns=renamed)


def parse_upload(payload: bytes, store: Store | None = None) -> UploadReport:
    """Check an official yield file without writing it."""

    store = store or default_store()
    empty = pd.DataFrame(columns=["year", "district", "crop", "yield_ton_ha"])
    if not payload or not payload.strip():
        return UploadReport(empty, ["The file is empty."], [], [], 0, 0)
    try:
        raw = _canonical_headers(_read_upload_frame(payload))
    except ValueError as error:
        return UploadReport(empty, [str(error)], [], [], 0, 0)

    missing = [
        name
        for name in ("year", "district", "crop", "yield_ton_ha")
        if name not in raw.columns
    ]
    if missing:
        return UploadReport(
            empty,
            [
                "The CSV needs the columns year, district, crop, "
                f"and yield_ton_ha. Missing: {', '.join(missing)}."
            ],
            [],
            [],
            0,
            int(len(raw)),
        )

    frames = _load_frames(store)
    current = _max_year(frames)
    earliest = _min_year(frames)
    known = {
        _norm(name): str(name)
        for name in frames["boro_rice"]["district"].astype(str)
    }
    crops = _crop_lookup()
    crop_names = [CROP_CATALOG[crop_id]["label"] for crop_id in CROP_CATALOG]
    district_names = list(known.values())

    kept = []
    row_errors = []
    blank_rows = 0
    seen: dict[tuple[int, str, str], int] = {}
    duplicates = 0

    for index, row in raw.iterrows():
        line = int(index) + 2
        if pd.isna(row["yield_ton_ha"]) or str(row["yield_ton_ha"]).strip() == "":
            blank_rows += 1
            continue
        try:
            year = int(float(row["year"]))
        except (TypeError, ValueError):
            row_errors.append(f"Line {line}: year is not a number.")
            continue
        if year < earliest or year > current + MAX_AHEAD:
            row_errors.append(
                f"Line {line}: {year} is outside "
                f"{earliest}–{current + MAX_AHEAD}."
            )
            continue
        district_key = _norm(row["district"])
        district = known.get(district_key)
        if district is None:
            hint = difflib.get_close_matches(
                district_key,
                list(known.keys()),
                n=1,
                cutoff=0.82,
            )
            suffix = f" Did you mean {known[hint[0]]}?" if hint else ""
            row_errors.append(
                f"Line {line}: unknown district '{row['district']}'.{suffix}"
            )
            continue
        crop_id = crops.get(_norm(row["crop"]))
        if crop_id is None:
            hint = difflib.get_close_matches(
                _norm(row["crop"]),
                [_norm(name) for name in crop_names],
                n=1,
                cutoff=0.75,
            )
            suffix = ""
            if hint:
                for name in crop_names:
                    if _norm(name) == hint[0]:
                        suffix = f" Did you mean {name}?"
                        break
            row_errors.append(
                f"Line {line}: unknown crop '{row['crop']}'.{suffix}"
            )
            continue
        yield_text = str(row["yield_ton_ha"]).strip().replace(" ", "")
        if "," in yield_text and "." not in yield_text:
            row_errors.append(
                f"Line {line}: use a dot for decimals, not a comma."
            )
            continue
        try:
            yield_value = float(yield_text)
        except ValueError:
            row_errors.append(f"Line {line}: yield is not a number.")
            continue
        if not (0 < yield_value <= YIELD_CAP):
            row_errors.append(
                f"Line {line}: yield must be greater than 0 "
                f"and at most {YIELD_CAP:g} t/ha."
            )
            continue
        key = (year, district, crop_id)
        if key in seen:
            duplicates += 1
        seen[key] = line
        kept.append(
            {
                "year": year,
                "district": district,
                "crop": crop_id,
                "yield_ton_ha": round(yield_value, 2),
            }
        )

    warnings = []
    if duplicates:
        warnings.append(
            f"{duplicates} repeated district-crop-year rows. "
            "The last value in the file is the one that will be stored."
        )
    if len(row_errors) > 12:
        hidden = len(row_errors) - 12
        row_errors = row_errors[:12]
        row_errors.append(f"{hidden} more rows were skipped.")

    rows = pd.DataFrame(kept, columns=["year", "district", "crop", "yield_ton_ha"])
    if not rows.empty:
        rows = rows.drop_duplicates(["year", "district", "crop"], keep="last")
    file_errors = []
    if rows.empty:
        file_errors.append(
            "The file has no yield numbers. Fill in the harvests you have, "
            "or use Open the next season and leave the file for later."
        )
    return UploadReport(
        rows.reset_index(drop=True),
        file_errors,
        row_errors,
        warnings,
        blank_rows,
        int(len(raw)),
    )


def score_season(
    year: int,
    crop_id: str | None = None,
    store: Store | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Model estimate beside the stored harvest for one season."""

    store = store or default_store()
    year = int(year)
    crop_ids = [crop_id] if crop_id else list(CROP_CATALOG)
    tables = []
    warnings = []

    for current_id in crop_ids:
        spec = CROP_CATALOG[current_id]
        model_path = crop_model_dir(current_id) / "best_model.pkl"
        if not model_path.exists():
            warnings.append(f"{spec['label']} has no trained model.")
            continue
        raw = _read_crop(store, current_id)
        if year not in set(raw["year"].astype(int)):
            warnings.append(f"{spec['label']} has no {year} rows.")
            continue
        featured = add_forecast_features(raw)
        subset = featured[featured["year"] == year].copy()
        model = joblib.load(model_path)
        scaler = joblib.load(crop_model_dir(current_id) / "scaler.pkl")
        features = joblib.load(crop_model_dir(current_id) / "feature_names.pkl")
        batch = pd.concat(
            [
                prediction_frame(row, features)
                for _, row in subset.iterrows()
            ],
            ignore_index=True,
        )
        predicted = model.predict(scaler.transform(batch))
        prior = raw.loc[
            raw["year"] == year - 1,
            ["district", "yield_ton_ha"],
        ].copy()
        prior["district"] = prior["district"].astype(str)
        prior_lookup = prior.set_index("district")["yield_ton_ha"]
        table = pd.DataFrame(
            {
                "Year": year,
                "Crop": spec["label"],
                "crop": current_id,
                "Division": subset["division"].astype(str).to_numpy(),
                "District": subset["district"].astype(str).to_numpy(),
                "Recorded (t/ha)": subset["yield_ton_ha"].astype(float).to_numpy(),
                "Model (t/ha)": np.round(predicted.astype(float), 2),
                "Prior year (t/ha)": subset["district"]
                .astype(str)
                .map(prior_lookup)
                .astype(float)
                .to_numpy(),
            }
        )
        tables.append(table)

    if not tables:
        return pd.DataFrame(), warnings

    result = pd.concat(tables, ignore_index=True)
    result["Recorded (t/ha)"] = result["Recorded (t/ha)"].round(2)
    result["Prior year (t/ha)"] = result["Prior year (t/ha)"].round(2)
    result["Gap (t/ha)"] = (
        result["Model (t/ha)"] - result["Recorded (t/ha)"]
    ).round(2)
    span = result["Recorded (t/ha)"].clip(lower=0.5) * 0.18
    limit = span.where(span > 0.35, 0.35)
    result["Check"] = np.where(
        result["Gap (t/ha)"].abs() > limit,
        "Review",
        "In range",
    )
    result["_review"] = result["Check"].eq("Review")
    result = result.sort_values(
        ["_review", "Crop", "District"],
        ascending=[False, True, True],
    ).drop(columns=["_review", "crop"])
    ordered = [
        "Year",
        "Crop",
        "Division",
        "District",
        "Recorded (t/ha)",
        "Model (t/ha)",
        "Prior year (t/ha)",
        "Gap (t/ha)",
        "Check",
    ]
    return result[ordered].reset_index(drop=True), warnings
