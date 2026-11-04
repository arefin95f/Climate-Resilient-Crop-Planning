"""MySQL access for the AgriSight-BD modeling tables.

This computer uses MySQL when .env or the environment points at a server.
A deployed site with no database settings uses the CSV and model files
shipped in the project, which are the same records loaded into MySQL.
"""

import json
import os
import re

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

from src.config import CROP_CATALOG, CROP_DATA_DIR, DATA_RAW, MODEL_DIR, PROJECT_ROOT


SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
DATASET_COLUMNS = [
    "year",
    "district",
    "division",
    "latitude",
    "longitude",
    "crop",
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
CLIMATE_COLUMNS = [
    "temp_max_mean",
    "temp_min_mean",
    "rainfall_total",
    "humidity_mean",
    "solar_radiation_mean",
]
SOIL_COLUMNS = {
    "soil_ph": "ph",
    "soil_soc": "soc",
    "soil_nitrogen": "nitrogen",
    "soil_cec": "cec",
    "soil_clay": "clay",
    "soil_sand": "sand",
}

_engine: Engine | None = None
_mysql_ready: bool | None = None


def _load_env_file() -> None:
    """Load project .env values that are not already set in the process."""

    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_streamlit_secrets() -> None:
    """Read MYSQL_* keys from Streamlit secrets when the app is deployed."""

    try:
        import streamlit as st

        secrets = st.secrets
        for key in (
            "MYSQL_HOST",
            "MYSQL_PORT",
            "MYSQL_USER",
            "MYSQL_PASSWORD",
            "MYSQL_DATABASE",
        ):
            if key in secrets:
                os.environ.setdefault(key, str(secrets[key]))
    except Exception:
        return


def mysql_configured() -> bool:
    """True when this process was given a MySQL host or password."""

    _load_env_file()
    _load_streamlit_secrets()
    return any(
        os.environ.get(key)
        for key in ("MYSQL_HOST", "MYSQL_PASSWORD", "MYSQL_DATABASE")
    )


def mysql_settings() -> dict:
    """Connection settings from the environment."""

    _load_env_file()
    database = os.environ.get("MYSQL_DATABASE", "agrisight_bd")
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError(f"Unsupported MySQL database name: {database}")
    return {
        "host": os.environ.get("MYSQL_HOST", "localhost"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", ""),
        "database": database,
    }


def _url(database: str | None) -> URL:
    settings = mysql_settings()
    return URL.create(
        drivername="mysql+pymysql",
        username=settings["user"],
        password=settings["password"],
        host=settings["host"],
        port=settings["port"],
        database=database,
    )


def get_engine() -> Engine:
    """Shared engine for the AgriSight database."""

    global _engine
    if _engine is None:
        _engine = create_engine(
            _url(mysql_settings()["database"]),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
    return _engine


def reset_engine() -> None:
    """Drop the cached engine after settings or schema changes."""

    global _engine, _mysql_ready
    _mysql_ready = None
    if _engine is not None:
        _engine.dispose()
        _engine = None


def use_mysql() -> bool:
    """Use MySQL on this machine, and the project files when it is not configured."""

    global _mysql_ready
    if _mysql_ready is not None:
        return _mysql_ready
    if not mysql_configured():
        _mysql_ready = False
        return False
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        _mysql_ready = True
    except Exception:
        _mysql_ready = False
    return _mysql_ready


def _schema_statements() -> list[str]:
    script = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = []
    for chunk in script.split(";"):
        lines = [
            line
            for line in chunk.splitlines()
            if not line.strip().startswith("--")
        ]
        statement = "\n".join(lines).strip()
        if statement:
            statements.append(statement)
    return statements


def apply_schema() -> None:
    """Create the database and replace its tables with sql/schema.sql."""

    settings = mysql_settings()
    database = settings["database"]
    admin = create_engine(_url(None), pool_pre_ping=True)
    try:
        with admin.begin() as connection:
            connection.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
    except Exception as error:
        raise ConnectionError(
            "Cannot reach MySQL at "
            f"{settings['host']}:{settings['port']} as {settings['user']}. "
            "Set MYSQL_PASSWORD in the environment or in a .env file."
        ) from error
    finally:
        admin.dispose()

    reset_engine()
    with get_engine().begin() as connection:
        for statement in _schema_statements():
            connection.execute(text(statement))


def _lookup_ids(values: list[str]) -> dict[str, int]:
    ordered = sorted(set(values))
    return {value: index for index, value in enumerate(ordered, start=1)}


def _tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a modeling frame into the normalized tables."""

    missing = [
        column
        for column in DATASET_COLUMNS
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Modeling data is missing columns: {missing}")

    data = data.copy()
    division_ids = _lookup_ids(data["division"].astype(str).tolist())
    district_ids = _lookup_ids(data["district"].astype(str).tolist())

    divisions = pd.DataFrame(
        {"id": list(division_ids.values()), "name": list(division_ids.keys())}
    )
    district_info = (
        data.groupby("district", as_index=False)[["division", "latitude", "longitude"]]
        .first()
    )
    districts = pd.DataFrame(
        {
            "id": district_info["district"].map(district_ids),
            "name": district_info["district"].astype(str),
            "division_id": district_info["division"].astype(str).map(division_ids),
            "latitude": pd.to_numeric(district_info["latitude"]),
            "longitude": pd.to_numeric(district_info["longitude"]),
        }
    )

    crop_rows = []
    for crop_id in data["crop"].astype(str).drop_duplicates():
        spec = CROP_CATALOG.get(crop_id, {})
        source = spec.get("yield_source", "sample")
        if source not in {"bbs", "sample"}:
            source = "sample"
        crop_rows.append(
            {
                "id": crop_id,
                "label": spec.get("label", crop_id),
                "crop_group": spec.get("group", "Other"),
                "season": spec.get("season", ""),
                "yield_source": source,
            }
        )
    crops = pd.DataFrame(crop_rows)

    soil_source = (
        data.groupby(["district", "year"], as_index=False)[list(SOIL_COLUMNS)]
        .first()
    )
    soil = soil_source.rename(columns=SOIL_COLUMNS)
    soil.insert(0, "district_id", soil["district"].map(district_ids))
    soil = soil.drop(columns=["district"])
    soil["year"] = soil["year"].astype(int)

    climate = (
        data.groupby(["district", "year"], as_index=False)[CLIMATE_COLUMNS]
        .first()
    )
    climate.insert(0, "district_id", climate["district"].map(district_ids))
    climate = climate.drop(columns=["district"])
    climate["year"] = climate["year"].astype(int)

    observations = data[
        [
            "district",
            "crop",
            "year",
            "yield_ton_ha",
            "rainfall_mar_jun",
            "temp_max_mar_jun",
            "ndvi_mean",
            "ndvi_peak",
            "ndvi_std",
        ]
    ].copy()
    observations["district_id"] = observations["district"].map(district_ids)
    observations["crop_id"] = observations["crop"].astype(str)
    observations["year"] = observations["year"].astype(int)
    observations = observations.drop(columns=["district", "crop"])
    duplicate_keys = observations.duplicated(
        ["district_id", "crop_id", "year"]
    )
    if duplicate_keys.any():
        raise ValueError("Crop datasets contain duplicate district-year-crop rows")

    return {
        "divisions": divisions,
        "districts": districts,
        "crops": crops,
        "soil": soil,
        "climate": climate,
        "observations": observations,
    }


def publish_datasets(frames: list[pd.DataFrame]) -> dict[str, int]:
    """Replace the MySQL tables with these crop modeling frames."""

    if not frames:
        raise ValueError("No crop datasets to publish")

    tables = _tables(pd.concat(frames, ignore_index=True))
    apply_schema()
    with get_engine().begin() as connection:
        for name in (
            "divisions",
            "districts",
            "crops",
            "soil",
            "climate",
            "observations",
        ):
            tables[name].to_sql(
                name,
                connection,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )
    return {name: len(frame) for name, frame in tables.items()}


def publish_trained_models() -> int:
    """Store model scores and feature lists. The .pkl files stay on disk."""

    rows = []
    for crop_id in CROP_CATALOG:
        path = MODEL_DIR / crop_id / "model_metadata.json"
        if not path.exists():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        year_start, year_end = metadata["year_range"]
        rows.append(
            {
                "crop_id": crop_id,
                "best_model": metadata["best_model"],
                "feature_names": json.dumps(metadata["feature_names"]),
                "results": json.dumps(metadata["results"]),
                "n_samples": int(metadata["n_samples"]),
                "year_start": int(year_start),
                "year_end": int(year_end),
                "mean_yield": float(metadata["mean_yield"]),
                "artifact_dir": f"data/processed/model_artifacts/{crop_id}",
            }
        )
    if not rows:
        return 0

    statement = text(
        """
        INSERT INTO trained_models (
            crop_id, best_model, feature_names, results,
            n_samples, year_start, year_end, mean_yield, artifact_dir
        ) VALUES (
            :crop_id, :best_model, CAST(:feature_names AS JSON), CAST(:results AS JSON),
            :n_samples, :year_start, :year_end, :mean_yield, :artifact_dir
        )
        ON DUPLICATE KEY UPDATE
            best_model = VALUES(best_model),
            feature_names = VALUES(feature_names),
            results = VALUES(results),
            n_samples = VALUES(n_samples),
            year_start = VALUES(year_start),
            year_end = VALUES(year_end),
            mean_yield = VALUES(mean_yield),
            artifact_dir = VALUES(artifact_dir)
        """
    )
    with get_engine().begin() as connection:
        connection.execute(statement, rows)
    return len(rows)


def save_model_metadata(metadata: dict) -> None:
    """Update one crop after training. Keeps the model file as the artifact."""

    crop_id = metadata["crop"]
    year_start, year_end = metadata["year_range"]
    statement = text(
        """
        INSERT INTO trained_models (
            crop_id, best_model, feature_names, results,
            n_samples, year_start, year_end, mean_yield, artifact_dir
        ) VALUES (
            :crop_id, :best_model, CAST(:feature_names AS JSON), CAST(:results AS JSON),
            :n_samples, :year_start, :year_end, :mean_yield, :artifact_dir
        )
        ON DUPLICATE KEY UPDATE
            best_model = VALUES(best_model),
            feature_names = VALUES(feature_names),
            results = VALUES(results),
            n_samples = VALUES(n_samples),
            year_start = VALUES(year_start),
            year_end = VALUES(year_end),
            mean_yield = VALUES(mean_yield),
            artifact_dir = VALUES(artifact_dir)
        """
    )
    with get_engine().begin() as connection:
        connection.execute(
            statement,
            {
                "crop_id": crop_id,
                "best_model": metadata["best_model"],
                "feature_names": json.dumps(metadata["feature_names"]),
                "results": json.dumps(metadata["results"]),
                "n_samples": int(metadata["n_samples"]),
                "year_start": int(year_start),
                "year_end": int(year_end),
                "mean_yield": float(metadata["mean_yield"]),
                "artifact_dir": f"data/processed/model_artifacts/{crop_id}",
            },
        )


def read_frame(query: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""

    with get_engine().connect() as connection:
        return pd.read_sql(text(query), connection, params=params or {})


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    text_columns = {"district", "division", "crop"}
    for column in frame.columns:
        if column not in text_columns:
            frame[column] = pd.to_numeric(frame[column])
    if "year" in frame.columns:
        frame["year"] = frame["year"].astype(int)
    return frame


def _mysql_crop_frame(crop_id: str) -> pd.DataFrame:
    columns = ", ".join(DATASET_COLUMNS)
    frame = read_frame(
        f"SELECT {columns} FROM crop_dataset WHERE crop = :crop "
        "ORDER BY year, district",
        {"crop": crop_id},
    )
    if frame.empty:
        raise LookupError(
            f"No MySQL rows for crop '{crop_id}'. "
            "Load the database with: python scripts/load_mysql.py"
        )
    return _numeric_frame(frame)


def _file_crop_frame(crop_id: str) -> pd.DataFrame:
    path = CROP_DATA_DIR / f"{crop_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Crop dataset not found: {path}")
    return _numeric_frame(pd.read_csv(path))


def load_crop_frame(crop_id: str) -> pd.DataFrame:
    """Modeling rows for one crop."""

    if use_mysql():
        return _mysql_crop_frame(crop_id)
    return _file_crop_frame(crop_id)


def load_districts() -> pd.DataFrame:
    """District name, division, and centroid."""

    if use_mysql():
        frame = read_frame(
            """
            SELECT
                d.name AS district,
                dv.name AS division,
                d.latitude,
                d.longitude
            FROM districts AS d
            INNER JOIN divisions AS dv ON dv.id = d.division_id
            ORDER BY d.name
            """
        )
        return _numeric_frame(frame)
    return _numeric_frame(pd.read_csv(DATA_RAW / "districts.csv"))


def load_catalog_summary() -> pd.DataFrame:
    """Row counts and mean yield for every crop."""

    if use_mysql():
        frame = read_frame(
            """
            SELECT
                crop,
                COUNT(*) AS records,
                COUNT(DISTINCT district) AS districts,
                MIN(year) AS start_year,
                MAX(year) AS end_year,
                AVG(yield_ton_ha) AS mean_yield
            FROM crop_dataset
            GROUP BY crop
            """
        )
        return _numeric_frame(frame)
    return _numeric_frame(pd.read_csv(CROP_DATA_DIR / "catalog.csv"))


def _json_value(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _mysql_model_metadata(crop_id: str) -> dict:
    frame = read_frame(
        """
        SELECT
            c.id AS crop,
            c.label,
            c.crop_group,
            c.season,
            c.yield_source,
            m.best_model,
            m.feature_names,
            m.results,
            m.n_samples,
            m.year_start,
            m.year_end,
            m.mean_yield
        FROM trained_models AS m
        INNER JOIN crops AS c ON c.id = m.crop_id
        WHERE m.crop_id = :crop
        """,
        {"crop": crop_id},
    )
    if frame.empty:
        raise LookupError(f"No MySQL model record for crop '{crop_id}'.")
    row = frame.iloc[0]
    return {
        "crop": row["crop"],
        "label": row["label"],
        "group": row["crop_group"],
        "season": row["season"],
        "yield_source": row["yield_source"],
        "best_model": row["best_model"],
        "feature_names": _json_value(row["feature_names"]),
        "results": _json_value(row["results"]),
        "n_samples": int(row["n_samples"]),
        "year_range": [int(row["year_start"]), int(row["year_end"])],
        "mean_yield": float(row["mean_yield"]),
    }


def _file_model_metadata(crop_id: str) -> dict:
    path = MODEL_DIR / crop_id / "model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_metadata(crop_id: str) -> dict:
    """Scores and feature list for one trained crop model."""

    if use_mysql():
        return _mysql_model_metadata(crop_id)
    return _file_model_metadata(crop_id)
