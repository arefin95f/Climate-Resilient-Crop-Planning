"""Train one yield model for every crop dataset.

XGBoost is the single model. On a forward time split it has the lowest
error of the models compared on the recorded Boro, Aman, and maize series.
"""

import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import (
    CROP_CATALOG,
    MODEL_DIR,
    N_SPLITS,
    PRIMARY_CROP,
    RANDOM_STATE,
    crop_model_dir,
)
from src.db import load_crop_frame
from src.models.features import add_forecast_features, candidate_feature_sets


def _xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=700,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=6,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def _impute(train_frame: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fill gaps from the training fold only, then return arrays."""

    medians = train_frame.median()
    train_values = train_frame.fillna(medians).to_numpy()
    other_values = other.fillna(medians).to_numpy()
    return train_values, other_values


def _evaluate(model, features: pd.DataFrame, target: np.ndarray) -> dict:
    """Forward-chaining RMSE and MAE on the original feature scale."""

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    rmse_scores = []
    mae_scores = []

    for train_idx, test_idx in splitter.split(features):
        train_values, test_values = _impute(
            features.iloc[train_idx],
            features.iloc[test_idx],
        )

        clone = model.__class__(**model.get_params())
        clone.fit(train_values, target[train_idx])
        predicted = clone.predict(test_values)
        rmse_scores.append(
            float(np.sqrt(mean_squared_error(target[test_idx], predicted)))
        )
        mae_scores.append(
            float(mean_absolute_error(target[test_idx], predicted))
        )

    return {
        "rmse_mean": round(float(np.mean(rmse_scores)), 4),
        "rmse_std": round(float(np.std(rmse_scores)), 4),
        "mae_mean": round(float(np.mean(mae_scores)), 4),
        "mae_std": round(float(np.std(mae_scores)), 4),
    }


def train_crop(crop_id: str) -> dict:
    """Fit the yield model for one crop and save its artifacts."""

    spec = CROP_CATALOG[crop_id]
    frame = load_crop_frame(crop_id)
    frame = add_forecast_features(frame)
    feature_names = candidate_feature_sets(frame)["XGBoost"]
    target = frame["yield_ton_ha"].to_numpy()
    features = frame[feature_names]
    metrics = _evaluate(_xgboost(), features, target)

    filled = features.fillna(features.median())
    # The app always calls scaler.transform. The trees were fit on the
    # original units, so the scaler must leave those units unchanged.
    scaler = StandardScaler(with_mean=False, with_std=False)
    scaler.fit(filled)
    model = _xgboost()
    model.fit(filled.to_numpy(), target)

    model_dir = crop_model_dir(crop_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "best_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(feature_names, model_dir / "feature_names.pkl")

    metadata = {
        "crop": crop_id,
        "label": spec["label"],
        "group": spec["group"],
        "season": spec["season"],
        "yield_source": spec["yield_source"],
        "best_model": "XGBoost",
        "feature_names": feature_names,
        "results": {"XGBoost": metrics},
        "n_samples": int(len(frame)),
        "year_range": [int(frame["year"].min()), int(frame["year"].max())],
        "mean_yield": round(float(target.mean()), 3),
    }
    with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    try:
        from src.db import save_model_metadata

        save_model_metadata(metadata)
    except Exception as error:
        print(f"  Model file saved. MySQL record was not updated: {error}")

    print(
        f"  {spec['label']:<16} XGBoost  "
        f"RMSE {metrics['rmse_mean']:.3f}  MAE {metrics['mae_mean']:.3f}"
    )
    return metadata


def train_all_crops() -> None:
    """Train every crop and keep the Boro artifacts at the legacy path."""

    print("Training yield models on a forward time split...")
    for crop_id in CROP_CATALOG:
        train_crop(crop_id)

    legacy_dir = crop_model_dir(PRIMARY_CROP)
    for name in (
        "best_model.pkl",
        "scaler.pkl",
        "feature_names.pkl",
        "model_metadata.json",
    ):
        shutil.copy2(legacy_dir / name, MODEL_DIR / name)

    print(f"\nModels saved under {MODEL_DIR}")


if __name__ == "__main__":
    train_all_crops()
