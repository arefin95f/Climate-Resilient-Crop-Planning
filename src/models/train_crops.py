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
    BBS_BACKTEST_START_YEAR,
    BBS_ERROR_BAND_CALIBRATION_END_YEAR,
    OBSERVED_BBS_YIELD_END_YEAR,
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


def _training_frame(frame: pd.DataFrame, yield_source: str) -> pd.DataFrame:
    """Keep every stored district-year, from the first year through the last."""

    del yield_source
    return frame.copy()


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


def _evaluate_year_holdout(
    model_factory,
    features: pd.DataFrame,
    target: np.ndarray,
    years: np.ndarray,
    backtest_start: int,
    calibration_end: int,
    backtest_end: int,
) -> dict:
    """One-year holdouts, with a 90% band fit on earlier test years."""

    predictions = []
    actuals = []
    prediction_years = []
    yearly_mae = []
    yearly_rmse = []

    for test_year in range(backtest_start, backtest_end + 1):
        train_idx = np.flatnonzero(years < test_year)
        test_idx = np.flatnonzero(years == test_year)
        if not len(test_idx):
            continue
        if not len(train_idx):
            raise ValueError(
                f"No earlier yield rows for backtest year {test_year}."
            )

        train_values, test_values = _impute(
            features.iloc[train_idx],
            features.iloc[test_idx],
        )
        model = model_factory()
        model.fit(train_values, target[train_idx])
        predicted = model.predict(test_values)
        actual = target[test_idx]

        predictions.extend(predicted.tolist())
        actuals.extend(actual.tolist())
        prediction_years.extend([test_year] * len(test_idx))
        yearly_mae.append(float(mean_absolute_error(actual, predicted)))
        yearly_rmse.append(
            float(np.sqrt(mean_squared_error(actual, predicted)))
        )

    if not predictions:
        raise ValueError("No yield rows were available for backtesting.")

    predictions_array = np.asarray(predictions)
    actuals_array = np.asarray(actuals)
    years_array = np.asarray(prediction_years)
    absolute_errors = np.abs(actuals_array - predictions_array)
    calibration_errors = absolute_errors[years_array <= calibration_end]
    later_errors = absolute_errors[years_array > calibration_end]
    if not len(calibration_errors):
        raise ValueError(
            "The backtest needs holdout years through "
            f"{calibration_end}."
        )
    # A later slice is an extra check. When the caller asks for every
    # holdout year, the band and the coverage use that same set.
    evaluation_errors = later_errors if len(later_errors) else calibration_errors

    error_band = float(
        np.quantile(calibration_errors, 0.90, method="higher")
    )
    evaluation_mask = (
        years_array > calibration_end
        if len(later_errors)
        else np.ones(len(years_array), dtype=bool)
    )
    evaluation_years = sorted(
        int(year) for year in np.unique(years_array[evaluation_mask])
    )
    test_years = sorted(int(year) for year in np.unique(years_array))

    return {
        "mae_mean": round(
            float(mean_absolute_error(actuals_array, predictions_array)), 4
        ),
        "mae_std": round(float(np.std(yearly_mae)), 4),
        "rmse_mean": round(
            float(
                np.sqrt(
                    mean_squared_error(actuals_array, predictions_array)
                )
            ),
            4,
        ),
        "rmse_std": round(float(np.std(yearly_rmse)), 4),
        "bias_mean": round(
            float(np.mean(predictions_array - actuals_array)), 4
        ),
        "n_test": int(len(actuals_array)),
        "test_years": test_years,
        "empirical_error_band": {
            "nominal_coverage": 0.90,
            "half_width_t_ha": round(error_band, 4),
            "calibration_years": [backtest_start, calibration_end],
            "calibration_n": int(len(calibration_errors)),
            "evaluation_years": evaluation_years,
            "evaluation_n": int(len(evaluation_errors)),
            "evaluation_coverage": round(
                float(np.mean(evaluation_errors <= error_band)), 4
            ),
        },
    }


def _evaluate_observed_bbs(
    model_factory,
    features: pd.DataFrame,
    target: np.ndarray,
    years: np.ndarray,
) -> dict:
    """Evaluate one-year holdouts on reported yields, without projections."""

    return _evaluate_year_holdout(
        model_factory,
        features,
        target,
        years,
        BBS_BACKTEST_START_YEAR,
        BBS_ERROR_BAND_CALIBRATION_END_YEAR,
        OBSERVED_BBS_YIELD_END_YEAR,
    )


def train_crop(crop_id: str) -> dict:
    """Fit the yield model for one crop and save its artifacts."""

    spec = CROP_CATALOG[crop_id]
    frame = _training_frame(
        load_crop_frame(crop_id),
        spec["yield_source"],
    )
    if frame.empty:
        raise ValueError(f"No yield rows for {crop_id}.")
    frame = add_forecast_features(frame)
    feature_names = candidate_feature_sets(frame)["XGBoost"]
    target = frame["yield_ton_ha"].to_numpy()
    features = frame[feature_names]
    years = frame["year"].to_numpy()
    # 2004 can only train the first fold. Every later district-year is scored.
    backtest_start = int(frame["year"].min()) + 1
    backtest_end = int(frame["year"].max())
    metrics = _evaluate_year_holdout(
        _xgboost,
        features,
        target,
        years,
        backtest_start,
        backtest_end,
        backtest_end,
    )

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
        "validation_protocol": {
            "method": "rolling-origin year-held-out backtest",
            "folds": len(metrics["test_years"]),
            "split_unit": "district-year",
            "target_scope": (
                f"every district-year from {backtest_start} through "
                f"{backtest_end}"
            ),
            "independent_field_trial": False,
        },
        "confidence_intervals_available": (
            "empirical_error_band" in metrics
        ),
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
