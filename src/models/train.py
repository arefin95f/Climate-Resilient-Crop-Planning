"""Model training with TimeSeriesSplit cross-validation."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import FEATURE_COLUMNS, MODEL_DIR, N_SPLITS, RANDOM_STATE
from src.analysis.eda import load_merged


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Extract feature matrix and target, sorted by time."""
    df = df.sort_values(["year", "district"]).reset_index(drop=True)
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()
    y = df["yield_ton_ha"].copy()
    return X, y, available


def evaluate_model(model, X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> dict:
    """TimeSeriesSplit cross-validation with RMSE and MAE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores, mae_scores = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)

        rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        mae_scores.append(mean_absolute_error(y_test, preds))

    return {
        "rmse_mean": round(np.mean(rmse_scores), 4),
        "rmse_std": round(np.std(rmse_scores), 4),
        "mae_mean": round(np.mean(mae_scores), 4),
        "mae_std": round(np.std(mae_scores), 4),
    }


def train_all_models() -> dict:
    """Train baseline, Random Forest, and XGBoost; save best model."""
    print("Training models with TimeSeriesSplit CV...")
    df = load_merged()
    X, y, feature_names = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
    }

    results = {}
    best_name, best_rmse = None, float("inf")
    best_model = None

    for name, model in models.items():
        print(f"  Evaluating {name}...")
        metrics = evaluate_model(model, X_scaled, y.values)
        results[name] = metrics
        print(f"    RMSE: {metrics['rmse_mean']:.4f} (+/- {metrics['rmse_std']:.4f})")
        print(f"    MAE:  {metrics['mae_mean']:.4f} (+/- {metrics['mae_std']:.4f})")

        if metrics["rmse_mean"] < best_rmse:
            best_rmse = metrics["rmse_mean"]
            best_name = name
            best_model = model

    best_model.fit(X_scaled, y.values)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

    metadata = {
        "best_model": best_name,
        "feature_names": feature_names,
        "results": results,
        "n_samples": len(df),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {best_name} (RMSE={best_rmse:.4f})")
    print(f"Artifacts saved to {MODEL_DIR}")

    return {"model": best_model, "scaler": scaler, "features": feature_names, "results": results, "data": df}


if __name__ == "__main__":
    train_all_models()
