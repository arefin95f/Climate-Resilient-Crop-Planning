"""SHAP explainability analysis for the best model."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.analysis.eda import load_merged
from src.config import FEATURE_COLUMNS, MODEL_DIR, REPORTS_FIGURES
from src.models.train import prepare_features


def run_shap_analysis(max_samples: int = 500, save: bool = True) -> dict:
    """Generate SHAP summary plot for the best trained model."""
    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Train models first: python -m src.models.train")

    model = joblib.load(model_path)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    df = load_merged()
    X, y, _ = prepare_features(df)
    X_scaled = scaler.transform(X)

    if len(X_scaled) > max_samples:
        idx = np.random.default_rng(42).choice(len(X_scaled), max_samples, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    print("Computing SHAP values...")
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
    except NotImplementedError:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap_vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    shap.summary_plot(shap_vals, features=X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance — Boro Rice Yield Prediction", fontweight="bold")
    plt.tight_layout()

    if save:
        REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(REPORTS_FIGURES / "shap_summary.png", dpi=150, bbox_inches="tight")
        print(f"  Saved SHAP plot to {REPORTS_FIGURES / 'shap_summary.png'}")

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

    print("\nTop feature drivers (mean |SHAP|):")
    for name, val in importance[:5]:
        print(f"  {name}: {val:.4f}")

    return {"shap_values": shap_values, "importance": importance, "figure": fig}


if __name__ == "__main__":
    run_shap_analysis()
