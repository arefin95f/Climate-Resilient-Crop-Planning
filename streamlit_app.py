"""Streamlit Cloud entry point — runs from repo root."""

import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data" / "processed" / "model_artifacts"
APP_FILE = ROOT / "src" / "app" / "streamlit_app.py"


# Build the model artifacts on first run (e.g. on Streamlit Cloud).
if not (MODEL_DIR / "best_model.pkl").exists():
    subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py")],
        check=True,
    )

sys.path.insert(0, str(ROOT))

# Run the real app file directly instead of importing main from it.
runpy.run_path(
    str(APP_FILE),
    run_name="__main__",
)