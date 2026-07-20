"""Streamlit Cloud entry point — runs from repo root."""

import subprocess

import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "data" / "processed" / "model_artifacts"



if not (MODEL_DIR / "best_model.pkl").exists():

    subprocess.run([sys.executable, str(ROOT / "run_pipeline.py")], check=True)



sys.path.insert(0, str(ROOT))

from src.app.streamlit_app import main



main()


