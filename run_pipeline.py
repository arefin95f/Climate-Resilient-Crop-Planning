#!/usr/bin/env python

"""

AgriSight-BD: End-to-end pipeline runner.



Usage:

    python run_pipeline.py              # Fetch live data and train models

    python run_pipeline.py --refresh    # Ignore API caches and re-fetch

    python run_pipeline.py --app        # Launch Streamlit app after pipeline

"""

import argparse

import sys

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))





def main():

    parser = argparse.ArgumentParser(description="AgriSight-BD Pipeline")

    parser.add_argument("--refresh", action="store_true", help="Ignore cached API data and re-fetch")

    parser.add_argument("--skip-train", action="store_true", help="Skip model training")

    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figures")

    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")

    parser.add_argument("--app", action="store_true", help="Launch Streamlit app")

    args = parser.parse_args()



    print("\n" + "=" * 60)

    print("  AgriSight-BD: Geospatial ML for Climate-Resilient Agriculture")

    print("=" * 60 + "\n")



    from src.data.pipeline import run_pipeline

    run_pipeline(refresh=args.refresh)



    if not args.skip_eda:

        from src.analysis.eda import run_eda

        run_eda()



    if not args.skip_train:

        from src.models.train import train_all_models

        train_all_models()



    if not args.skip_shap and not args.skip_train:

        from src.analysis.shap_analysis import run_shap_analysis

        run_shap_analysis()



    print("\n" + "=" * 60)

    print("  Pipeline complete!")

    print("  Launch app:  streamlit run streamlit_app.py")

    print("=" * 60 + "\n")



    if args.app:

        import subprocess

        subprocess.run([

            sys.executable, "-m", "streamlit", "run",

            str(PROJECT_ROOT / "streamlit_app.py"),

        ])





if __name__ == "__main__":

    main()


