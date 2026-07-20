# AgriSight: A Geospatial ML Framework for Climate-Resilient Crop Planning in Bangladesh



**AgriSight-BD** predicts district-level Boro rice yield and recommends climate-resilient crop strategies using live data from NASA POWER, ISRIC SoilGrids, Google Earth Engine (MODIS NDVI), and BBS agricultural statistics.



## Features



| Phase | Deliverable |

|-------|-------------|

| **Phase 1** | Data pipeline — NASA POWER + ISRIC + GEE NDVI + BBS yield merge |

| **Phase 2** | EDA (correlation heatmaps, choropleth maps, time-series decomposition), ML comparison (Linear Regression, Random Forest, XGBoost) with TimeSeriesSplit CV, SHAP explainability |

| **Phase 3** | Streamlit web app — Prediction Playground, What-If scenarios, Crop Suitability Map |



## Quick Start



```bash

# 1. Create virtual environment

python -m venv .venv

.venv\Scripts\activate        # Windows

# source .venv/bin/activate   # Linux/Mac



# 2. Install dependencies

pip install -r requirements.txt



# 3. Authenticate Google Earth Engine (one-time)

python scripts/authenticate_gee.py

# Optional: set your Cloud project ID (required for newer GEE accounts)

# set GEE_PROJECT=your-gcp-project-id



# 4. Run full pipeline (live APIs)

python run_pipeline.py



# Re-fetch all API data (ignore caches)

python run_pipeline.py --refresh



# 5. Launch web app

streamlit run streamlit_app.py

```



## Project Structure



```

AgriSight-BD/

├── data/

│   ├── raw/                  # Districts, BBS yield, API caches

│   └── processed/            # merged_dataset.csv, model artifacts

├── reports/figures/          # EDA and SHAP plots

├── scripts/                  # Utility scripts

├── src/

│   ├── config.py             # Central configuration

│   ├── data/                 # Data pipeline modules

│   │   ├── bbs_loader.py     # BBS yield loading

│   │   ├── nasa_power.py     # NASA POWER API

│   │   ├── soilgrids.py      # ISRIC SoilGrids API

│   │   ├── ndvi.py           # MODIS NDVI via GEE

│   │   └── pipeline.py       # Merge & clean

│   ├── analysis/

│   │   ├── eda.py            # Exploratory analysis

│   │   └── shap_analysis.py  # SHAP explainability

│   ├── models/

│   │   └── train.py          # ML training & CV

│   └── app/

│       └── streamlit_app.py  # Web application UI

├── streamlit_app.py          # Streamlit Cloud entry point

├── run_pipeline.py           # End-to-end runner

└── requirements.txt

```



## Data Sources



| Source | Data | Module |

|--------|------|--------|

| [BBS Yearbook](https://bbs.gov.bd/) | Crop yield (Boro, Aman, Maize) | `bbs_loader.py` |

| [NASA POWER](https://power.larc.nasa.gov/) | Temperature, rainfall, humidity, solar | `nasa_power.py` |

| [ISRIC SoilGrids](https://soilgrids.org/) | pH, SOC, nitrogen, clay, sand | `soilgrids.py` |

| [Google Earth Engine](https://earthengine.google.com/) | MODIS NDVI vegetation health | `ndvi.py` |



## Models



Three models compared with **TimeSeriesSplit** cross-validation (respects temporal order):



- **Linear Regression** — baseline

- **Random Forest** — tabular workhorse

- **XGBoost** — typically best performer



Metrics: RMSE and MAE. Best model saved with SHAP feature importance analysis.



## Deploy to Streamlit Cloud



1. Push this repo to GitHub

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect repo, set main file: `streamlit_app.py`

4. Add GEE credentials via Streamlit secrets if needed



**Note:** Run `python run_pipeline.py` locally first and commit `data/processed/` artifacts, or let the app run the pipeline on first launch (requires network access and GEE auth).



## Optional Enhancements



```bash

# Download district boundary GeoJSON for polygon choropleths

python scripts/download_geojson.py

```



## License



MIT — for academic and research use.


