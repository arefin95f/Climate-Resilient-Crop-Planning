# 🌾 AgriSight-BD  
**A Geospatial ML Framework for Climate-Resilient Crop Planning in Bangladesh**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agrisight-bd.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 🚀 Live Demo

**👉 [Try AgriSight-BD Now](https://agrisight-bd.streamlit.app)**

*Interactive Prediction Playground • What-If Scenarios • Crop Suitability Maps*

---

## 📖 Overview

**AgriSight-BD** predicts district-level **Boro rice yield** and recommends climate-resilient crop strategies by integrating real-world data from:

| Data Source | Type | Provider |
|-------------|------|----------|
| 🌦️ Climate | Temperature, rainfall, humidity, solar radiation | [NASA POWER](https://power.larc.nasa.gov/) |
| 🌍 Soil | pH, organic carbon, nitrogen, clay, sand | [ISRIC SoilGrids](https://soilgrids.org/) |
| 🛰️ Vegetation | MODIS NDVI (vegetation health index) | [Google Earth Engine](https://earthengine.google.com/) |
| 📊 Crop Yield | District-level Boro, Aman, Maize statistics | [Bangladesh Bureau of Statistics](https://bbs.gov.bd/) |

---

## 🎯 Key Features

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 1 – Data Engineering** | Automated pipeline: NASA POWER + ISRIC + GEE NDVI + BBS yield merge | ✅ Complete |
| **Phase 2 – Analysis & Modeling** | EDA (correlation heatmaps, choropleth maps, time-series decomposition), ML comparison (Linear Regression, Random Forest, **XGBoost**) with TimeSeriesSplit CV, **SHAP explainability** | ✅ Complete |
| **Phase 3 – Production** | Streamlit web app with Prediction Playground, What-If Scenarios, and Crop Suitability Map | ✅ Live |

---

## 🧠 Models & Methodology

Three regression models compared using **TimeSeriesSplit** cross-validation (respects temporal ordering to prevent data leakage):

| Model | Role | Performance |
|-------|------|-------------|
| **Linear Regression** | Baseline | — |
| **Random Forest** | Tabular data workhorse | Improved |
| **XGBoost** | State-of-the-art gradient boosting | 🏆 Best |

**Evaluation Metrics:** RMSE (Root Mean Square Error) and MAE (Mean Absolute Error)

**Explainability:** SHAP (SHapley Additive exPlanations) reveals *which features drive predictions* — critical insight for agricultural decision-makers.

---
