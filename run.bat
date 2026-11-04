@echo off
cd /d "%~dp0"
title AgriSight-BD
echo Starting AgriSight-BD at http://localhost:8501
python -m streamlit run streamlit_app.py --server.port 8501 --server.headless false
if errorlevel 1 (
  echo.
  echo AgriSight-BD did not start. Check that Python and Streamlit are installed.
  pause
)
