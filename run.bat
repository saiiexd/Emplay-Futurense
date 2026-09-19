@echo off
echo ==========================================
echo Starting RFP Document Intelligence System
echo ==========================================

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo [1/2] Running Backend Extraction Pipeline...
echo ------------------------------------------
python main.py --bid ../Bid1 --bid ../Bid2

echo.
echo [2/2] Starting Frontend Dashboard...
echo ------------------------------------------
streamlit run app.py
