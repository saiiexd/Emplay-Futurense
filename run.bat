@echo off
echo ==========================================
echo Starting RFP Document Intelligence System
echo ==========================================

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo [1/2] Running Offline Validation Suite...
echo ------------------------------------------
echo No external LLM or API calls are made.
pytest -q
set "status=%errorlevel%"
if not "%status%"=="0" (
    echo Offline validation failed; dashboard will not start.
    exit /b %status%
)

echo.
echo [2/2] Starting Observability Dashboard...
echo ------------------------------------------
streamlit run app.py
