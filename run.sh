#!/bin/bash
echo "=========================================="
echo "Starting RFP Document Intelligence System"
echo "=========================================="

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo ""
echo "[1/2] Running Offline Validation Suite..."
echo "------------------------------------------"
echo "No external LLM or API calls are made."
pytest -q
status=$?
if [ "$status" -ne 0 ]; then
    echo "Offline validation failed; dashboard will not start."
    exit "$status"
fi

echo ""
echo "[2/2] Starting Observability Dashboard..."
echo "------------------------------------------"
streamlit run app.py
