#!/bin/bash
echo "=========================================="
echo "Starting RFP Document Intelligence System"
echo "=========================================="

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo ""
echo "[1/2] Running CLI Extraction Pipeline..."
echo "------------------------------------------"
echo "Extraction is executed by the CLI (main.py), not by the dashboard."
python main.py --bid ../Bid1 --bid ../Bid2

echo ""
echo "[2/2] Starting Observability Dashboard..."
echo "------------------------------------------"
streamlit run app.py
