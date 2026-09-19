#!/bin/bash
echo "=========================================="
echo "Starting RFP Document Intelligence System"
echo "=========================================="

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo ""
echo "[1/2] Running Backend Extraction Pipeline..."
echo "------------------------------------------"
python main.py --bid ../Bid1 --bid ../Bid2

echo ""
echo "[2/2] Starting Frontend Dashboard..."
echo "------------------------------------------"
streamlit run app.py
