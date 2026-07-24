#!/bin/bash
# Shell script to run the Streamlit app on Linux/Mac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🎙️  Speech Emotion Recognition - Starting Web App      ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[WARNING] Virtual environment not found."
    echo "[INFO] Using system Python..."
fi

# Check if Streamlit is installed
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Streamlit is not installed!"
    echo "[INFO] Please run: pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Run Streamlit app
echo "[INFO] Starting Streamlit server..."
echo "[INFO] App will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
