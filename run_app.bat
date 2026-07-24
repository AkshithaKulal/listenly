@echo off
REM Batch script to run the Streamlit app on Windows

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║     🎙️  Speech Emotion Recognition - Starting Web App      ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found.
    echo [INFO] Using system Python...
)

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit is not installed!
    echo [INFO] Please run: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Run Streamlit app
echo [INFO] Starting Streamlit server...
echo [INFO] App will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
