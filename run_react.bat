@echo off
REM Start API + React UI for Auralis

echo.
echo  Listenly - Speech Emotion Recognition
echo  -------------------------------------
echo.

if exist "venv\Scripts\activate.bat" (
  call venv\Scripts\activate.bat
)

start "Listenly API" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && python -m uvicorn api:app --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "Listenly UI" cmd /k "cd /d %~dp0frontend && npm run dev"

echo API:  http://127.0.0.1:8000
echo UI:   http://localhost:5173
echo.
pause
