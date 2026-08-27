@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\hayate.exe" (
  echo [HAYATE] Virtual environment is not ready.
  echo Run: uv sync --extra generation --extra webui
  pause
  exit /b 1
)

rem OpenVPN tunnel detected on this host: 10.8.0.0/24
".venv\Scripts\hayate.exe" webui --host 0.0.0.0 --port 7860 --open-browser --allow-network --trusted-client-network 10.8.0.0/24
if errorlevel 1 pause
