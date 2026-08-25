@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\hayate.exe" (
  echo [HAYATE] Virtual environment is not ready.
  echo Run: uv sync --extra generation --extra webui
  pause
  exit /b 1
)

".venv\Scripts\hayate.exe" webui --host 127.0.0.1 --port 7860 --open-browser
if errorlevel 1 pause
