@echo off
setlocal
cd /d "%~dp0"

echo [HAYATE] One-click setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_hayate.ps1" %*
if errorlevel 1 (
  echo.
  echo [HAYATE] Setup failed. See the message above.
  pause
  exit /b 1
)
echo.
pause
