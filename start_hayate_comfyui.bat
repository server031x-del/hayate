@echo off
setlocal EnableExtensions
title HAYATE ComfyUI

rem This launcher keeps the existing PowerShell configuration in one place.
set "HAYATE_ROOT=%~dp0"
set "PS_SCRIPT=%HAYATE_ROOT%scripts\start_comfyui_hayate.ps1"
set "COMFY_ROOT=M:\Project\HAYATE-ComfyUI"

if not exist "%PS_SCRIPT%" (
    echo ERROR: Startup script was not found:
    echo        %PS_SCRIPT%
    pause
    exit /b 1
)

if not exist "%COMFY_ROOT%\main.py" (
    echo ERROR: ComfyUI was not found:
    echo        %COMFY_ROOT%
    pause
    exit /b 1
)

rem Do not create a second server if port 8189 is already listening.
netstat -ano -p tcp | findstr /R /C:":8189 .*LISTENING" >nul
if not errorlevel 1 (
    echo HAYATE ComfyUI is already running on port 8189.
    echo URL: http://127.0.0.1:8189
    pause
    exit /b 0
)

echo Starting HAYATE ComfyUI with the accelerated profile...
echo Listen: 0.0.0.0:8189
echo Close this window or press Ctrl+C to stop ComfyUI.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" ^
    -ComfyRoot "%COMFY_ROOT%" ^
    -Listen "0.0.0.0" ^
    -Port 8189 ^
    -EnableTurbo

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo ComfyUI exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
