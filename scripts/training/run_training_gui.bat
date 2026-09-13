@echo off
title BCI Tower Defense - Rhythm Training Studio
echo Launching BCI Rhythm Training Studio GUI...
set SCRIPT_DIR=%~dp0
set VENV_PY=%SCRIPT_DIR%..\..\..\tower-defense-bci\python\.venv\Scripts\python.exe

if exist "%VENV_PY%" (
    "%VENV_PY%" "%SCRIPT_DIR%gui.py" %*
) else (
    python "%SCRIPT_DIR%gui.py" %*
)
