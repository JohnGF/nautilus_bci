@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ====================================================================
echo      Nautilus BCI: Multimodal BIDS Experiment 3-GUI Orchestrator
echo ====================================================================
echo.
echo Launching 3 GUI Windows (No terminal prompts):
echo   1. 🧠 Multimodal BCI Master Dashboard (LSL Signals & Waveforms)
echo   2. 📺 BCI Experimental Task Selector (Choose paradigm)
echo   3. 🔴 Standalone BIDS Recording Engine (GUI Control Center)
echo ====================================================================
echo.

:: Default BIDS Root folder
set BIDS_ROOT=bids_dataset_multimodal

:: Detect Python environment
set PYTHON_EXE=uv run python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=".venv\Scripts\python.exe"
)

:: 1. Start the Multimodal Dashboard in a background window
echo [+] Launching GUI 1: Multimodal BCI Master Dashboard...
start "Multimodal BCI Dashboard" %PYTHON_EXE% visualizers/multimodal_bci_dashboard.py --bids-root "%BIDS_ROOT%"

:: Give dashboard a moment to spin up
timeout /t 2 /nobreak > nul

:: 2. Launch the BCI experimental task selector in a separate background window
echo [+] Launching GUI 2: BCI Task Selector App...
start "BCI Task Selector" %PYTHON_EXE% tasks/task_launcher.py

:: Give task launcher a moment to initialize
timeout /t 1 /nobreak > nul

:: 3. Launch the Standalone BIDS Recorder GUI
echo [+] Launching GUI 3: Standalone BIDS Recorder GUI...
start "BIDS Recorder Studio" %PYTHON_EXE% recorders/run_standalone_recorder_gui.py --root "%BIDS_ROOT%"

echo.
echo ====================================================================
echo All 3 GUI Windows successfully launched!
echo Configure subjects/sessions and start recording directly from the GUIs.
echo ====================================================================
timeout /t 5 /nobreak > nul
