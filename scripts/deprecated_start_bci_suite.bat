@echo off
set DATASET_FOLDER=bids_dataset

echo Starting BCI Motor Imagery & BIDS Studio...
echo Target BIDS Dataset Folder: %DATASET_FOLDER%

cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_bci_suite.py --bids-root "%DATASET_FOLDER%"
) else (
    uv run python run_bci_suite.py --bids-root "%DATASET_FOLDER%"
)
pause
