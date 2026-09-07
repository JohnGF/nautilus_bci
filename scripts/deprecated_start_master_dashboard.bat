@echo off
set DATASET_FOLDER=bids_dataset_multimodal

echo Starting Multimodal BCI Master Dashboard Studio...
echo Target BIDS Dataset Folder: %DATASET_FOLDER%

cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" visualizers/multimodal_bci_dashboard.py --bids-root "%DATASET_FOLDER%"
) else (
    uv run python visualizers/multimodal_bci_dashboard.py --bids-root "%DATASET_FOLDER%"
)
pause
