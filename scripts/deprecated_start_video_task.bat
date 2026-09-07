@echo off
echo Starting Video Dataset Task Presentation Suite...
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tasks/video_dataset_task.py
) else (
    uv run python tasks/video_dataset_task.py
)
pause
