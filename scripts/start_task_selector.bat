@echo off
echo Starting BCI Task Selector App...
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tasks/task_launcher.py
) else (
    uv run python tasks/task_launcher.py
)
pause
