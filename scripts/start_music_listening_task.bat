@echo off
echo ======================================================================
echo Starting BCI Full-Length Music Listening Paradigm...
echo Plays complete music tracks from start to finish with LSL markers
echo ======================================================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tasks/music_full_track_task.py
) else (
    uv run python tasks/music_full_track_task.py
)
pause
