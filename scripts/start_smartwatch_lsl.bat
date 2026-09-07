@echo off
echo Starting Smartwatch PPG and IMU LSL Bridge...
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" bridges/smartwatch_lsl_bridge.py --mode udp --port 5005
) else (
    uv run python bridges/smartwatch_lsl_bridge.py --mode udp --port 5005
)
pause
