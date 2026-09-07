#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

echo "===================================================================="
echo "     Nautilus BCI: Multimodal BIDS Experiment 3-GUI Orchestrator"
echo "===================================================================="
echo ""
echo "Launching 3 GUI Windows (No terminal prompts):"
echo "  1. 🧠 Multimodal BCI Master Dashboard (LSL Signals & Waveforms)"
echo "  2. 📺 BCI Experimental Task Selector (Choose paradigm)"
echo "  3. 🔴 Standalone BIDS Recording Engine (GUI Control Center)"
echo "===================================================================="
echo ""

# Default BIDS Root folder
BIDS_ROOT="bids_dataset_multimodal"

# Detect Python environment
PYTHON_EXE="uv run python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
fi

# 1. Start the Multimodal Dashboard in the background
echo "[+] Launching GUI 1: Multimodal BCI Master Dashboard..."
$PYTHON_EXE visualizers/multimodal_bci_dashboard.py --bids-root "$BIDS_ROOT" &

# Give dashboard a moment to spin up
sleep 2

# 2. Launch the BCI experimental task selector in the background
echo "[+] Launching GUI 2: BCI Task Selector App..."
$PYTHON_EXE tasks/task_launcher.py &

# Give task launcher a moment to initialize
sleep 1

# 3. Launch the Standalone BIDS Recorder GUI in the background
echo "[+] Launching GUI 3: Standalone BIDS Recorder GUI..."
$PYTHON_EXE recorders/run_standalone_recorder_gui.py --root "$BIDS_ROOT" &

echo ""
echo "===================================================================="
echo "All 3 GUI Windows successfully launched!"
echo "Configure subjects/sessions and start recording directly from the GUIs."
echo "===================================================================="
sleep 5
