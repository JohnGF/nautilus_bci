#!/usr/bin/env bash
# ==============================================================================
# run_training_gui.sh
# ==============================================================================
# Linux launcher for BCI Tower Defense Rhythm Training Studio GUI.
# Auto-detects the virtual environment and launches gui.py.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Candidate virtual environments in priority order
CAND_VENVS=(
    "${SCRIPT_DIR}/../../../tower-defense-bci/python/.venv/bin/python"
    "${SCRIPT_DIR}/../../tower-defense-bci/python/.venv/bin/python"
    "/home/guilhermecoto/Documentos/Lasige/tower-defense-bci/python/.venv/bin/python"
    "${SCRIPT_DIR}/.venv/bin/python"
    "${SCRIPT_DIR}/../.venv/bin/python"
)

PYTHON_EXE=""
for cand in "${CAND_VENVS[@]}"; do
    if [ -f "$cand" ] && [ -x "$cand" ]; then
        PYTHON_EXE="$cand"
        break
    fi
done

if [ -z "$PYTHON_EXE" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python)"
    else
        echo "[ERROR] Python not found in system or virtual environments." >&2
        exit 1
    fi
fi

echo "===================================================================="
echo "   BCI Tower Defense — Rhythm Training & Real-Time Studio (GUI)"
echo "===================================================================="
echo "[+] Using Python: ${PYTHON_EXE}"

# Check for tkinter availability
if ! "${PYTHON_EXE}" -c "import tkinter" >/dev/null 2>&1; then
    echo ""
    echo "[!] Notice: 'tkinter' or 'libtk8.6.so' is missing for ${PYTHON_EXE}."
    echo "    To launch the GUI on Linux, install Tk:"
    echo "      • Arch / CachyOS / Manjaro: sudo pacman -S tk"
    echo "      • Ubuntu / Debian:          sudo apt install python3-tk"
    echo "      • Fedora:                   sudo dnf install python3-tkinter"
    echo ""
    echo "    (You can also run headless training via train_and_run.py)"
    echo "===================================================================="
    echo ""
fi

echo "[+] Launching GUI..."

# Ensure PYTHONPATH includes training and script directory
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH}"

exec "${PYTHON_EXE}" "${SCRIPT_DIR}/gui.py" "$@"
