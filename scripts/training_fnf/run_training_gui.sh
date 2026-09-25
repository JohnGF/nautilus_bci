#!/bin/bash
# Launch FNF BCI Training Studio
cd "$(dirname "$0")"
export PYTHONPATH=../analysis:$PYTHONPATH
if [ -f "../.venv/bin/python" ]; then
    ../.venv/bin/python gui.py
else
    python gui.py
fi
