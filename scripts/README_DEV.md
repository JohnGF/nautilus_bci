# Nautilus BCI: Development & Testing Setup

This branch (`dev/mock-event-testing`) is designed for development and testing on a standard PC without requiring access to the physical g.Nautilus EEG hardware.

The primary goals of this setup are:
1. Provide a lightweight GUI that strictly uses mock EEG data.
2. Ensure the LSL marker system correctly formats and exports into BIDS `_events.tsv` format across all tasks.
3. Allow easy installation on any OS without proprietary hardware SDKs.

## Installation

We use [uv](https://github.com/astral-sh/uv) to manage Python dependencies rapidly.

1. **Clone the repository and checkout this branch:**
   ```bash
   git clone <repository_url>
   cd nautilus_bci
   git checkout dev/mock-event-testing
   ```

2. **Sync the basic dependencies:**
   This command installs all the required tools (PySide6, MNE, BIDS, pylsl, etc) *without* the hardware-specific dependencies (like `pygds`).
   ```bash
   cd scripts
   uv sync
   ```

*(Note: If you are on the actual data collection machine with the g.Nautilus plugged in, you would install the hardware dependencies by running `uv sync --extra hardware` instead).*

## How to Test the Event System

You can run an automated integration test that spins up the mock streamer, loops through all major task paradigms (Motor Imagery, Music Memory, Video Dataset), injects test markers, exports them via the BIDS recorder, and finally verifies that every single marker was successfully saved to the BIDS `_events.tsv` file.

```bash
cd scripts
uv run python test_all_tasks_events.py
```

If it succeeds, you will see a `[SUCCESS] All tasks successfully generated and recorded LSL event markers!` message in the terminal.

## How to Use the Development GUI

If you want to manually test the UI workflows, launch the development suite:

```bash
cd scripts
uv run python run_dev_suite.py
```

This launches a version of the main Control Panel where:
- The EEG Streamer is hardcoded to launch `mock_lsl_streamer.py`.
- You can manually test starting the BIDS recording and launching the task GUIs without needing actual hardware locks.

---

## How to Write a New BIDS-Compliant Task

To create a new experimental task paradigm that seamlessly integrates with the 3-GUI orchestrator, LSL, and BIDS recorder, follow these implementation steps:

### 1. Inherit from `BaseTaskApp`
Your main window class must extend the `BaseTaskApp` class. This automatically initializes the LSL marker stream outlet under the hood.

```python
from tasks.common.base_task import BaseTaskApp

class MyCustomTaskApp(BaseTaskApp):
    def __init__(self):
        # Initialize LSL marker stream name and unique source ID
        super().__init__(marker_name='MotorImageryMarkers', source_id='Custom_Task_Markers_2026')
        self.bids_root = "bids_dataset_multimodal" # Default fallback
```

### 2. Implement Participant Inputs
Do not nest the BIDS recording widget in your task config screen. Instead, include simple text fields for Subject and Session IDs:

```python
self.sub_input = QLineEdit("01")
self.ses_input = QLineEdit("01")
```

### 3. Send Phase Event Markers with Durations
Whenever your task transitions to a new phase (e.g. stimulus presentation, resting period), emit an event marker using `send_marker(event_name, duration)`. Always specify the duration of the phase in seconds so the BIDS recorder can log the duration column accurately:

```python
# Event name, duration in seconds
self.send_marker("Visual_Stimulus_Cue_Left", 3.0)

# Resting period
self.send_marker("Rest", 2.5)
```

### 4. Parse Launch Arguments in `main()`
To ensure the orchestrator (Task Selector) can pre-populate participant metadata in your task window, parse command-line arguments and set them on your app window:

```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="01")
    parser.add_argument("--ses", type=str, default="01")
    parser.add_argument("--bids-root", type=str, default="bids_dataset_multimodal")
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = MyCustomTaskApp()
    
    # Pre-populate configs
    if args.bids_root:
        window.bids_root = args.bids_root
    if args.sub:
        window.sub_input.setText(args.sub.replace('sub-', ''))
    if args.ses:
        window.ses_input.setText(args.ses.replace('ses-', ''))

    window.show()
    sys.exit(app.exec())
```

### 5. Add to the `tasks/` Folder
Save your script in the `scripts/tasks/` directory. Include a detailed module docstring at the very top of your file. 

The **BCI Task Selector** (`tasks/task_launcher.py`) dynamically discovers files in this directory and uses the first line of the docstring as the card title, and the rest as the card description.

---

### BCI Task Template / Boilerplate Code

Below is a complete, copy-pasteable boilerplate template for a new BIDS-compliant task script. You can save this as `tasks/my_new_task.py` and modify it:

```python
"""
My Custom Experimental Task
This is a detailed description of what the task does. It will show up on the Task Selector card.
"""

import sys
import os
import random
import time

# Ensure parent directory is in path for imports
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFormLayout, QGroupBox,
    QProgressBar, QStackedWidget
)
from PySide6.QtGui import QFont, QColor, QPalette

# 1. Import BaseTaskApp
from tasks.common.base_task import BaseTaskApp

class CustomTaskApp(BaseTaskApp):
    def __init__(self):
        # 2. Fulfill superclass init with LSL Marker Name and unique Source ID
        super().__init__(marker_name='MotorImageryMarkers', source_id='Custom_Task_Markers_2026')
        self.setWindowTitle("Custom Experimental Task")
        self.resize(800, 600)
        self.bids_root = "bids_dataset_multimodal"

        # Trial & Timing configurations
        self.t_fixation = 2.0
        self.t_stimulus = 3.0
        self.t_rest = 2.0

        self.trials = ["Condition_A", "Condition_B", "Condition_A", "Condition_B"]
        self.trial_idx = 0

        # Timing timers
        self.trial_timer = QTimer()
        self.trial_timer.setSingleShot(True)
        self.trial_timer.timeout.connect(self.advance_phase)

        self.init_ui()

    def init_ui(self):
        # Apply BCI-Suite consistent dark palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 18, 25))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        palette.setColor(QPalette.Base, QColor(25, 30, 42))
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.Button, QColor(35, 45, 65))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))
        self.setPalette(palette)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Build screens
        self.config_screen = self.create_config_screen()
        self.task_screen = self.create_task_screen()

        self.stacked_widget.addWidget(self.config_screen)
        self.stacked_widget.addWidget(self.task_screen)
        self.stacked_widget.setCurrentWidget(self.config_screen)

    def create_config_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)

        title = QLabel("My Custom BCI Task")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setStyleSheet("color: #4DEEEA;")
        layout.addWidget(title)

        # 3. Simple Participant Metadata inputs (No BIDS Widget)
        meta_group = QGroupBox("👤 Participant Metadata")
        meta_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 12px; }")
        meta_form = QFormLayout(meta_group)
        
        self.sub_input = QLineEdit("01")
        self.sub_input.setStyleSheet("background-color: #191E2A; color: white; padding: 4px;")
        self.ses_input = QLineEdit("01")
        self.ses_input.setStyleSheet("background-color: #191E2A; color: white; padding: 4px;")
        
        meta_form.addRow("Subject ID:", self.sub_input)
        meta_form.addRow("Session ID:", self.ses_input)
        layout.addWidget(meta_group)

        # Start button
        self.btn_start = QPushButton("🚀 Start Experiment Session")
        self.btn_start.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_start.setStyleSheet("background-color: #00ADB5; color: white; padding: 12px; border-radius: 6px;")
        self.btn_start.clicked.connect(self.start_experiment)
        layout.addWidget(self.btn_start)

        return widget

    def create_task_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.lbl_symbol = QLabel("+")
        self.lbl_symbol.setFont(QFont("Arial", 80, QFont.Bold))
        self.lbl_symbol.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_symbol)

        self.lbl_instruction = QLabel("PREPARE")
        self.lbl_instruction.setFont(QFont("Arial", 20, QFont.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_instruction)

        return widget

    def start_experiment(self):
        self.trial_idx = 0
        sub = self.sub_input.text().strip()
        ses = self.ses_input.text().strip()
        
        # 4. Emit experiment start marker
        self.send_marker(f"Experiment_Start_Sub_{sub}_Ses_{ses}_Paradigm_CustomTask")

        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.run_next_trial()

    def run_next_trial(self):
        if self.trial_idx >= len(self.trials):
            self.finish_experiment()
            return

        self.current_condition = self.trials[self.trial_idx]
        
        # 5. Emit Phase starts with actual durations
        self.send_marker("Fixation", self.t_fixation)
        self.lbl_symbol.setText("+")
        self.lbl_instruction.setText("Focus on center cross...")
        self.current_phase = "Fixation"
        self.trial_timer.start(int(self.t_fixation * 1000))

    def advance_phase(self):
        if self.current_phase == "Fixation":
            # 6. Stimulus Phase
            self.send_marker(f"Stimulus_{self.current_condition}", self.t_stimulus)
            self.lbl_symbol.setText("🎯")
            self.lbl_instruction.setText(f"PRESENTING: {self.current_condition}")
            self.current_phase = "Stimulus"
            self.trial_timer.start(int(self.t_stimulus * 1000))

        elif self.current_phase == "Stimulus":
            # 7. Rest Phase
            self.send_marker("Rest", self.t_rest)
            self.lbl_symbol.setText("•")
            self.lbl_instruction.setText("Relax & Rest")
            self.current_phase = "Rest"
            self.trial_timer.start(int(self.t_rest * 1000))

        elif self.current_phase == "Rest":
            self.trial_idx += 1
            self.run_next_trial()

    def finish_experiment(self):
        self.send_marker("Experiment_End")
        self.lbl_symbol.setText("🎉")
        self.lbl_instruction.setText("SESSION COMPLETE!")
        QTimer.singleShot(3000, self.reset_to_config)

    def reset_to_config(self):
        self.stacked_widget.setCurrentWidget(self.config_screen)

# 8. Fulfill Command Line argument parsing in main()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="01")
    parser.add_argument("--ses", type=str, default="01")
    parser.add_argument("--bids-root", type=str, default="bids_dataset_multimodal")
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = CustomTaskApp()

    # Pre-populate configurations received from orchestrator launcher
    if args.bids_root:
        window.bids_root = args.bids_root
    if args.sub:
        window.sub_input.setText(args.sub.replace('sub-', ''))
    if args.ses:
        window.ses_input.setText(args.ses.replace('ses-', ''))

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```


