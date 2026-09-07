"""
Reusable BIDS Recorder Control Widget for Task & Visualizer GUIs
==================================================================

Provides an integrated QGroupBox widget with:
  - Subject ID, Session ID, and Task Name inputs
  - Target BIDS Dataset Folder line edit + Browse directory button
  - 1-Click Multimodal BIDS Recording Start/Stop button
  - Live background statistics update (sample counts & LSL markers)

Usage:
  from recorders.bids_recorder_widget import BIDSRecorderControlWidget
  recorder_widget = BIDSRecorderControlWidget(default_task="video", default_bids_root="bids_dataset")
  layout.addWidget(recorder_widget)
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QFont, QColor

from recorders.multimodal_bids_recorder import MultimodalBIDSRecorder
from utils.bids_utils import get_formatted_next_session


class BIDSRecorderControlWidget(QtWidgets.QGroupBox):
    def __init__(self, default_task="multimodal", default_bids_root="bids_dataset", parent=None):
        super().__init__("🔴 BIDS Dataset Recording Engine", parent)
        self.default_task = default_task
        self.default_bids_root = default_bids_root
        self.recorder = None

        self.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #4DEEEA;
                border: 1px solid #2C354A;
                border-radius: 8px;
                margin-top: 6px;
                padding: 12px;
            }
            QLineEdit {
                background-color: #191E2A;
                color: #FFFFFF;
                border: 1px solid #2C354A;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QLabel {
                color: #F0F0F5;
                font-weight: bold;
            }
        """)

        self.init_ui()

        # Auto-update session based on existing sessions for subject/bids root
        self.txt_sub.textChanged.connect(self.auto_update_session)
        self.txt_outdir.textChanged.connect(self.auto_update_session)
        self.auto_update_session()

        # Update timer for live recording statistics
        self.stat_timer = QtCore.QTimer(self)
        self.stat_timer.timeout.connect(self.update_live_stats)
        self.stat_timer.start(1000)

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Form Row 1: Subject, Session, Task Name
        row1 = QtWidgets.QHBoxLayout()

        row1.addWidget(QtWidgets.QLabel("Sub:"))
        self.txt_sub = QtWidgets.QLineEdit("01")
        self.txt_sub.setFixedWidth(50)
        row1.addWidget(self.txt_sub)

        row1.addWidget(QtWidgets.QLabel("Ses:"))
        self.txt_ses = QtWidgets.QLineEdit("01")
        self.txt_ses.setFixedWidth(50)
        row1.addWidget(self.txt_ses)

        row1.addWidget(QtWidgets.QLabel("Task:"))
        self.txt_task = QtWidgets.QLineEdit(self.default_task)
        self.txt_task.setFixedWidth(100)
        row1.addWidget(self.txt_task)

        row1.addWidget(QtWidgets.QLabel("BIDS Folder:"))
        self.txt_outdir = QtWidgets.QLineEdit(self.default_bids_root)
        self.txt_outdir.setMinimumWidth(160)
        row1.addWidget(self.txt_outdir)

        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_browse.setStyleSheet("background-color: #2C354A; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold;")
        self.btn_browse.clicked.connect(self.browse_bids_folder)
        row1.addWidget(self.btn_browse)

        layout.addLayout(row1)

        # Control Row 2: Record Button & Live Status Label
        row2 = QtWidgets.QHBoxLayout()

        self.btn_record = QtWidgets.QPushButton("🔴 Start BIDS Recording")
        self.btn_record.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_record.setStyleSheet("background-color: #C0392B; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold;")
        self.btn_record.clicked.connect(self.toggle_recording)
        row2.addWidget(self.btn_record)

        self.lbl_status = QtWidgets.QLabel("STATUS: Ready (Offline)")
        self.lbl_status.setStyleSheet("color: #A0A5B5; font-weight: bold; padding-left: 10px;")
        row2.addWidget(self.lbl_status, stretch=1)

        layout.addLayout(row2)

    def auto_update_session(self):
        sub = self.txt_sub.text().strip()
        outdir = self.txt_outdir.text().strip()
        curr_ses = self.txt_ses.text().strip()
        next_ses = get_formatted_next_session(outdir, sub, curr_ses)
        self.txt_ses.blockSignals(True)
        self.txt_ses.setText(next_ses)
        self.txt_ses.blockSignals(False)

    def browse_bids_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Target BIDS Dataset Directory", self.txt_outdir.text().strip())
        if folder:
            self.txt_outdir.setText(folder)

    def toggle_recording(self):
        if self.recorder is None or not self.recorder.is_recording:
            sub = self.txt_sub.text().strip()
            ses = self.txt_ses.text().strip()
            task = self.txt_task.text().strip()
            outdir = self.txt_outdir.text().strip()

            if not sub or not ses or not task or not outdir:
                QtWidgets.QMessageBox.warning(self, "Missing Configuration", "Please specify Subject ID, Session ID, Task Name, and BIDS Target Folder.")
                return

            try:
                self.recorder = MultimodalBIDSRecorder(bids_root=outdir)
                connected = self.recorder.discover_and_connect_streams(timeout=3.0)
                if not connected:
                    raise RuntimeError("No active LSL streams discovered on local network.")

                self.recorder.start_recording()
                self.btn_record.setText("⏹ Stop & Export BIDS Dataset")
                self.btn_record.setStyleSheet("background-color: #D63031; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold;")
                self.lbl_status.setText("STATUS: 🔴 RECORDING LIVE...")
                self.lbl_status.setStyleSheet("color: #2ECC71; font-weight: bold;")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Recording Failed", f"Could not start BIDS recording:\n{e}")
                self.recorder = None
        else:
            sub = self.txt_sub.text().strip()
            ses = self.txt_ses.text().strip()
            task = self.txt_task.text().strip()
            outdir = self.txt_outdir.text().strip()

            try:
                self.recorder.stop_and_export_bids(subject_id=sub, session_id=ses, task_name=task)
                QtWidgets.QMessageBox.information(self, "Dataset Exported", f"Multimodal BIDS dataset saved successfully to:\n{outdir}")
                self.auto_update_session()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting BIDS dataset:\n{e}")

            self.recorder = None
            self.btn_record.setText("🔴 Start BIDS Recording")
            self.btn_record.setStyleSheet("background-color: #C0392B; color: white; padding: 8px 16px; border-radius: 5px; font-weight: bold;")
            self.lbl_status.setText("STATUS: Recording Stopped & Saved")
            self.lbl_status.setStyleSheet("color: #A0A5B5; font-weight: bold;")

    def update_live_stats(self):
        if self.recorder and self.recorder.is_recording:
            n_eeg = 0
            # Search dynamically through inlets to find EEG stream and its buffers
            for sname, info in self.recorder.inlets.items():
                if info['type'] == 'EEG':
                    n_eeg = len(self.recorder.data_buffers.get(sname, []))
                    break
            n_mrk = len(self.recorder.marker_events)
            self.lbl_status.setText(f"STATUS: 🔴 RECORDING LIVE | {n_eeg} EEG samples | {n_mrk} Events")

    def get_subject_id(self):
        return self.txt_sub.text().strip()

    def get_session_id(self):
        return self.txt_ses.text().strip()

    def get_bids_folder(self):
        return self.txt_outdir.text().strip()
