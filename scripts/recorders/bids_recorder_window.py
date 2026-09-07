"""
Standalone BIDS Recorder Window
=================================

A detachable QMainWindow that manages LSL stream discovery, recording,
and BIDS export. All task apps send markers via LSL independently —
this window is the single entry point for recording.

Usage:
    uv run python recorders/bids_recorder_window.py

Or launch from a task app:
    from recorders.bids_recorder_window import BIDSRecorderWindow
    window = BIDSRecorderWindow()
    window.show()
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QFont, QColor, QAction
from PySide6.QtWidgets import QMainWindow, QStatusBar

from recorders.bids_recorder_widget import BIDSRecorderControlWidget
from recorders.bids_recorder import BIDSRecorder


class BIDSRecorderWindow(QMainWindow):
    """Standalone window for BIDS recording. All tasks connect via LSL."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BIDS Recorder")
        self.setMinimumSize(700, 200)
        self.recorder = None
        self.is_recording = False

        self._init_ui()
        self._init_menu()
        self._init_status()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        self.recorder_widget = BIDSRecorderControlWidget(
            default_task="leftright", default_bids_root="bids_dataset"
        )
        layout.addWidget(self.recorder_widget)

    def _init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New Session", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._new_session)
        file_menu.addAction(new_act)

        export_act = QAction("&Export Now", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._export_now)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _init_status(self):
        self.statusBar().showMessage("Ready — no active recording")

    def _new_session(self):
        self.recorder_widget.txt_sub.clear()
        self.recorder_widget.txt_ses.clear()
        self.recorder_widget.txt_task.clear()
        self.recorder_widget.txt_outdir.clear()
        self.statusBar().showMessage("New session — configure and start recording")

    def _export_now(self):
        if self.recorder and self.recorder.is_recording:
            self.recorder_widget.toggle_recording()
        else:
            self.statusBar().showMessage("No active recording to export")

    def _show_about(self):
        QtWidgets.QMessageBox.about(
            self,
            "About BIDS Recorder",
            "<h3>BIDS Recorder</h3>"
            "<p>Standalone LSL stream recorder for BIDS datasets.</p>"
            "<p>All task apps send markers via LSL. "
            "This window discovers streams, records, and exports.</p>"
            "<p>Shortcuts: Ctrl+N (New), Ctrl+E (Export), Ctrl+Q (Quit)</p>",
        )

    def closeEvent(self, event):
        if self.recorder and self.recorder.is_recording:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Recording Active",
                "A recording is in progress. Stop and export before quitting?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.recorder_widget.toggle_recording()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Standalone BIDS Recorder Window")
    parser.add_argument("--bids-root", default="bids_dataset", help="Default BIDS root folder")
    parser.add_argument("--task", default="leftright", help="Default task name")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = BIDSRecorderWindow()
    window.recorder_widget.txt_outdir.setText(args.bids_root)
    window.recorder_widget.txt_task.setText(args.task)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()