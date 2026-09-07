"""
Standalone Multimodal BIDS Recorder GUI
=======================================

Launches the reusable BIDSRecorderControlWidget in a dedicated standalone PySide6 window.
This provides the 3rd GUI for managing recording configurations and start/stop triggers.
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from PySide6 import QtWidgets, QtGui, QtCore
from recorders.bids_recorder_widget import BIDSRecorderControlWidget

class StandaloneBidsRecorderWindow(QtWidgets.QMainWindow):
    def __init__(self, default_task="leftright", default_bids_root="bids_dataset_multimodal"):
        super().__init__()
        self.setWindowTitle("Multimodal BIDS Recording Studio")
        self.resize(550, 220)

        # Style palette for dark theme consistent with dashboard
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(15, 18, 25))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(240, 240, 245))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 30, 42))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(240, 240, 245))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(35, 45, 65))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(240, 240, 245))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0F1219;
                color: #F0F0F5;
            }
        """)

        # Add the BIDS recorder widget
        self.recorder_widget = BIDSRecorderControlWidget(
            default_task=default_task, 
            default_bids_root=default_bids_root
        )

        # Main Layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.addWidget(self.recorder_widget)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone BIDS Recorder GUI")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--ses", type=str, default="01", help="Session ID")
    parser.add_argument("--task", type=str, default="leftright", help="Task Name")
    parser.add_argument("--root", type=str, default="bids_dataset_multimodal", help="BIDS root folder")
    args, unknown = parser.parse_known_args()

    app = QtWidgets.QApplication(sys.argv)
    window = StandaloneBidsRecorderWindow(default_task=args.task, default_bids_root=args.root)
    
    # Pre-populate custom parameters if passed
    window.recorder_widget.txt_sub.setText(args.sub)
    window.recorder_widget.txt_ses.setText(args.ses)
    
    window.show()
    sys.exit(app.exec())
