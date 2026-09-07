"""
BCI Dynamic Task Launcher & Selector Studio
===========================================
Dynamically discovers and presents all experimental task paradigms 
located in the `scripts/tasks/` directory with 1-click execution.
"""

import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import ast
import subprocess

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QFrame, QLineEdit
)
from PySide6.QtGui import QFont, QColor, QPalette

# BIDSRecorderControlWidget removed (managed by standalone recorder GUI)


def extract_task_metadata(filepath):
    """Extract docstring and module details from a task python file."""
    filename = os.path.basename(filepath)
    default_name = os.path.splitext(filename)[0].replace('_', ' ').title()
    desc = "Experimental BCI task module."

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        tree = ast.parse(code)
        doc = ast.get_docstring(tree)
        if doc:
            lines = [l.strip() for l in doc.strip().split('\n') if l.strip()]
            if lines:
                default_name = lines[0]
                if len(lines) > 1:
                    desc = " ".join(lines[1:])
    except Exception:
        pass

    # Custom icon mapping
    icon = "🎮"
    if "video" in filename.lower():
        icon = "📹"
    elif "music" in filename.lower() and "calib" not in filename.lower():
        icon = "🎵"
    elif "calib" in filename.lower():
        icon = "🎛️"
    elif "motor" in filename.lower() or "left" in filename.lower():
        icon = "🧠"

    return {
        'filename': filename,
        'filepath': filepath,
        'title': f"{icon} {default_name}",
        'description': desc
    }


def discover_tasks(tasks_dir):
    """Scan tasks/ directory for all runnable task scripts."""
    tasks = []
    if not os.path.exists(tasks_dir):
        return tasks

    skip_files = {'__init__.py', 'task_launcher.py', 'convert_music_tracks.py'}
    for f in sorted(os.listdir(tasks_dir)):
        if f.endswith('.py') and f not in skip_files:
            fpath = os.path.join(tasks_dir, f)
            meta = extract_task_metadata(fpath)
            tasks.append(meta)

    return tasks


class TaskLauncherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Experimental Task Selector Studio")
        self.resize(960, 720)

        base_dir = os.path.dirname(__file__)
        self.tasks_dir = os.path.abspath(base_dir)
        self.active_processes = []

        self.init_ui()

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(15, 18, 25))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        palette.setColor(QPalette.Base, QColor(25, 30, 42))
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.Button, QColor(35, 45, 65))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0F1219;
                color: #F0F0F5;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #4DEEEA;
                border: 1px solid #2C354A;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #191E2A;
                color: #FFFFFF;
                border: 1px solid #2C354A;
                border-radius: 4px;
                padding: 5px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 25, 30, 25)
        main_layout.setSpacing(15)

        # Header Title
        title = QLabel("🧠 BCI Task Selector & Stimulus Launcher")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        main_layout.addWidget(title)

        subtitle = QLabel("Select an experimental task to present stimuli & synchronize LSL markers")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        main_layout.addWidget(subtitle)

        # Metadata / Controls Box (Decoupled from BIDS GUI)
        self.controls_group = QGroupBox("👤 Session Parameters")
        self.controls_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C354A; border-radius: 8px; margin-top: 10px; padding: 15px; }")
        ctrl_layout = QHBoxLayout(self.controls_group)
        
        ctrl_layout.addWidget(QLabel("Sub:"))
        self.txt_sub = QLineEdit("01")
        self.txt_sub.setFixedWidth(50)
        ctrl_layout.addWidget(self.txt_sub)
        
        ctrl_layout.addWidget(QLabel("Ses:"))
        self.txt_ses = QLineEdit("01")
        self.txt_ses.setFixedWidth(50)
        ctrl_layout.addWidget(self.txt_ses)
        
        btn_refresh = QPushButton("🔄 Refresh Task Folder")
        btn_refresh.setStyleSheet("background-color: #2C354A; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold;")
        btn_refresh.clicked.connect(self.populate_task_cards)
        ctrl_layout.addWidget(btn_refresh)
        
        main_layout.addWidget(self.controls_group)

        # Scroll Area for Task Cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget()
        self.cards_layout = QVBoxLayout(self.scroll_content)
        self.cards_layout.setSpacing(12)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # Status Footer
        self.lbl_footer = QLabel("Task Folder: scripts/tasks/")
        self.lbl_footer.setStyleSheet("color: #74B9FF; font-size: 11px;")
        main_layout.addWidget(self.lbl_footer)

        self.populate_task_cards()

    def populate_task_cards(self):
        # Clear existing cards
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        tasks = discover_tasks(self.tasks_dir)
        if not tasks:
            no_task_lbl = QLabel(f"No task scripts found in: {self.tasks_dir}")
            no_task_lbl.setStyleSheet("color: #FF7675; font-size: 14px;")
            self.cards_layout.addWidget(no_task_lbl)
            return

        self.lbl_footer.setText(f"✅ Discovered {len(tasks)} task paradigm(s) in scripts/tasks/")

        card_colors = ['#00B894', '#6C5CE7', '#00ADB5', '#E17055', '#D63031', '#0984E3']

        for idx, task in enumerate(tasks):
            card = QFrame()
            color = card_colors[idx % len(card_colors)]
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: #191E2A;
                    border: 1px solid #2C354A;
                    border-left: 5px solid {color};
                    border-radius: 8px;
                    padding: 12px;
                }}
            """)
            card_layout = QHBoxLayout(card)

            # Left Text Details
            text_layout = QVBoxLayout()
            t_title = QLabel(task['title'])
            t_title.setFont(QFont("Arial", 14, QFont.Bold))
            t_title.setStyleSheet("color: #F0F0F5;")

            t_desc = QLabel(task['description'])
            t_desc.setFont(QFont("Arial", 10))
            t_desc.setWordWrap(True)
            t_desc.setStyleSheet("color: #A0A5B5;")

            t_file = QLabel(f"File: tasks/{task['filename']}")
            t_file.setFont(QFont("Consolas", 9))
            t_file.setStyleSheet("color: #74B9FF;")

            text_layout.addWidget(t_title)
            text_layout.addWidget(t_desc)
            text_layout.addWidget(t_file)

            card_layout.addLayout(text_layout, stretch=1)

            # Right Launch Button
            btn_launch = QPushButton("▶ LAUNCH TASK")
            btn_launch.setFont(QFont("Arial", 11, QFont.Bold))
            btn_launch.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    padding: 10px 18px;
                    border-radius: 6px;
                }}
                QPushButton:hover {{
                    background-color: #00ADB5;
                }}
            """)
            
            # Bind filepath to click handler
            filepath = task['filepath']
            btn_launch.clicked.connect(lambda checked=False, fp=filepath: self.launch_task_script(fp))
            card_layout.addWidget(btn_launch)

            self.cards_layout.addWidget(card)

        self.cards_layout.addStretch()

    def launch_task_script(self, filepath):
        print(f"[+] Launching task script: {filepath}")
        try:
            sub = self.txt_sub.text().strip()
            ses = self.txt_ses.text().strip()
            outdir = "bids_dataset_multimodal"
            proc = subprocess.Popen([sys.executable, filepath, "--sub", sub, "--ses", ses, "--bids-root", outdir])
            self.active_processes.append(proc)
        except Exception as e:
            print(f"[-] Error launching task {filepath}: {e}")

    def closeEvent(self, event):
        for p in self.active_processes:
            try:
                p.terminate()
            except Exception:
                pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = TaskLauncherApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
