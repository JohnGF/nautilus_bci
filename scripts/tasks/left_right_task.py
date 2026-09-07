import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import sys
import os
import random
import time
import wave
import numpy as np

from PySide6.QtCore import Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFormLayout, QGroupBox,
    QProgressBar, QStackedWidget, QMessageBox, QCheckBox
)
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


# BIDSRecorderControlWidget removed (managed by standalone recorder GUI)


def generate_cue_wavs(sound_dir):
    """Generate high-quality PCM WAV sound cues for audio-guided BCI paradigms."""
    os.makedirs(sound_dir, exist_ok=True)
    sample_rate = 44100

    def make_wav(filename, freq_or_spec, duration=0.35, volume=0.85):
        filepath = os.path.abspath(os.path.join(sound_dir, filename))
        if os.path.exists(filepath):
            return filepath
        
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)

        if isinstance(freq_or_spec, (int, float)):
            waveform = volume * np.sin(2 * np.pi * freq_or_spec * t)
        elif isinstance(freq_or_spec, tuple):  # Chirp / Sweep (f_start, f_end)
            f_start, f_end = freq_or_spec
            freqs = np.linspace(f_start, f_end, num_samples)
            waveform = volume * np.sin(2 * np.pi * freqs * t)
        elif isinstance(freq_or_spec, list):  # Sequenced beeps [(freq, start_sec, dur_sec)]
            waveform = np.zeros(num_samples)
            for f_val, start_s, dur_s in freq_or_spec:
                i_start = int(sample_rate * start_s)
                i_end = int(sample_rate * (start_s + dur_s))
                if i_end > num_samples:
                    i_end = num_samples
                sub_t = np.linspace(0, dur_s, i_end - i_start, False)
                waveform[i_start:i_end] = volume * np.sin(2 * np.pi * f_val * sub_t)
        else:
            waveform = np.zeros(num_samples)

        # 15ms fade in/out envelope to prevent popping/clicking
        fade_len = int(sample_rate * 0.015)
        if fade_len > 0 and len(waveform) > 2 * fade_len:
            waveform[:fade_len] *= np.linspace(0, 1, fade_len)
            waveform[-fade_len:] *= np.linspace(1, 0, fade_len)

        pcm_data = (waveform * 32767).astype(np.int16)
        with wave.open(filepath, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(pcm_data.tobytes())
        return filepath

    files = {}
    files['top'] = make_wav("cue_top.wav", 1000.0, 0.4, volume=0.85)       # High tone (TOP)
    files['bottom'] = make_wav("cue_bottom.wav", 320.0, 0.4, volume=0.9)   # Low tone (BOTTOM)
    files['left'] = make_wav("cue_left.wav", 520.0, 0.4, volume=0.85)      # Mid-low tone (LEFT)
    files['right'] = make_wav("cue_right.wav", 780.0, 0.4, volume=0.85)    # Mid-high tone (RIGHT)
    
    files['high'] = files['top']
    files['low'] = files['bottom']
    files['go'] = make_wav("cue_go.wav", [(800.0, 0.0, 0.12), (1200.0, 0.15, 0.15)], 0.35, volume=0.85)  # Double beep (GO!)
    files['rest'] = make_wav("cue_rest.wav", (550.0, 220.0), 0.4, volume=0.75) # Descending tone (Rest)
    return files


from tasks.common.base_task import BaseTaskApp

class LeftRightTaskApp(BaseTaskApp):
    def __init__(self):
        super().__init__(marker_name='MotorImageryMarkers', source_id='MI_Task_Markers_2026')
        self.setWindowTitle("BCI All-In-One Motor Imagery Suite (Top, Bottom, Left, Right)")
        self.resize(1080, 820)

        # Sound Effects & Cue Generation Setup
        self.sound_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "sounds"))
        self.cue_files = generate_cue_wavs(self.sound_dir)

        # LSL Marker Outlet setup

        # Dual Audio Engine Setup
        self.music_player = QMediaPlayer()
        self.music_audio = QAudioOutput()
        self.music_player.setAudioOutput(self.music_audio)
        self.music_audio.setVolume(0.80)

        self.cue_player = QMediaPlayer()
        self.cue_audio = QAudioOutput()
        self.cue_player.setAudioOutput(self.cue_audio)
        self.cue_audio.setVolume(0.95)

        self.audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "brain_rhythm_audio.wav"))
        if os.path.exists(self.audio_path):
            self.music_player.setSource(QUrl.fromLocalFile(self.audio_path))
            self.music_player.setLoops(QMediaPlayer.Infinite)

        # Paradigm Parameters
        self.t_fixation = 2.0  # seconds
        self.t_cue = 1.0       # seconds
        self.t_task = 4.0      # seconds
        self.t_rest = 2.0      # seconds

        # State tracking
        self.current_block_idx = 0
        self.blocks = []
        self.trials = []
        self.trial_idx = 0

        self.trial_timer = QTimer()
        self.trial_timer.setSingleShot(True)
        self.trial_timer.timeout.connect(self.advance_trial_phase)

        self.tick_timer = QTimer()
        self.tick_timer.timeout.connect(self.update_progress)
        self.phase_start_time = 0.0
        self.current_phase_duration = 1.0

        self.init_ui()

    def play_cue_sound(self, sound_key):
        """Play overlay audio cue tone layered directly over background music."""
        if sound_key in self.cue_files and os.path.exists(self.cue_files[sound_key]):
            self.cue_player.stop()
            self.cue_player.setSource(QUrl.fromLocalFile(self.cue_files[sound_key]))
            self.cue_player.play()

    def init_ui(self):
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

        self.config_screen = self.create_config_screen()
        self.stacked_widget.addWidget(self.config_screen)

        self.task_screen = self.create_task_screen()
        self.stacked_widget.addWidget(self.task_screen)

        self.stacked_widget.setCurrentWidget(self.config_screen)

    def create_config_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 25, 40, 25)
        layout.setSpacing(12)

        title = QLabel("BCI 4-Direction All-In-One Motor Imagery Suite")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4DEEEA;")
        layout.addWidget(title)

        subtitle = QLabel("Top • Bottom • Left • Right All-In-One Paradigm with Dual Audio Tones")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #A0A5B5;")
        layout.addWidget(subtitle)

        # Participant Metadata Box (Decoupled from BIDS GUI)
        meta_group = QGroupBox("👤 Participant Metadata")
        meta_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 12px; }")
        meta_form = QFormLayout(meta_group)
        
        self.sub_input = QLineEdit("01")
        self.sub_input.setFixedWidth(60)
        self.sub_input.setStyleSheet("background-color: #191E2A; color: white; border: 1px solid #2C354A; border-radius: 4px; padding: 4px;")
        
        self.ses_input = QLineEdit("01")
        self.ses_input.setFixedWidth(60)
        self.ses_input.setStyleSheet("background-color: #191E2A; color: white; border: 1px solid #2C354A; border-radius: 4px; padding: 4px;")
        
        meta_form.addRow("Subject ID:", self.sub_input)
        meta_form.addRow("Session ID:", self.ses_input)
        layout.addWidget(meta_group)

        # Setup Form Box
        config_group = QGroupBox("Paradigm & Audio Configuration")
        config_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; color: #4DEEEA; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding-top: 12px; }")
        form = QFormLayout(config_group)
        form.setSpacing(10)

        self.movement_axis_combo = QComboBox()
        self.movement_axis_combo.addItems([
            "4-Direction All-in-One: Top, Bottom, Left, Right (4-Class Paradigm)",
            "2-Direction: Top vs. Bottom (Up / Down)",
            "2-Direction: Left vs. Right (Hand Movement)"
        ])
        self.movement_axis_combo.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")
        form.addRow("Movement Paradigm:", self.movement_axis_combo)

        self.presentation_mode_combo = QComboBox()
        self.presentation_mode_combo.addItems([
            "Audio-Only + Tones (Eyes Closed Mode - Distinct Direction Tones)",
            "Audio + Visual (Screen Arrows + Overlay Audio Tones)",
            "Visual-Only (Silent Screen Arrows)"
        ])
        self.presentation_mode_combo.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")
        form.addRow("Presentation & Eye Mode:", self.presentation_mode_combo)

        self.block_order_combo = QComboBox()
        self.block_order_combo.addItems([
            "1. No Music (Silent) → 2. Music",
            "1. Music → 2. No Music (Silent)",
            "Single Block: No Music Only",
            "Single Block: Music Only"
        ])
        self.block_order_combo.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")
        form.addRow("Background Music Order:", self.block_order_combo)

        self.reps_combo = QComboBox()
        self.reps_combo.addItems([
            "5 per direction (20 trials/block)",
            "10 per direction (40 trials/block - Recommended Wave 2 Pilot)",
            "15 per direction (60 trials/block)"
        ])
        self.reps_combo.setCurrentIndex(1)
        self.reps_combo.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")
        form.addRow("Trials per Block:", self.reps_combo)

        self.music_track_input = QLineEdit("Billie Jean - Michael Jackson (117 BPM)")
        self.music_track_input.setStyleSheet("background: #191E2A; color: white; padding: 5px; border-radius: 4px;")
        form.addRow("Background Music Label:", self.music_track_input)

        layout.addWidget(config_group)

        # Cue Audio Legend Box
        legend_group = QGroupBox("🔊 4-Direction Audio Tone Legend (Eyes Closed)")
        legend_group.setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; color: #FFEAA7; border: 1px solid #2C3545; border-radius: 8px; margin-top: 5px; padding: 8px; }")
        leg_layout = QHBoxLayout(legend_group)
        
        lbl1 = QLabel("⬆️ <b>Top:</b> 1000 Hz")
        lbl1.setStyleSheet("color: #00E676; font-size: 11px;")
        lbl2 = QLabel("⬇️ <b>Bottom:</b> 320 Hz")
        lbl2.setStyleSheet("color: #E040FB; font-size: 11px;")
        lbl3 = QLabel("⬅️ <b>Left:</b> 520 Hz")
        lbl3.setStyleSheet("color: #74B9FF; font-size: 11px;")
        lbl4 = QLabel("➡️ <b>Right:</b> 780 Hz")
        lbl4.setStyleSheet("color: #FF9F43; font-size: 11px;")

        leg_layout.addWidget(lbl1)
        leg_layout.addWidget(lbl2)
        leg_layout.addWidget(lbl3)
        leg_layout.addWidget(lbl4)
        layout.addWidget(legend_group)

        # Buttons Row
        btn_layout = QHBoxLayout()
        
        self.btn_practice = QPushButton("🎯 Quick Practice (4 Trials)")
        self.btn_practice.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_practice.setStyleSheet("background-color: #2A3B5C; color: white; padding: 10px 20px; border-radius: 6px;")
        self.btn_practice.clicked.connect(self.start_practice)
        btn_layout.addWidget(self.btn_practice)

        self.btn_start = QPushButton("🚀 Start 4-Direction All-In-One Session")
        self.btn_start.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_start.setStyleSheet("background-color: #00ADB5; color: white; padding: 10px 20px; border-radius: 6px;")
        self.btn_start.clicked.connect(self.start_experiment)
        btn_layout.addWidget(self.btn_start)

        layout.addLayout(btn_layout)
        return widget

    def create_task_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header Info
        header = QHBoxLayout()
        self.lbl_status = QLabel("Condition: No Music | Block 1/2")
        self.lbl_status.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #74B9FF;")
        header.addWidget(self.lbl_status)

        header.addStretch()

        self.lbl_trial_count = QLabel("Trial 0 / 40")
        self.lbl_trial_count.setFont(QFont("Arial", 14, QFont.Bold))
        self.lbl_trial_count.setStyleSheet("color: #A0A5B5;")
        header.addWidget(self.lbl_trial_count)
        layout.addLayout(header)

        layout.addStretch()

        # Center Visual Paradigm Area
        self.lbl_cue_symbol = QLabel("+")
        self.lbl_cue_symbol.setFont(QFont("Arial", 90, QFont.Bold))
        self.lbl_cue_symbol.setAlignment(Qt.AlignCenter)
        self.lbl_cue_symbol.setStyleSheet("color: #FFEAA7;")
        layout.addWidget(self.lbl_cue_symbol)

        self.lbl_instruction = QLabel("FOCUS ON THE FIXATION CROSS")
        self.lbl_instruction.setFont(QFont("Arial", 22, QFont.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setStyleSheet("color: #DFE6E9; margin-top: 20px;")
        layout.addWidget(self.lbl_instruction)

        # Progress Bar for Task Phase Timing
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background-color: #191E2A; border-radius: 6px; }
            QProgressBar::chunk { background-color: #00E676; border-radius: 6px; }
        """)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # Inter-block Continue Button
        self.btn_next_block = QPushButton("Press SPACE or Click to Begin Next Block")
        self.btn_next_block.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_next_block.setStyleSheet("background-color: #00E676; color: #000000; padding: 15px; border-radius: 8px;")
        self.btn_next_block.clicked.connect(self.start_current_block)
        self.btn_next_block.hide()
        layout.addWidget(self.btn_next_block)

        return widget

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and self.btn_next_block.isVisible():
            self.start_current_block()
        elif event.key() == Qt.Key_Escape:
            self.stop_experiment()

    def start_practice(self):
        self.setup_blocks(practice=True)
        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.start_current_block()

    def start_experiment(self):
        self.setup_blocks(practice=False)
        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.start_current_block()

    def setup_blocks(self, practice=False):
        reps_map = {0: 5, 1: 10, 2: 15}
        reps = 1 if practice else reps_map[self.reps_combo.currentIndex()]

        axis_idx = self.movement_axis_combo.currentIndex()
        if axis_idx == 0:
            self.directions = ["Top", "Bottom", "Left", "Right"]
            self.axis_code = "AllInOne4Class"
        elif axis_idx == 1:
            self.directions = ["Top", "Bottom"]
            self.axis_code = "TopBottom"
        else:
            self.directions = ["Left", "Right"]
            self.axis_code = "LeftRight"

        mode_idx = self.presentation_mode_combo.currentIndex()
        self.audio_cues_enabled = mode_idx in [0, 1]
        self.eyes_closed_mode = (mode_idx == 0)

        order_idx = self.block_order_combo.currentIndex()
        if practice:
            self.blocks = [{'name': 'Practice Mode (Music)', 'music': True}]
        elif order_idx == 0:
            self.blocks = [{'name': 'Block 1 (Music)', 'music': True}, {'name': 'Block 2 (No Music)', 'music': False}]
        elif order_idx == 1:
            self.blocks = [{'name': 'Block 1 (No Music)', 'music': False}, {'name': 'Block 2 (Music)', 'music': True}]
        elif order_idx == 2:
            self.blocks = [{'name': 'Block 1 (Music)', 'music': True}]
        else:
            self.blocks = [{'name': 'Block 1 (No Music)', 'music': False}]

        self.reps_per_direction = reps
        self.current_block_idx = 0

        sub = self.sub_input.text()
        ses = self.ses_input.text()
        mode = "AudioEyesClosed" if self.eyes_closed_mode else ("AudioVisual" if self.audio_cues_enabled else "VisualOnly")

        self.send_marker(f"Experiment_Start_Sub_{sub}_Ses_{ses}_Paradigm_{self.axis_code}_Mode_{mode}")

    def prepare_block_trials(self):
        trials = []
        for d in self.directions:
            trials.extend([d] * self.reps_per_direction)
        random.shuffle(trials)
        return trials

    def start_current_block(self):
        if self.current_block_idx >= len(self.blocks):
            self.finish_experiment()
            return

        self.btn_next_block.hide()
        block_info = self.blocks[self.current_block_idx]
        self.trials = self.prepare_block_trials()
        self.trial_idx = 0

        if block_info['music']:
            if os.path.exists(self.audio_path):
                self.music_player.play()
                position_ms = self.music_player.position()
                self.send_marker(f"Audio_Started_BillieJean_PosMS_{position_ms}")
            else:
                self.send_marker("Audio_Started_BillieJean_Synthetic")
        else:
            self.music_player.stop()
            self.send_marker("Audio_Stopped")

        block_marker = f"Block_Start_{'Music' if block_info['music'] else 'NoMusic'}_Idx_{self.current_block_idx+1}"
        self.send_marker(block_marker)

        mode_str = "Eyes Closed (Tones)" if self.eyes_closed_mode else "Visual Screen"
        self.lbl_status.setText(f"Condition: {block_info['name']} | Mode: {mode_str}")
        self.run_next_trial()

    def run_next_trial(self):
        if self.trial_idx >= len(self.trials):
            self.finish_block()
            return

        self.current_trial_direction = self.trials[self.trial_idx]
        self.lbl_trial_count.setText(f"Trial {self.trial_idx + 1} / {len(self.trials)}")
        self.send_marker(f"Trial_Start_{self.trial_idx + 1}_Dir_{self.current_trial_direction}")
        self.send_marker("Fixation", self.t_fixation)

        fixation_msg = "CLOSE EYES & PREPARE" if self.eyes_closed_mode else "CLEAR YOUR MIND"
        self.set_phase("Fixation", "+", fixation_msg, "#FFEAA7", self.t_fixation)

    def advance_trial_phase(self):
        dir_name = self.current_trial_direction
        symbol_map = {"Top": "↑", "Bottom": "↓", "Left": "←", "Right": "→"}
        color_map = {"Top": "#00E676", "Bottom": "#E040FB", "Left": "#74B9FF", "Right": "#FF9F43"}
        tone_map = {"Top": "1000 Hz", "Bottom": "320 Hz", "Left": "520 Hz", "Right": "780 Hz"}

        cue_symbol = symbol_map.get(dir_name, "+")
        cue_color = color_map.get(dir_name, "#00E676")

        if self.current_phase_name == "Fixation":
            # Phase 2: Cue (1.0s)
            if self.eyes_closed_mode:
                tone_hz = tone_map.get(dir_name, "Tone")
                instruction = f"LISTEN: {tone_hz} ➔ THINK {dir_name.upper()}"
            else:
                instruction = f"PREPARE: THINK {dir_name.upper()}"

            self.send_marker(f"Cue_{dir_name}", self.t_cue)
            
            if self.audio_cues_enabled:
                sound_key = dir_name.lower()
                self.play_cue_sound(sound_key)
                self.send_marker(f"AudioCue_Played_{dir_name.upper()}")

            self.set_phase("Cue", cue_symbol, instruction, cue_color, self.t_cue)

        elif self.current_phase_name == "Cue":
            # Phase 3: Task Imagery (4.0s)
            instruction = f"MENTALLY IMAGINE MOVING {dir_name.upper()}"
            self.send_marker(f"Task_{dir_name}", self.t_task)
            
            if self.audio_cues_enabled:
                self.play_cue_sound('go')

            self.set_phase("Task", cue_symbol, instruction, cue_color, self.t_task)

        elif self.current_phase_name == "Task":
            # Phase 4: Rest (2.0s)
            self.send_marker("Rest", self.t_rest)
            if self.audio_cues_enabled:
                self.play_cue_sound('rest')
                
            rest_msg = "RELAX & REST (Eyes Closed Ok)" if self.eyes_closed_mode else "RELAX & REST"
            self.set_phase("Rest", "•", rest_msg, "#A0A5B5", self.t_rest)

        elif self.current_phase_name == "Rest":
            self.send_marker(f"Trial_End_{self.trial_idx + 1}")
            self.trial_idx += 1
            self.run_next_trial()

    def set_phase(self, phase_name, symbol, instruction, color, duration):
        self.current_phase_name = phase_name
        self.lbl_cue_symbol.setText(symbol)
        self.lbl_cue_symbol.setStyleSheet(f"color: {color};")
        self.lbl_instruction.setText(instruction)
        self.lbl_instruction.setStyleSheet(f"color: {color}; margin-top: 20px;")

        self.phase_start_time = time.time()
        self.current_phase_duration = duration

        self.progress_bar.setValue(0)
        self.tick_timer.start(30)

        self.trial_timer.start(int(duration * 1000))

    def update_progress(self):
        elapsed = time.time() - self.phase_start_time
        progress = min(1.0, elapsed / self.current_phase_duration)
        self.progress_bar.setValue(int(progress * 100))

    def finish_block(self):
        self.tick_timer.stop()
        self.trial_timer.stop()
        self.music_player.stop()
        self.cue_player.stop()

        self.send_marker(f"Block_End_Idx_{self.current_block_idx+1}")
        self.current_block_idx += 1

        if self.current_block_idx < len(self.blocks):
            next_info = self.blocks[self.current_block_idx]
            self.lbl_cue_symbol.setText("⏸")
            self.lbl_cue_symbol.setStyleSheet("color: #74B9FF;")
            self.lbl_instruction.setText(f"BLOCK COMPLETE!\nNext: {next_info['name']}")
            self.btn_next_block.show()
        else:
            self.finish_experiment()

    def finish_experiment(self):
        self.tick_timer.stop()
        self.trial_timer.stop()
        self.music_player.stop()
        self.cue_player.stop()

        self.send_marker("Experiment_End")

        self.lbl_cue_symbol.setText("🎉")
        self.lbl_cue_symbol.setStyleSheet("color: #00E676;")
        self.lbl_instruction.setText("EXPERIMENT SESSION COMPLETE!\nThank you for participating.")
        
        QTimer.singleShot(3000, self.reset_to_config)

    def stop_experiment(self):
        self.tick_timer.stop()
        self.trial_timer.stop()
        self.music_player.stop()
        self.cue_player.stop()
        self.reset_to_config()

    def reset_to_config(self):
        self.stacked_widget.setCurrentWidget(self.config_screen)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Motor Imagery Task")
    parser.add_argument("--sub", type=str, default="01", help="Subject ID")
    parser.add_argument("--ses", type=str, default="01", help="Session ID")
    parser.add_argument("--bids-root", "--dataset-folder", type=str, default="bids_dataset", help="Target BIDS dataset output directory")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = LeftRightTaskApp()

    if args.sub:
        window.sub_input.setText(args.sub.replace('sub-', ''))
    if args.ses:
        window.ses_input.setText(args.ses.replace('ses-', ''))

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
