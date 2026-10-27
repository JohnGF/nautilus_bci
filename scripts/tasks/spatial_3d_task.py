"""
3D Spatial Motor Imagery & 6-DoF Neurofeedback Experiment Suite
==============================================================
Provides a concrete 3D spatial object paradigm capable of moving in
full 6 Degrees of Freedom:
- Translations: Forward, Backward, Left, Right, Up, Down
- Rotations: Pitch Up, Pitch Down, Yaw Left, Yaw Right, Roll Left, Roll Right

Includes:
- Native hardware-accelerated OpenGL 3D viewport (QOpenGLWidget)
- Holographic 3D cue guidance (giving participants a concrete spatial target to imagine)
- Dynamic action execution (60 FPS vehicle physics, thruster trails, banking)
- Spatial directional audio cue synthesis
- Full BIDS-compliant LSL event marker pipeline (BaseTaskApp)
- Free Flight & Interactive Calibration mode
"""

import sys
import os
import math
import random
import time
import wave
import numpy as np

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from PySide6.QtCore import Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QFormLayout, QGroupBox,
    QProgressBar, QStackedWidget, QMessageBox, QCheckBox, QSlider, QFrame
)
from PySide6.QtGui import QFont, QColor, QPalette, QMatrix4x4, QVector3D
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram, QOpenGLBuffer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from tasks.common.base_task import BaseTaskApp
from utils.paths import ASSETS_SOUNDS


# =====================================================================
# SPATIAL AUDIO CUE SYNTHESIS
# =====================================================================
def generate_spatial_cue_wavs(sound_dir):
    """Generate stereo PCM WAV sound cues for 6-DoF directional guidance."""
    os.makedirs(sound_dir, exist_ok=True)
    sample_rate = 44100

    def make_stereo_wav(filename, left_func, right_func, duration=0.4, volume=0.85):
        filepath = os.path.abspath(os.path.join(sound_dir, filename))
        if os.path.exists(filepath):
            return filepath

        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)

        wav_l = left_func(t, num_samples) * volume
        wav_r = right_func(t, num_samples) * volume

        fade_len = int(sample_rate * 0.02)
        if fade_len > 0 and num_samples > 2 * fade_len:
            ramp = np.linspace(0, 1, fade_len)
            wav_l[:fade_len] *= ramp
            wav_l[-fade_len:] *= ramp[::-1]
            wav_r[:fade_len] *= ramp
            wav_r[-fade_len:] *= ramp[::-1]

        pcm_l = (np.clip(wav_l, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_r = (np.clip(wav_r, -1.0, 1.0) * 32767).astype(np.int16)

        interleaved = np.empty((num_samples * 2,), dtype=np.int16)
        interleaved[0::2] = pcm_l
        interleaved[1::2] = pcm_r

        with wave.open(filepath, 'wb') as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(interleaved.tobytes())
        return filepath

    files = {}

    # Forward: Rising acceleration sweep (center)
    files['forward'] = make_stereo_wav(
        "cue_spatial_forward.wav",
        lambda t, n: np.sin(2 * np.pi * np.geomspace(220, 660, n) * t),
        lambda t, n: np.sin(2 * np.pi * np.geomspace(220, 660, n) * t),
        duration=0.35
    )

    # Backward: Falling deceleration sweep (center)
    files['backward'] = make_stereo_wav(
        "cue_spatial_backward.wav",
        lambda t, n: np.sin(2 * np.pi * np.geomspace(660, 200, n) * t),
        lambda t, n: np.sin(2 * np.pi * np.geomspace(660, 200, n) * t),
        duration=0.35
    )

    # Left: Panned 100% Left
    files['left'] = make_stereo_wav(
        "cue_spatial_left.wav",
        lambda t, n: np.sin(2 * np.pi * 520.0 * t),
        lambda t, n: np.zeros(n),
        duration=0.35
    )

    # Right: Panned 100% Right
    files['right'] = make_stereo_wav(
        "cue_spatial_right.wav",
        lambda t, n: np.zeros(n),
        lambda t, n: np.sin(2 * np.pi * 780.0 * t),
        duration=0.35
    )

    # Up: Bright high-pitch chord
    files['up'] = make_stereo_wav(
        "cue_spatial_up.wav",
        lambda t, n: 0.6 * np.sin(2 * np.pi * 920.0 * t) + 0.4 * np.sin(2 * np.pi * 1380.0 * t),
        lambda t, n: 0.6 * np.sin(2 * np.pi * 920.0 * t) + 0.4 * np.sin(2 * np.pi * 1380.0 * t),
        duration=0.35
    )

    # Down: Deep bass resonance
    files['down'] = make_stereo_wav(
        "cue_spatial_down.wav",
        lambda t, n: np.sin(2 * np.pi * 180.0 * t) + 0.3 * np.sin(2 * np.pi * 90.0 * t),
        lambda t, n: np.sin(2 * np.pi * 180.0 * t) + 0.3 * np.sin(2 * np.pi * 90.0 * t),
        duration=0.38
    )

    # Pitch Up / Down
    files['pitch_up'] = make_stereo_wav(
        "cue_spatial_pitch_up.wav",
        lambda t, n: np.sin(2 * np.pi * np.linspace(350, 750, n) * t),
        lambda t, n: np.sin(2 * np.pi * np.linspace(350, 750, n) * t),
        duration=0.35
    )
    files['pitch_down'] = make_stereo_wav(
        "cue_spatial_pitch_down.wav",
        lambda t, n: np.sin(2 * np.pi * np.linspace(750, 320, n) * t),
        lambda t, n: np.sin(2 * np.pi * np.linspace(750, 320, n) * t),
        duration=0.35
    )

    # Yaw Left / Right
    files['yaw_left'] = make_stereo_wav(
        "cue_spatial_yaw_left.wav",
        lambda t, n: np.sin(2 * np.pi * 600.0 * t) * (1.0 - t / 0.35),
        lambda t, n: np.sin(2 * np.pi * 300.0 * t) * (t / 0.35) * 0.3,
        duration=0.35
    )
    files['yaw_right'] = make_stereo_wav(
        "cue_spatial_yaw_right.wav",
        lambda t, n: np.sin(2 * np.pi * 300.0 * t) * (t / 0.35) * 0.3,
        lambda t, n: np.sin(2 * np.pi * 600.0 * t) * (1.0 - t / 0.35),
        duration=0.35
    )

    # Roll Left / Right
    files['roll_left'] = make_stereo_wav(
        "cue_spatial_roll_left.wav",
        lambda t, n: np.sin(2 * np.pi * (500 + 150 * np.sin(2 * np.pi * 6 * t)) * t),
        lambda t, n: np.sin(2 * np.pi * (500 - 150 * np.sin(2 * np.pi * 6 * t)) * t) * 0.4,
        duration=0.38
    )
    files['roll_right'] = make_stereo_wav(
        "cue_spatial_roll_right.wav",
        lambda t, n: np.sin(2 * np.pi * (500 - 150 * np.sin(2 * np.pi * 6 * t)) * t) * 0.4,
        lambda t, n: np.sin(2 * np.pi * (500 + 150 * np.sin(2 * np.pi * 6 * t)) * t),
        duration=0.38
    )

    # General cues
    files['fixation'] = make_stereo_wav(
        "cue_spatial_fixation.wav",
        lambda t, n: np.sin(2 * np.pi * 500.0 * t),
        lambda t, n: np.sin(2 * np.pi * 500.0 * t),
        duration=0.15, volume=0.5
    )
    files['rest'] = make_stereo_wav(
        "cue_spatial_rest.wav",
        lambda t, n: 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 554.37 * t),
        lambda t, n: 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 659.25 * t),
        duration=0.4, volume=0.6
    )

    return files


# =====================================================================
# 3D GEOMETRY GENERATORS
# =====================================================================
def build_nautilus_mesh():
    """Build the Nautilus Cyber-Shuttle procedural 3D mesh (triangles + lines).
    Returns (tri_vbo_bytes, num_tri_verts, line_vbo_bytes, num_line_verts)
    Format: x, y, z, r, g, b, a
    """
    tri_verts = []
    line_verts = []

    # Colors
    c_hull = [0.15, 0.22, 0.35, 1.0]        # Dark blue fuselage
    c_hull_hi = [0.22, 0.35, 0.55, 1.0]     # Top fuselage highlight
    c_wing = [0.0, 0.65, 0.85, 0.95]        # Cyan wings
    c_cockpit = [0.2, 0.95, 0.9, 0.95]      # Bright cockpit glass
    c_fin = [0.9, 0.3, 0.5, 0.95]           # Neon magenta vertical fin
    c_thruster = [0.1, 0.1, 0.15, 1.0]      # Thruster bell
    c_glow = [1.0, 0.7, 0.2, 1.0]           # Engine nozzle glow
    c_wire = [0.3, 0.9, 1.0, 0.7]           # Structural wireframe

    # Fuselage Nose Cone
    nose = [0.0, 0.0, 2.2]
    f_top = [0.0, 0.4, 0.8]
    f_bot = [0.0, -0.3, 0.8]
    f_left = [-0.6, 0.0, 0.8]
    f_right = [0.6, 0.0, 0.8]

    # Nose faces (4 triangles)
    for p1, p2, col in [
        (f_top, f_left, c_hull_hi),
        (f_top, f_right, c_hull_hi),
        (f_bot, f_left, c_hull),
        (f_bot, f_right, c_hull),
    ]:
        tri_verts.extend(nose + col)
        tri_verts.extend(p1 + col)
        tri_verts.extend(p2 + col)

    # Mid Fuselage Body to Aft Tail
    tail_top = [0.0, 0.35, -1.8]
    tail_bot = [0.0, -0.25, -1.8]
    tail_left = [-0.45, 0.0, -1.8]
    tail_right = [0.45, 0.0, -1.8]

    # Fuselage Body Sides (quads as 2 tris)
    def add_quad(v1, v2, v3, v4, col):
        tri_verts.extend(v1 + col)
        tri_verts.extend(v2 + col)
        tri_verts.extend(v3 + col)
        tri_verts.extend(v1 + col)
        tri_verts.extend(v3 + col)
        tri_verts.extend(v4 + col)

    add_quad(f_top, f_right, tail_right, tail_top, c_hull_hi)
    add_quad(f_top, tail_top, tail_left, f_left, c_hull_hi)
    add_quad(f_bot, tail_bot, tail_right, f_right, c_hull)
    add_quad(f_bot, f_left, tail_left, tail_bot, c_hull)

    # Swept Delta Wings (Port and Starboard)
    wing_l_root_front = [-0.55, 0.0, 0.7]
    wing_l_root_back = [-0.45, 0.0, -1.5]
    wing_l_tip = [-2.4, 0.05, -1.1]

    wing_r_root_front = [0.55, 0.0, 0.7]
    wing_r_root_back = [0.45, 0.0, -1.5]
    wing_r_tip = [2.4, 0.05, -1.1]

    # Port wing (top & bottom)
    tri_verts.extend(wing_l_root_front + c_wing)
    tri_verts.extend(wing_l_tip + c_wing)
    tri_verts.extend(wing_l_root_back + c_wing)

    # Starboard wing
    tri_verts.extend(wing_r_root_front + c_wing)
    tri_verts.extend(wing_r_root_back + c_wing)
    tri_verts.extend(wing_r_tip + c_wing)

    # Wingtip Navigation Beacons (Red Port, Green Starboard)
    nav_red = [1.0, 0.1, 0.1, 1.0]
    nav_green = [0.1, 1.0, 0.3, 1.0]
    line_verts.extend(wing_l_tip + nav_red)
    line_verts.extend([-2.4, 0.25, -1.1] + nav_red)
    line_verts.extend(wing_r_tip + nav_green)
    line_verts.extend([2.4, 0.25, -1.1] + nav_green)

    # Cockpit Canopy
    c_front = [0.0, 0.25, 1.4]
    c_peak = [0.0, 0.65, 0.3]
    c_side_l = [-0.25, 0.35, 0.4]
    c_side_r = [0.25, 0.35, 0.4]
    c_back = [0.0, 0.45, -0.4]

    tri_verts.extend(c_front + c_cockpit)
    tri_verts.extend(c_side_l + c_cockpit)
    tri_verts.extend(c_peak + c_cockpit)

    tri_verts.extend(c_front + c_cockpit)
    tri_verts.extend(c_peak + c_cockpit)
    tri_verts.extend(c_side_r + c_cockpit)

    tri_verts.extend(c_peak + c_cockpit)
    tri_verts.extend(c_side_l + c_cockpit)
    tri_verts.extend(c_back + c_cockpit)

    tri_verts.extend(c_peak + c_cockpit)
    tri_verts.extend(c_back + c_cockpit)
    tri_verts.extend(c_side_r + c_cockpit)

    # Vertical Fin (Rudder)
    fin_base_front = [0.0, 0.35, -0.5]
    fin_tip = [0.0, 1.15, -1.7]
    fin_base_back = [0.0, 0.30, -1.75]

    tri_verts.extend(fin_base_front + c_fin)
    tri_verts.extend(fin_tip + c_fin)
    tri_verts.extend(fin_base_back + c_fin)

    # Twin Engine Nozzles (Aft Rings)
    for offset_x in [-0.22, 0.22]:
        z_nozzle = -1.85
        r_nz = 0.14
        num_seg = 12
        for i in range(num_seg):
            th1 = 2 * math.pi * i / num_seg
            th2 = 2 * math.pi * (i + 1) / num_seg
            p1 = [offset_x + r_nz * math.cos(th1), r_nz * math.sin(th1), z_nozzle]
            p2 = [offset_x + r_nz * math.cos(th2), r_nz * math.sin(th2), z_nozzle]
            center = [offset_x, 0.0, z_nozzle]
            tri_verts.extend(center + c_glow)
            tri_verts.extend(p1 + c_glow)
            tri_verts.extend(p2 + c_glow)

    # Edge wireframe highlights
    for e1, e2 in [
        (nose, f_top), (nose, f_bot), (nose, f_left), (nose, f_right),
        (f_top, tail_top), (f_left, tail_left), (f_right, tail_right), (f_bot, tail_bot),
        (wing_l_root_front, wing_l_tip), (wing_l_tip, wing_l_root_back),
        (wing_r_root_front, wing_r_tip), (wing_r_tip, wing_r_root_back),
        (fin_base_front, fin_tip), (fin_tip, fin_base_back)
    ]:
        line_verts.extend(e1 + c_wire)
        line_verts.extend(e2 + c_wire)

    tri_arr = np.array(tri_verts, dtype=np.float32)
    line_arr = np.array(line_verts, dtype=np.float32)

    return tri_arr, len(tri_arr) // 7, line_arr, len(line_arr) // 7


def build_cube_mesh():
    """Build a 6-DoF Orientation Cube with distinctly colored faces and wireframe edges."""
    tri_verts = []
    line_verts = []

    # 8 corners of unit cube (-1 to 1)
    c = 1.0
    v = [
        [-c, -c, -c], [c, -c, -c], [c, c, -c], [-c, c, -c],
        [-c, -c,  c], [c, -c,  c], [c, c,  c], [-c, c,  c]
    ]

    # Face definitions (Front, Back, Top, Bottom, Left, Right)
    faces = [
        # Front (+Z, Cyan)
        (4, 5, 6, 7, [0.0, 0.85, 0.95, 0.85]),
        # Back (-Z, Orange)
        (1, 0, 3, 2, [1.0, 0.55, 0.15, 0.85]),
        # Top (+Y, Green)
        (3, 2, 6, 7, [0.15, 0.9, 0.45, 0.85]),
        # Bottom (-Y, Magenta)
        (0, 1, 5, 4, [0.85, 0.2, 0.75, 0.85]),
        # Left (-X, Blue)
        (0, 4, 7, 3, [0.2, 0.45, 0.95, 0.85]),
        # Right (+X, Yellow)
        (5, 1, 2, 6, [0.95, 0.85, 0.15, 0.85]),
    ]

    for idx1, idx2, idx3, idx4, col in faces:
        p1, p2, p3, p4 = v[idx1], v[idx2], v[idx3], v[idx4]
        # Quad as two triangles
        tri_verts.extend(p1 + col)
        tri_verts.extend(p2 + col)
        tri_verts.extend(p3 + col)
        tri_verts.extend(p1 + col)
        tri_verts.extend(p3 + col)
        tri_verts.extend(p4 + col)

    # Wireframe edges (bright white)
    edge_col = [1.0, 1.0, 1.0, 0.9]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i1, i2 in edges:
        line_verts.extend(v[i1] + edge_col)
        line_verts.extend(v[i2] + edge_col)

    # Direction arrow on front (+Z) face pointing forward
    arrow_col = [1.0, 1.0, 1.0, 1.0]
    line_verts.extend([0.0, 0.6, c + 0.02] + arrow_col + [0.0, -0.4, c + 0.02] + arrow_col)
    line_verts.extend([0.0, 0.6, c + 0.02] + arrow_col + [-0.35, 0.15, c + 0.02] + arrow_col)
    line_verts.extend([0.0, 0.6, c + 0.02] + arrow_col + [0.35, 0.15, c + 0.02] + arrow_col)

    tri_arr = np.array(tri_verts, dtype=np.float32)
    line_arr = np.array(line_verts, dtype=np.float32)
    return tri_arr, len(tri_arr) // 7, line_arr, len(line_arr) // 7


def build_grid_mesh(size=24.0, step=1.5, y=-2.5):
    """Build a synthwave floor reference grid."""
    lines = []
    c_grid = [0.1, 0.35, 0.5, 0.4]
    c_axis_z = [0.0, 0.75, 0.95, 0.7]

    vals = np.arange(-size, size + 0.1, step)
    for x in vals:
        col = c_axis_z if abs(x) < 0.01 else c_grid
        lines.extend([float(x), y, -size] + col)
        lines.extend([float(x), y, size] + col)

    for z in vals:
        lines.extend([-size, y, float(z)] + c_grid)
        lines.extend([size, y, float(z)] + c_grid)

    arr = np.array(lines, dtype=np.float32)
    return arr, len(arr) // 7


def build_arrow_mesh(direction_str):
    """Build a 3D pulsing directional guide arrow for holographic cue guidance."""
    tri_verts = []
    line_verts = []
    col = [0.2, 1.0, 0.7, 0.8]  # Neon spring green
    col_edge = [1.0, 1.0, 1.0, 1.0]

    # Base arrow pointing in +Z direction
    shaft_len = 2.0
    shaft_r = 0.12
    head_len = 1.0
    head_r = 0.4

    # Build cone head
    num_seg = 12
    tip = [0.0, 0.0, shaft_len + head_len]
    base_center = [0.0, 0.0, shaft_len]

    for i in range(num_seg):
        th1 = 2 * math.pi * i / num_seg
        th2 = 2 * math.pi * (i + 1) / num_seg
        p1 = [head_r * math.cos(th1), head_r * math.sin(th1), shaft_len]
        p2 = [head_r * math.cos(th2), head_r * math.sin(th2), shaft_len]

        tri_verts.extend(tip + col)
        tri_verts.extend(p1 + col)
        tri_verts.extend(p2 + col)

        tri_verts.extend(base_center + col)
        tri_verts.extend(p2 + col)
        tri_verts.extend(p1 + col)

    # Shaft cylinder
    for i in range(num_seg):
        th1 = 2 * math.pi * i / num_seg
        th2 = 2 * math.pi * (i + 1) / num_seg
        b1 = [shaft_r * math.cos(th1), shaft_r * math.sin(th1), 0.0]
        b2 = [shaft_r * math.cos(th2), shaft_r * math.sin(th2), 0.0]
        t1 = [shaft_r * math.cos(th1), shaft_r * math.sin(th1), shaft_len]
        t2 = [shaft_r * math.cos(th2), shaft_r * math.sin(th2), shaft_len]

        tri_verts.extend(b1 + col)
        tri_verts.extend(b2 + col)
        tri_verts.extend(t2 + col)

        tri_verts.extend(b1 + col)
        tri_verts.extend(t2 + col)
        tri_verts.extend(t1 + col)

    arr = np.array(tri_verts, dtype=np.float32)
    return arr, len(arr) // 7


# =====================================================================
# SHADERS
# =====================================================================
VERTEX_SHADER_SRC = """
attribute highp vec3 a_pos;
attribute highp vec4 a_color;
uniform highp mat4 u_mvp;
varying highp vec4 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
varying highp vec4 v_color;
uniform highp float u_alpha_mult;
void main() {
    gl_FragColor = vec4(v_color.rgb, v_color.a * u_alpha_mult);
}
"""


# =====================================================================
# 3D OPENGL CANVAS WIDGET
# =====================================================================
class Spatial3DCanvas(QOpenGLWidget):
    """Hardware accelerated 60 FPS 3D Viewport with 6-DoF vehicle rendering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

        # Kinematic State (Position, Velocity, Rotation, Angular Velocity)
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # pitch, yaw, roll (deg)
        self.target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Control & Feedback Mode
        self.autonomous_control = True
        self.active_thrust_dir = None
        self.camera_dist = 8.5
        self.camera_pitch = 18.0
        self.camera_yaw = 0.0

        # Object selection: 'nautilus' or 'cube'
        self.object_type = 'nautilus'
        self.camera_mode = 'fixed'  # 'fixed' (perspective depth scaling), 'elastic', or 'chase'
        self.cam_eye_lag = None

        # Cue Hologram state
        self.show_cue_guide = False
        self.cue_direction = None
        self.cue_pulse_phase = 0.0

        # Thruster particle exhaust trails
        self.particles = []  # list of (x, y, z, vx, vy, vz, life, max_life, color)

        # Animation timer ~60 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_physics_and_render)
        self.last_time = time.time()
        self.timer.start(16)

        # Keyboard state for Free Flight
        self.keys_pressed = set()

    def initializeGL(self):
        f = self.context().functions()
        f.glEnable(0x0B71)  # GL_DEPTH_TEST
        f.glEnable(0x0BE2)  # GL_BLEND
        f.glBlendFunc(0x0302, 0x0303)  # GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        f.glClearColor(0.04, 0.06, 0.09, 1.0)

        # Shaders
        self.prog = QOpenGLShaderProgram(self)
        self.prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX_SHADER_SRC)
        self.prog.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAGMENT_SHADER_SRC)
        self.prog.link()

        # Build Geometry VBOs
        self.init_geometry()

    def init_geometry(self):
        # 1. Nautilus Mesh
        n_tri, n_tri_n, n_line, n_line_n = build_nautilus_mesh()
        self.vbo_nautilus_tri = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_nautilus_tri.create()
        self.vbo_nautilus_tri.bind()
        self.vbo_nautilus_tri.allocate(n_tri.tobytes(), n_tri.nbytes)
        self.nautilus_tri_count = n_tri_n

        self.vbo_nautilus_line = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_nautilus_line.create()
        self.vbo_nautilus_line.bind()
        self.vbo_nautilus_line.allocate(n_line.tobytes(), n_line.nbytes)
        self.nautilus_line_count = n_line_n

        # 2. Cube Mesh
        c_tri, c_tri_n, c_line, c_line_n = build_cube_mesh()
        self.vbo_cube_tri = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_cube_tri.create()
        self.vbo_cube_tri.bind()
        self.vbo_cube_tri.allocate(c_tri.tobytes(), c_tri.nbytes)
        self.cube_tri_count = c_tri_n

        self.vbo_cube_line = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_cube_line.create()
        self.vbo_cube_line.bind()
        self.vbo_cube_line.allocate(c_line.tobytes(), c_line.nbytes)
        self.cube_line_count = c_line_n

        # 3. Floor Grid
        g_data, g_n = build_grid_mesh(size=30.0, step=1.5, y=-2.5)
        self.vbo_grid = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_grid.create()
        self.vbo_grid.bind()
        self.vbo_grid.allocate(g_data.tobytes(), g_data.nbytes)
        self.grid_line_count = g_n

        # 4. Guide Arrow
        a_data, a_n = build_arrow_mesh("forward")
        self.vbo_arrow = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_arrow.create()
        self.vbo_arrow.bind()
        self.vbo_arrow.allocate(a_data.tobytes(), a_data.nbytes)
        self.arrow_tri_count = a_n

        # Dynamic Particles VBO
        self.vbo_particles = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_particles.create()

    def resizeGL(self, w, h):
        f = self.context().functions()
        f.glViewport(0, 0, w, h)

    def paintGL(self):
        f = self.context().functions()
        f.glClear(0x00004000 | 0x00000100)  # GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT

        if not hasattr(self, 'prog'):
            return

        self.prog.bind()
        self.prog.setUniformValue("u_alpha_mult", 1.0)

        # Projection Matrix
        w, h = max(1, self.width()), max(1, self.height())
        aspect = float(w) / float(h)
        proj = QMatrix4x4()
        proj.perspective(45.0, aspect, 0.1, 150.0)

        # Camera View Matrix (Perspective depth: decoupled fixed observer, elastic damped, or tethered chase)
        view = QMatrix4x4()
        rad_pitch = math.radians(self.camera_pitch)
        rad_yaw = math.radians(self.camera_yaw)

        mode = getattr(self, 'camera_mode', 'fixed')
        if mode == 'fixed':
            # Fixed Observer: Camera stays stationary relative to workspace center.
            # Forward (+Z) recedes into the distance -> natural perspective shrinkage.
            # Backward (-Z) advances toward the observer -> natural perspective expansion/growth.
            cam_x = self.camera_dist * math.sin(rad_yaw)
            cam_y = self.camera_dist * math.sin(rad_pitch)
            cam_z = -self.camera_dist * math.cos(rad_yaw) * math.cos(rad_pitch)
            target_x = self.pos[0] * 0.25
            target_y = 0.2 + self.pos[1] * 0.25
            target_z = 0.1 + self.pos[2] * 0.15
        elif mode == 'elastic':
            # Elastic Damped: Follows with lag so surge causes temporary shrink/grow
            desired_cam = np.array([
                self.pos[0] * 0.5,
                self.pos[1] * 0.5 + self.camera_dist * math.sin(rad_pitch),
                self.pos[2] * 0.35 - self.camera_dist * math.cos(rad_yaw) * math.cos(rad_pitch)
            ], dtype=np.float32)
            if self.cam_eye_lag is None:
                self.cam_eye_lag = desired_cam.copy()
            self.cam_eye_lag += (desired_cam - self.cam_eye_lag) * 0.15
            cam_x = float(self.cam_eye_lag[0])
            cam_y = float(self.cam_eye_lag[1])
            cam_z = float(self.cam_eye_lag[2])
            target_x = self.pos[0]
            target_y = self.pos[1] + 0.2
            target_z = self.pos[2] + 0.1
        else:
            # Tethered Chase: Camera rigidly follows vehicle position
            cam_x = self.pos[0] + self.camera_dist * math.sin(rad_yaw)
            cam_y = self.pos[1] + self.camera_dist * math.sin(rad_pitch)
            cam_z = self.pos[2] - self.camera_dist * math.cos(rad_yaw) * math.cos(rad_pitch)
            target_x = self.pos[0]
            target_y = self.pos[1] + 0.2
            target_z = self.pos[2] + 0.1

        view.lookAt(
            QVector3D(cam_x, cam_y, cam_z),
            QVector3D(target_x, target_y, target_z),
            QVector3D(0.0, 1.0, 0.0)
        )

        stride = 7 * 4

        # 1. Render Floor Reference Grid
        mvp_grid = proj * view
        self.prog.setUniformValue("u_mvp", mvp_grid)
        self.vbo_grid.bind()
        self.prog.enableAttributeArray("a_pos")
        self.prog.setAttributeBuffer("a_pos", 0x1406, 0, 3, stride)
        self.prog.enableAttributeArray("a_color")
        self.prog.setAttributeBuffer("a_color", 0x1406, 3 * 4, 4, stride)
        f.glDrawArrays(0x0001, 0, self.grid_line_count)  # GL_LINES
        self.vbo_grid.release()

        # 2. Render Holographic Cue Guide (if active)
        if self.show_cue_guide and self.cue_direction:
            self.render_cue_hologram(f, proj, view)

        # 3. Render Vehicle Model
        model = QMatrix4x4()
        model.translate(self.pos[0], self.pos[1], self.pos[2])
        model.rotate(self.rot[1], 0.0, 1.0, 0.0)  # Yaw
        model.rotate(self.rot[0], 1.0, 0.0, 0.0)  # Pitch
        model.rotate(self.rot[2], 0.0, 0.0, 1.0)  # Roll

        mvp_model = proj * view * model
        self.prog.setUniformValue("u_mvp", mvp_model)

        if self.object_type == 'nautilus':
            # Solid Triangles
            self.vbo_nautilus_tri.bind()
            self.prog.enableAttributeArray("a_pos")
            self.prog.setAttributeBuffer("a_pos", 0x1406, 0, 3, stride)
            self.prog.enableAttributeArray("a_color")
            self.prog.setAttributeBuffer("a_color", 0x1406, 3 * 4, 4, stride)
            f.glDrawArrays(0x0004, 0, self.nautilus_tri_count)  # GL_TRIANGLES
            self.vbo_nautilus_tri.release()

            # Wireframe Lines
            self.vbo_nautilus_line.bind()
            self.prog.enableAttributeArray("a_pos")
            self.prog.setAttributeBuffer("a_pos", 0x1406, 0, 3, stride)
            self.prog.enableAttributeArray("a_color")
            self.prog.setAttributeBuffer("a_color", 0x1406, 3 * 4, 4, stride)
            f.glDrawArrays(0x0001, 0, self.nautilus_line_count)  # GL_LINES
            self.vbo_nautilus_line.release()
        else:
            # 6-DoF Cube
            self.vbo_cube_tri.bind()
            self.prog.enableAttributeArray("a_pos")
            self.prog.setAttributeBuffer("a_pos", 0x1406, 0, 3, stride)
            self.prog.enableAttributeArray("a_color")
            self.prog.setAttributeBuffer("a_color", 0x1406, 3 * 4, 4, stride)
            f.glDrawArrays(0x0004, 0, self.cube_tri_count)
            self.vbo_cube_tri.release()

            self.vbo_cube_line.bind()
            self.prog.enableAttributeArray("a_pos")
            self.prog.setAttributeBuffer("a_pos", 0x1406, 0, 3, stride)
            self.prog.enableAttributeArray("a_color")
            self.prog.setAttributeBuffer("a_color", 0x1406, 3 * 4, 4, stride)
            f.glDrawArrays(0x0001, 0, self.cube_line_count)
            self.vbo_cube_line.release()

        # 4. Render Thruster Particles
        self.render_particles(f, proj, view)

        self.prog.release()

    def render_cue_hologram(self, f, proj, view):
        """Render a pulsing translucent guide arrow showing the target direction/rotation."""
        pulse = 0.5 + 0.4 * math.sin(self.cue_pulse_phase * 6.0)
        self.prog.setUniformValue("u_alpha_mult", float(pulse))

        guide = QMatrix4x4()
        guide.translate(self.pos[0], self.pos[1], self.pos[2])

        d = self.cue_direction.lower()
        if d == 'forward':
            pass  # default arrow points +Z
        elif d == 'backward':
            guide.rotate(180, 0, 1, 0)
        elif d == 'left':
            guide.rotate(90, 0, 1, 0)
        elif d == 'right':
            guide.rotate(-90, 0, 1, 0)
        elif d == 'up':
            guide.rotate(-90, 1, 0, 0)
        elif d == 'down':
            guide.rotate(90, 1, 0, 0)
        elif d == 'pitch_up':
            guide.rotate(-45, 1, 0, 0)
        elif d == 'pitch_down':
            guide.rotate(45, 1, 0, 0)
        elif d == 'yaw_left':
            guide.rotate(45, 0, 1, 0)
        elif d == 'yaw_right':
            guide.rotate(-45, 0, 1, 0)
        elif d == 'roll_left':
            guide.rotate(45, 0, 0, 1)
        elif d == 'roll_right':
            guide.rotate(-45, 0, 0, 1)

        mvp_guide = proj * view * guide
        self.prog.setUniformValue("u_mvp", mvp_guide)

        stride = 7 * 4
        self.vbo_arrow.bind()
        self.prog.enableAttributeArray("a_pos")
        self.prog.setAttributeBuffer("a_pos", 0x1406, 0, 3, stride)
        self.prog.enableAttributeArray("a_color")
        self.prog.setAttributeBuffer("a_color", 0x1406, 3 * 4, 4, stride)
        f.glDrawArrays(0x0004, 0, self.arrow_tri_count)
        self.vbo_arrow.release()

        self.prog.setUniformValue("u_alpha_mult", 1.0)

    def render_particles(self, f, proj, view):
        """Render active thruster exhaust particles."""
        if not self.particles:
            return

        p_verts = []
        for p in self.particles:
            alpha = (p['life'] / p['max_life'])
            c = p['color']
            # Small line segment in direction of velocity
            px, py, pz = p['pos']
            vx, vy, vz = p['vel']
            p_verts.extend([px, py, pz, c[0], c[1], c[2], alpha * c[3]])
            p_verts.extend([px - vx * 0.08, py - vy * 0.08, pz - vz * 0.08, c[0], c[1], c[2], 0.0])

        arr = np.array(p_verts, dtype=np.float32)
        mvp = proj * view
        self.prog.setUniformValue("u_mvp", mvp)

        self.vbo_particles.bind()
        self.vbo_particles.allocate(arr.tobytes(), arr.nbytes)
        stride = 7 * 4
        self.prog.enableAttributeArray("a_pos")
        self.prog.setAttributeBuffer("a_pos", 0x1406, 0, 3, stride)
        self.prog.enableAttributeArray("a_color")
        self.prog.setAttributeBuffer("a_color", 0x1406, 3 * 4, 4, stride)
        f.glDrawArrays(0x0001, 0, len(arr) // 7)  # GL_LINES
        self.vbo_particles.release()

    def emit_thruster_particles(self, dir_str):
        """Spawn engine exhaust particles based on active motion direction scaled by intensity."""
        u = getattr(self, 'active_intensity', 1.0)
        num = max(1, int(4 * u))
        d = dir_str.lower()
        aft_z = self.pos[2] - 1.85

        for _ in range(num):
            rx = (random.random() - 0.5) * 0.15
            ry = (random.random() - 0.5) * 0.15

            if d in ['forward', 'surge']:
                # Shoot out back
                pos = [self.pos[0] + (0.22 if random.random() > 0.5 else -0.22) + rx, self.pos[1] + ry, aft_z]
                vel = [rx * 2, ry * 2, (-5.0 - random.random() * 3.0) * u]
                color = [1.0, 0.65, 0.1, 0.9 * u]  # Orange thruster flame
            elif d == 'backward':
                # Reverse bow thrusters
                pos = [self.pos[0] + rx, self.pos[1] + ry, self.pos[2] + 1.8]
                vel = [rx * 2, ry * 2, (4.0 + random.random() * 2.0) * u]
                color = [0.2, 0.8, 1.0, 0.8 * u]  # Cyan brake jet
            elif d == 'left':
                # Lateral starboard RCS jet pushing vehicle left
                pos = [self.pos[0] + 0.6, self.pos[1] + ry, self.pos[2] + rx]
                vel = [(3.5 + random.random() * 2.0) * u, ry, rx]
                color = [0.2, 0.9, 0.9, 0.8 * u]
            elif d == 'right':
                # Lateral port RCS jet pushing vehicle right
                pos = [self.pos[0] - 0.6, self.pos[1] + ry, self.pos[2] + rx]
                vel = [(-3.5 - random.random() * 2.0) * u, ry, rx]
                color = [0.2, 0.9, 0.9, 0.8 * u]
            elif d in ['up', 'pitch_up']:
                # Ventral downward jets
                pos = [self.pos[0] + rx, self.pos[1] - 0.35, self.pos[2] + ry]
                vel = [rx, (-4.0 - random.random() * 2.0) * u, ry]
                color = [0.4, 1.0, 0.5, 0.8 * u]
            elif d in ['down', 'pitch_down']:
                # Dorsal upward jets
                pos = [self.pos[0] + rx, self.pos[1] + 0.4, self.pos[2] + ry]
                vel = [rx, (4.0 + random.random() * 2.0) * u, ry]
                color = [0.9, 0.3, 0.8, 0.8 * u]
            else:
                # Default ambient thruster idle
                pos = [self.pos[0] + (0.22 if random.random() > 0.5 else -0.22), self.pos[1], aft_z]
                vel = [rx, ry, -1.5 * u]
                color = [0.2, 0.7, 1.0, 0.4 * u]

            self.particles.append({
                'pos': pos,
                'vel': vel,
                'life': (0.35 + random.random() * 0.15) * (0.5 + 0.5 * u),
                'max_life': 0.5,
                'color': color
            })

    def update_physics_and_render(self):
        """Update continuous physics, spring-damper easing, and particle lifetimes."""
        now = time.time()
        dt = min(0.05, now - self.last_time)
        self.last_time = now

        # Update Cue Pulse
        self.cue_pulse_phase += dt

        # Handle Keyboard Inputs in Free Flight Mode
        if not self.autonomous_control:
            self.handle_manual_keys(dt)

        # Autonomous / Trial Easing
        if self.autonomous_control:
            # Smoothly lerp towards target position and target rotation
            spring_pos = 4.0
            spring_rot = 5.0
            self.pos += (self.target_pos - self.pos) * min(1.0, spring_pos * dt)
            self.rot += (self.target_rot - self.rot) * min(1.0, spring_rot * dt)

            # Subtle organic hover oscillation at rest
            if np.allclose(self.target_pos, 0.0, atol=0.05):
                t_hover = now * 1.8
                self.pos[1] = 0.08 * math.sin(t_hover)
                self.rot[2] = 1.5 * math.sin(t_hover * 0.8)

        # Emit thrusters if actively executing a directional action
        if self.active_thrust_dir:
            self.emit_thruster_particles(self.active_thrust_dir)

        # Update Particle Lifetimes
        alive = []
        for p in self.particles:
            p['life'] -= dt
            if p['life'] > 0:
                p['pos'][0] += p['vel'][0] * dt
                p['pos'][1] += p['vel'][1] * dt
                p['pos'][2] += p['vel'][2] * dt
                alive.append(p)
        self.particles = alive

        self.update()

    def handle_manual_keys(self, dt):
        """Manual 6-DoF flight controls with 256-step analog pressure ramping."""
        if len(self.keys_pressed) > 0:
            self.manual_ramp_step = min(255.0, getattr(self, 'manual_ramp_step', 0.0) + 600.0 * dt)
        else:
            self.manual_ramp_step = max(0.0, getattr(self, 'manual_ramp_step', 0.0) - 750.0 * dt)

        u = max(0.05, getattr(self, 'manual_ramp_step', 255.0) / 255.0)
        self.active_intensity = u
        thrust = 6.0 * u
        turn = 55.0 * u
        active_dir = None

        if Qt.Key_W in self.keys_pressed:
            self.pos[2] += thrust * dt
            active_dir = 'forward'
        if Qt.Key_S in self.keys_pressed:
            self.pos[2] -= thrust * dt
            active_dir = 'backward'
        if Qt.Key_A in self.keys_pressed:
            self.pos[0] -= thrust * dt
            self.rot[2] = min(30.0 * u, self.rot[2] + 45.0 * dt)
            active_dir = 'left'
        if Qt.Key_D in self.keys_pressed:
            self.pos[0] += thrust * dt
            self.rot[2] = max(-30.0 * u, self.rot[2] - 45.0 * dt)
            active_dir = 'right'
        if Qt.Key_Space in self.keys_pressed or Qt.Key_R in self.keys_pressed:
            self.pos[1] += thrust * dt
            active_dir = 'up'
        if Qt.Key_Shift in self.keys_pressed or Qt.Key_F in self.keys_pressed:
            self.pos[1] -= thrust * dt
            active_dir = 'down'

        # Rotations
        if Qt.Key_Up in self.keys_pressed:
            self.rot[0] += turn * dt
            active_dir = 'pitch_up'
        if Qt.Key_Down in self.keys_pressed:
            self.rot[0] -= turn * dt
            active_dir = 'pitch_down'
        if Qt.Key_Left in self.keys_pressed:
            self.rot[1] += turn * dt
            active_dir = 'yaw_left'
        if Qt.Key_Right in self.keys_pressed:
            self.rot[1] -= turn * dt
            active_dir = 'yaw_right'
        if Qt.Key_Q in self.keys_pressed:
            self.rot[2] += turn * dt
            active_dir = 'roll_left'
        if Qt.Key_E in self.keys_pressed:
            self.rot[2] -= turn * dt
            active_dir = 'roll_right'

        self.active_thrust_dir = active_dir

    def keyPressEvent(self, event):
        self.keys_pressed.add(event.key())
        if event.key() == Qt.Key_C:
            # Center reset
            self.pos[:] = 0.0
            self.rot[:] = 0.0
            self.vel[:] = 0.0
        elif event.key() == Qt.Key_M:
            # Toggle between Nautilus Shuttle and 6-DoF Cube
            self.object_type = 'cube' if self.object_type == 'nautilus' else 'nautilus'
            self.update()
        elif event.key() == Qt.Key_V:
            # Cycle Camera Mode: Fixed -> Elastic -> Chase
            modes = ['fixed', 'elastic', 'chase']
            idx = modes.index(self.camera_mode) if self.camera_mode in modes else 0
            self.camera_mode = modes[(idx + 1) % len(modes)]
            self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        self.keys_pressed.discard(event.key())
        super().keyReleaseEvent(event)

    def execute_action(self, direction_str, step=255):
        """Set autonomous target kinematic offsets scaled by 256-step intensity (uint8 0..255)."""
        d = direction_str.lower()
        self.active_thrust_dir = d
        u = max(0.02, min(1.0, float(step) / 255.0))
        self.active_intensity = u

        # Reset offsets from center
        target_p = np.zeros(3, dtype=np.float32)
        target_r = np.zeros(3, dtype=np.float32)

        dist = 3.5 * u
        deg = 35.0 * u

        if d == 'forward':
            # Surge forward: moves deeper into horizon -> object recedes and shrinks
            target_p[2] = 4.6 * u
            target_r[0] = -12.0 * u  # Slight nose pitch forward
        elif d == 'backward':
            # Surge backward: moves toward observer -> object approaches and enlarges
            target_p[2] = -3.8 * u
            target_r[0] = 12.0 * u
        elif d == 'left':
            target_p[0] = -dist
            target_r[2] = deg  # Bank left
        elif d == 'right':
            target_p[0] = dist
            target_r[2] = -deg  # Bank right
        elif d == 'up':
            target_p[1] = dist * 0.8
            target_r[0] = 18.0 * u
        elif d == 'down':
            target_p[1] = -dist * 0.8
            target_r[0] = -18.0 * u
        elif d == 'pitch_up':
            target_r[0] = deg * 1.2
            target_p[1] = 1.0 * u
        elif d == 'pitch_down':
            target_r[0] = -deg * 1.2
            target_p[1] = -1.0 * u
        elif d == 'yaw_left':
            target_r[1] = deg * 1.5
            target_r[2] = 15.0 * u
        elif d == 'yaw_right':
            target_r[1] = -deg * 1.5
            target_r[2] = -15.0 * u
        elif d == 'roll_left':
            target_r[2] = deg * 1.8
        elif d == 'roll_right':
            target_r[2] = -deg * 1.8

        self.target_pos = target_p
        self.target_rot = target_r

    def reset_to_center(self):
        """Reset targets to origin hover state."""
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_rot = np.zeros(3, dtype=np.float32)
        self.active_thrust_dir = None


# =====================================================================
# MAIN EXPERIMENT TASK APPLICATION
# =====================================================================
class Spatial3DTaskApp(BaseTaskApp):
    """3D Spatial Motor Imagery & 6-DoF Neurofeedback Task Application."""

    def __init__(self):
        super().__init__(marker_name='Spatial3DMarkers', source_id='Spatial3D_Task_2026')
        self.setWindowTitle("🚀 3D Spatial Motor Imagery & 6-DoF Neurofeedback Suite")
        self.resize(1120, 840)

        # Sound Effects & Cues
        self.sound_dir = os.path.join(ASSETS_SOUNDS, "spatial_cues")
        self.cue_files = generate_spatial_cue_wavs(self.sound_dir)

        # Audio Players
        self.cue_player = QMediaPlayer(self)
        self.cue_audio = QAudioOutput(self)
        self.cue_player.setAudioOutput(self.cue_audio)
        self.cue_audio.setVolume(0.9)

        # Timings (seconds)
        self.t_fixation = 2.0
        self.t_cue = 1.5
        self.t_task = 4.0
        self.t_rest = 2.0

        # Trial & Block State
        self.current_block_idx = 0
        self.blocks = []
        self.trials = []
        self.trial_idx = 0
        self.current_phase_name = "Idle"
        self.current_trial_direction = ""
        self.current_intensity_step = 255

        # Timers
        self.trial_timer = QTimer(self)
        self.trial_timer.setSingleShot(True)
        self.trial_timer.timeout.connect(self.advance_trial_phase)

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.phase_start_time = 0.0
        self.current_phase_duration = 1.0

        self.init_ui()

    def play_cue_sound(self, sound_key):
        k = sound_key.lower()
        if k in self.cue_files and os.path.exists(self.cue_files[k]):
            self.cue_player.stop()
            self.cue_player.setSource(QUrl.fromLocalFile(self.cue_files[k]))
            self.cue_player.play()

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(13, 17, 23))
        palette.setColor(QPalette.WindowText, QColor(240, 240, 245))
        palette.setColor(QPalette.Base, QColor(22, 27, 34))
        palette.setColor(QPalette.Text, QColor(240, 240, 245))
        palette.setColor(QPalette.Button, QColor(33, 38, 45))
        palette.setColor(QPalette.ButtonText, QColor(240, 240, 245))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0D1117;
                color: #F0F0F5;
            }
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #58A6FF;
                border: 1px solid #30363D;
                border-radius: 8px;
                margin-top: 6px;
                padding-top: 14px;
            }
            QLineEdit, QComboBox {
                background-color: #161B22;
                color: #FFFFFF;
                border: 1px solid #30363D;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton {
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 16px;
            }
        """)

        self.stacked_widget = QStackedWidget(self)
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
        layout.setSpacing(14)

        title = QLabel("🚀 3D Spatial Motor Imagery & 6-DoF Neurofeedback")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #58A6FF;")
        layout.addWidget(title)

        subtitle = QLabel("Full 6 Degrees of Freedom: Translations (Surge/Sway/Heave) & Rotations (Pitch/Yaw/Roll)")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #8B949E;")
        layout.addWidget(subtitle)

        # Participant Metadata Box
        meta_group = QGroupBox("👤 Participant Metadata")
        meta_form = QFormLayout(meta_group)
        self.sub_input = QLineEdit("01")
        self.sub_input.setFixedWidth(70)
        self.ses_input = QLineEdit("01")
        self.ses_input.setFixedWidth(70)
        meta_form.addRow("Subject ID:", self.sub_input)
        meta_form.addRow("Session ID:", self.ses_input)
        layout.addWidget(meta_group)

        # Paradigm & 3D Model Configuration
        config_group = QGroupBox("⚙️ 3D Paradigm & Object Settings")
        form = QFormLayout(config_group)
        form.setSpacing(10)

        self.paradigm_combo = QComboBox()
        self.paradigm_combo.addItems([
            "6-Direction Translation: Forward, Backward, Left, Right, Up, Down (Surge, Sway, Heave)",
            "6-Direction Rotation: Pitch Up/Down, Yaw Left/Right, Roll Left/Right",
            "Full 12-Direction 6-DoF: All Translations + Rotations",
            "4-Direction Ground Plane: Forward, Backward, Left, Right",
            "Flight Control 4-Class: Pitch Up/Down, Yaw Left/Right"
        ])
        form.addRow("Spatial Paradigm:", self.paradigm_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Nautilus Cyber-Shuttle (Fuselage, Delta Wings, Thrusters)",
            "6-DoF Orientation Cube (Vibrant Directional Faces)"
        ])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        form.addRow("3D Vehicle Model:", self.model_combo)

        self.cam_combo = QComboBox()
        self.cam_combo.addItems([
            "Fixed Observer (Forward Shrinks, Backward Grows)",
            "Elastic Damped Follow (Dynamic Surge Lag)",
            "Tethered Chase (Rigid Follow)"
        ])
        self.cam_combo.currentIndexChanged.connect(self.on_cam_mode_changed)
        form.addRow("Camera Mode:", self.cam_combo)

        self.reps_combo = QComboBox()
        self.reps_combo.addItems([
            "5 trials per direction (Standard Pilot)",
            "10 trials per direction (Full Calibration)",
            "15 trials per direction (Deep Dataset)"
        ])
        form.addRow("Trials per Direction:", self.reps_combo)

        self.audio_cue_check = QCheckBox("Enable Directional 3D Audio Cues")
        self.audio_cue_check.setChecked(True)
        form.addRow("Audio Guidance:", self.audio_cue_check)

        layout.addWidget(config_group)

        # Visual Direction Legend
        legend_group = QGroupBox("🧭 6-DoF Kinematics Mapping")
        leg_layout = QHBoxLayout(legend_group)
        lbl1 = QLabel("<b>Forward/Back:</b> Surge (Z-axis)")
        lbl1.setStyleSheet("color: #58A6FF; font-size: 11px;")
        lbl2 = QLabel("<b>Left/Right:</b> Sway (X-axis)")
        lbl2.setStyleSheet("color: #3FB950; font-size: 11px;")
        lbl3 = QLabel("<b>Up/Down:</b> Heave (Y-axis)")
        lbl3.setStyleSheet("color: #D29922; font-size: 11px;")
        lbl4 = QLabel("<b>Rotations:</b> Pitch, Yaw, Roll")
        lbl4.setStyleSheet("color: #F85149; font-size: 11px;")
        leg_layout.addWidget(lbl1)
        leg_layout.addWidget(lbl2)
        leg_layout.addWidget(lbl3)
        leg_layout.addWidget(lbl4)
        layout.addWidget(legend_group)

        # Proportional 256-Step Intensity Box (uint8 0..255 | float64 0.0..1.0)
        intensity_group = QGroupBox("⚡ Proportional Dynamic Activation (256 Discrete Steps / Analog Pressure)")
        i_layout = QVBoxLayout(intensity_group)
        self.lbl_intensity_val = QLabel("Intensity Level: 255 / 255 (Float64: 1.000 | Hex: 0xFF)")
        self.lbl_intensity_val.setStyleSheet("color: #58A6FF; font-size: 11px; font-weight: bold;")
        i_layout.addWidget(self.lbl_intensity_val)

        self.slider_intensity = QSlider(Qt.Horizontal)
        self.slider_intensity.setRange(1, 255)
        self.slider_intensity.setValue(255)
        self.slider_intensity.valueChanged.connect(self.on_intensity_changed)
        i_layout.addWidget(self.slider_intensity)
        layout.addWidget(intensity_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_free_flight = QPushButton("🎮 Free Flight & Calibration Test")
        self.btn_free_flight.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_free_flight.setStyleSheet("background-color: #1F6FEB; color: white;")
        self.btn_free_flight.clicked.connect(self.start_free_flight)
        btn_layout.addWidget(self.btn_free_flight)

        self.btn_practice = QPushButton("🎯 Quick Practice (4 Trials)")
        self.btn_practice.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_practice.setStyleSheet("background-color: #388BFD; color: white;")
        self.btn_practice.clicked.connect(self.start_practice)
        btn_layout.addWidget(self.btn_practice)

        self.btn_start = QPushButton("🚀 Start 3D Spatial Experiment")
        self.btn_start.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_start.setStyleSheet("background-color: #238636; color: white;")
        self.btn_start.clicked.connect(self.start_experiment)
        btn_layout.addWidget(self.btn_start)

        layout.addLayout(btn_layout)
        return widget

    def create_task_screen(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Header Info Bar
        header = QHBoxLayout()
        self.lbl_status = QLabel("Condition: 3D Spatial Motor Imagery")
        self.lbl_status.setFont(QFont("Arial", 13, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #58A6FF;")
        header.addWidget(self.lbl_status)

        header.addStretch()

        self.lbl_trial_count = QLabel("Trial 0 / 0")
        self.lbl_trial_count.setFont(QFont("Arial", 13, QFont.Bold))
        self.lbl_trial_count.setStyleSheet("color: #8B949E;")
        header.addWidget(self.lbl_trial_count)

        btn_exit = QPushButton("✕ Exit to Setup")
        btn_exit.setStyleSheet("background-color: #21262D; color: #F85149; padding: 4px 10px;")
        btn_exit.clicked.connect(self.stop_experiment)
        header.addWidget(btn_exit)

        layout.addLayout(header)

        # 3D OpenGL Viewport
        self.canvas_3d = Spatial3DCanvas(self)
        layout.addWidget(self.canvas_3d, stretch=1)

        # Center Prompt / Instruction Bar
        self.lbl_instruction = QLabel("FOCUS ON THE VEHICLE")
        self.lbl_instruction.setFont(QFont("Arial", 18, QFont.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_instruction.setStyleSheet("color: #F0F0F5; padding: 6px;")
        layout.addWidget(self.lbl_instruction)

        # Progress Bar for Phase Timing
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background-color: #161B22; border-radius: 4px; }
            QProgressBar::chunk { background-color: #2EA043; border-radius: 4px; }
        """)
        layout.addWidget(self.progress_bar)

        # Free Flight Instructions (hidden during trials)
        self.lbl_free_flight_help = QLabel(
            "🎮 <b>Free Flight Controls:</b> [W/S] Forward/Back | [A/D] Strafe Left/Right | [Space/Shift] Up/Down | [Arrows] Pitch/Yaw | [Q/E] Roll | [M] Switch Model | [V] Camera Perspective | [C] Center"
        )
        self.lbl_free_flight_help.setAlignment(Qt.AlignCenter)
        self.lbl_free_flight_help.setStyleSheet("color: #8B949E; font-size: 11px;")
        self.lbl_free_flight_help.hide()
        layout.addWidget(self.lbl_free_flight_help)

        return widget

    def on_cam_mode_changed(self, idx):
        modes = ['fixed', 'elastic', 'chase']
        self.canvas_3d.camera_mode = modes[idx] if idx < len(modes) else 'fixed'
        self.canvas_3d.update()

    def on_model_changed(self, idx):
        self.canvas_3d.object_type = 'cube' if idx == 1 else 'nautilus'
        self.canvas_3d.update()

    def on_intensity_changed(self, val):
        self.current_intensity_step = val
        u = val / 255.0
        hex_str = f"0x{val:02X}"
        self.lbl_intensity_val.setText(f"Intensity Level: {val} / 255 (Float64: {u:.3f} | Hex: {hex_str})")
        self.canvas_3d.active_intensity = u

    def start_free_flight(self):
        """Enter interactive free flight testing mode."""
        self.canvas_3d.object_type = 'cube' if self.model_combo.currentIndex() == 1 else 'nautilus'
        cam_idx = self.cam_combo.currentIndex()
        modes = ['fixed', 'elastic', 'chase']
        self.canvas_3d.camera_mode = modes[cam_idx] if cam_idx < len(modes) else 'fixed'
        self.canvas_3d.autonomous_control = False
        self.canvas_3d.show_cue_guide = False
        self.lbl_instruction.setText("FREE FLIGHT TESTING MODE")
        self.lbl_instruction.setStyleSheet("color: #58A6FF; font-weight: bold;")
        self.lbl_status.setText("Mode: Manual Free Flight & Calibration")
        self.lbl_trial_count.setText("Manual Pilot")
        self.lbl_free_flight_help.show()
        self.progress_bar.setValue(100)

        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.canvas_3d.setFocus()

    def start_practice(self):
        self.setup_experiment(practice=True)
        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.start_trials()

    def start_experiment(self):
        self.setup_experiment(practice=False)
        self.stacked_widget.setCurrentWidget(self.task_screen)
        self.start_trials()

    def setup_experiment(self, practice=False):
        self.canvas_3d.object_type = 'cube' if self.model_combo.currentIndex() == 1 else 'nautilus'
        cam_idx = self.cam_combo.currentIndex()
        modes = ['fixed', 'elastic', 'chase']
        self.canvas_3d.camera_mode = modes[cam_idx] if cam_idx < len(modes) else 'fixed'
        self.canvas_3d.autonomous_control = True
        self.lbl_free_flight_help.hide()

        p_idx = self.paradigm_combo.currentIndex()
        if p_idx == 0:
            self.directions = ["Forward", "Backward", "Left", "Right", "Up", "Down"]
            paradigm_code = "Translation6D"
        elif p_idx == 1:
            self.directions = ["Pitch_Up", "Pitch_Down", "Yaw_Left", "Yaw_Right", "Roll_Left", "Roll_Right"]
            paradigm_code = "Rotation6D"
        elif p_idx == 2:
            self.directions = [
                "Forward", "Backward", "Left", "Right", "Up", "Down",
                "Pitch_Up", "Pitch_Down", "Yaw_Left", "Yaw_Right", "Roll_Left", "Roll_Right"
            ]
            paradigm_code = "Full6DoF"
        elif p_idx == 3:
            self.directions = ["Forward", "Backward", "Left", "Right"]
            paradigm_code = "GroundPlane4D"
        else:
            self.directions = ["Pitch_Up", "Pitch_Down", "Yaw_Left", "Yaw_Right"]
            paradigm_code = "FlightControl4D"

        reps_map = {0: 5, 1: 10, 2: 15}
        reps = 1 if practice else reps_map[self.reps_combo.currentIndex()]

        # Build randomized trial sequence
        self.trials = []
        for d in self.directions:
            self.trials.extend([d] * reps)
        random.shuffle(self.trials)
        self.trial_idx = 0

        sub = self.sub_input.text().strip().replace('sub-', '')
        ses = self.ses_input.text().strip().replace('ses-', '')
        mode = "Practice" if practice else "Experiment"

        self.send_marker(f"Experiment_Start_Sub_{sub}_Ses_{ses}_Paradigm_{paradigm_code}_Mode_{mode}")

    def start_trials(self):
        self.trial_idx = 0
        self.run_next_trial()

    def run_next_trial(self):
        if self.trial_idx >= len(self.trials):
            self.finish_experiment()
            return

        self.current_trial_direction = self.trials[self.trial_idx]
        self.lbl_trial_count.setText(f"Trial {self.trial_idx + 1} / {len(self.trials)}")

        self.send_marker(f"Trial_Start_{self.trial_idx + 1}_Dir_{self.current_trial_direction}")
        self.send_marker("Fixation", self.t_fixation)

        if self.audio_cue_check.isChecked():
            self.play_cue_sound('fixation')

        self.canvas_3d.show_cue_guide = False
        self.canvas_3d.reset_to_center()

        self.set_phase("Fixation", "FOCUS & CLEAR YOUR MIND", "#8B949E", self.t_fixation)

    def advance_trial_phase(self):
        d = self.current_trial_direction

        if self.current_phase_name == "Fixation":
            # Phase 2: 3D Hologram Cue (1.5s)
            self.send_marker(f"Cue_{d}", self.t_cue)
            self.canvas_3d.show_cue_guide = True
            self.canvas_3d.cue_direction = d

            if self.audio_cue_check.isChecked():
                self.play_cue_sound(d)

            prompt = f"IMAGINE: {d.upper().replace('_', ' ')}"
            self.set_phase("Cue", prompt, "#58A6FF", self.t_cue)

        elif self.current_phase_name == "Cue":
            # Phase 3: Action Execution / Mental Imagery (4.0s)
            step = getattr(self, 'current_intensity_step', 255)
            u = step / 255.0
            self.send_marker(f"Task_{d}_step_{step}_val_{u:.3f}", self.t_task)
            self.canvas_3d.show_cue_guide = False
            self.canvas_3d.execute_action(d, step)

            prompt = f"ACTION: EXECUTING {d.upper().replace('_', ' ')} [{step}/255 | {int(u*100)}%]"
            self.set_phase("Task", prompt, "#2EA043", self.t_task)

        elif self.current_phase_name == "Task":
            # Phase 4: Rest (2.0s)
            self.send_marker("Rest", self.t_rest)
            self.canvas_3d.reset_to_center()

            if self.audio_cue_check.isChecked():
                self.play_cue_sound('rest')

            self.set_phase("Rest", "RELAX & REST", "#8B949E", self.t_rest)

        elif self.current_phase_name == "Rest":
            self.send_marker(f"Trial_End_{self.trial_idx + 1}")
            self.trial_idx += 1
            self.run_next_trial()

    def set_phase(self, phase_name, text, color, duration):
        self.current_phase_name = phase_name
        self.lbl_instruction.setText(text)
        self.lbl_instruction.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 20px;")

        self.phase_start_time = time.time()
        self.current_phase_duration = duration

        self.trial_timer.start(int(duration * 1000))
        self.progress_timer.start(25)  # 40 Hz progress bar update

    def update_progress(self):
        elapsed = time.time() - self.phase_start_time
        pct = min(1.0, max(0.0, elapsed / self.current_phase_duration))
        self.progress_bar.setValue(int(pct * 100))
        if pct >= 1.0:
            self.progress_timer.stop()

    def finish_experiment(self):
        self.send_marker("Experiment_Completed")
        self.canvas_3d.reset_to_center()
        self.lbl_instruction.setText("SESSION COMPLETE!")
        self.lbl_instruction.setStyleSheet("color: #3FB950; font-weight: bold; font-size: 22px;")
        QMessageBox.information(self, "Completed", "3D Spatial BCI Experiment Session Completed!")
        self.stacked_widget.setCurrentWidget(self.config_screen)

    def stop_experiment(self):
        self.trial_timer.stop()
        self.progress_timer.stop()
        self.canvas_3d.autonomous_control = True
        self.canvas_3d.reset_to_center()
        self.send_marker("Experiment_Aborted")
        self.stacked_widget.setCurrentWidget(self.config_screen)


def main():
    app = QApplication(sys.argv)
    window = Spatial3DTaskApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
