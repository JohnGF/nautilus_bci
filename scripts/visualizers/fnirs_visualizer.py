"""
Real-Time fNIRS (Functional Near-Infrared Spectroscopy) Cortical Visualizer
Designed for Artinis & g.tec fNIRS Systems & Lab Streaming Layer (LSL)

Hardware Geometry:
- 2 Receivers (R1 right, R2 left) at the center of each pod & 8 Transmitters (T1..T8) around them
- Selectable View Perspectives: Frontal (Forehead), Central/Motor (Top-Down), Occipital (Rear)
- Optode Pairing: T1..T4 -> R1 | T5..T8 -> R2 (8 Channels x 2 Wavelengths)
- True Modified Beer-Lambert Law (MBLL) with Scalp Coupling Index (SCI) Live Verification.
"""

import sys
import os
import time
import math
import re
from collections import namedtuple

import numpy as np

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Try importing pylsl
try:
    from pylsl import StreamInlet, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

# Try importing scipy for filtering
try:
    import scipy.signal as signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==============================================================================
# MBLL (Modified Beer-Lambert Law) Physical Constants
# ==============================================================================
EPSILON = {
    760: {'hbo': 1486.586, 'hbr': 3843.707},
    850: {'hbo': 2526.391, 'hbr': 1798.643}
}
DEFAULT_DPF = 6.0

def compute_mbll_matrix(d, lambda1=760, lambda2=850, dpf=DEFAULT_DPF):
    """Matrix from decadic (log10) delta-OD at [lambda1, lambda2] to [dHbO, dHbR] in uM."""
    e_matrix = np.array([
        [EPSILON[lambda1]['hbo'], EPSILON[lambda1]['hbr']],
        [EPSILON[lambda2]['hbo'], EPSILON[lambda2]['hbr']]
    ], dtype=np.float64)
    pathlength = d * dpf
    inv_e = np.linalg.pinv(e_matrix * pathlength) * 1e6  # Output in µM
    return inv_e


# ==============================================================================
# Stream Layout & Signal Processing
# ==============================================================================
KIND_CONC = "conc"            # Columns are already HbO / HbR
KIND_OD = "od"                # Columns are optical densities
KIND_INTENSITY = "intensity"  # Columns are raw light counts
AUX_LABEL_WORDS = ("battery", "counter", "link", "validation", "trigger")
LABEL_PATTERN = re.compile(r"T[Xx]?(\d+)\D*?(hbo|hbr|760|850)", re.IGNORECASE)

StreamLayout = namedtuple("StreamLayout", "kind first second labelled")
StreamLayout.__doc__ = """Columns of each channel C1..Cn: first = 760nm or HbO, second = 850nm or HbR.
kind is None until the signal magnitude can decide between OD and intensity."""


def is_fnirs_stream(name, stream_type):
    """True for fNIRS streams. EEG and diagnostics streams can share a device name, so they never match."""
    stream_type, name = stream_type.upper(), name.upper()
    if "NIRS" in stream_type:
        return True
    if stream_type in ("EEG", "DIAGNOSTICS"):
        return False
    return any(key in name for key in ("ARTINIS", "FNIRS", "OCTAMON"))


def parse_stream_layout(labels, units, channel_count):
    """Find each channel's columns from the LSL labels, e.g. the bridge's R1_T1_760nm, R1_T1_850nm, ...

    Unlabelled streams fall back to the g.SENSOR manual's order: odd channels 760 nm, even channels 850 nm.
    """
    pairs, tags = {}, set()
    for col, label in enumerate(labels):
        match = LABEL_PATTERN.search(label or "")
        if match:
            tag = match.group(2).lower()
            tags.add(tag)
            pairs.setdefault(int(match.group(1)), {})[tag in ("hbo", "760")] = col
    channels = [pairs[ch] for ch in sorted(pairs)]
    if channels and all(len(cols) == 2 for cols in channels):
        if "hbo" in tags:
            kind = KIND_CONC
        else:
            kind = KIND_OD if any((unit or "").strip().upper() == "OD" for unit in units) else None
        return StreamLayout(kind, [c[True] for c in channels][:8], [c[False] for c in channels][:8], True)

    columns = [
        col for col in range(channel_count)
        if not (col < len(labels) and any(word in (labels[col] or "").lower() for word in AUX_LABEL_WORDS))
    ]
    n_pairs = min(8, len(columns) // 2)
    return StreamLayout(None, columns[0:2 * n_pairs:2], columns[1:2 * n_pairs:2], False)


def guess_signal_kind(samples):
    """OD sits near 0-4 while raw light counts are in the hundreds or more."""
    return KIND_INTENSITY if np.nanmedian(np.abs(samples)) > 150.0 else KIND_OD


def zero_phase_filter(sos, data):
    """Zero-phase filter along time, or the data unchanged if it is too short to pad.

    The default odd padding forces the newest sample to about 0, which would blank the live
    head map, so this pads evenly. The newest few seconds are still only an estimate.
    """
    if sos is None or data.shape[0] <= 30:
        return data
    centered = data - np.mean(data, axis=0)
    return signal.sosfiltfilt(sos, centered, axis=0, padtype='even', padlen=data.shape[0] - 1)


def to_concentrations(first, second, base_first, base_second, kind, matrix, sos=None):
    """Convert (time, channel) columns to baseline-relative (dHbO, dHbR) in uM.

    OD from the device is assumed decadic, the same convention as the EPSILON table.
    """
    if kind == KIND_INTENSITY:
        d_first = -np.log10(np.maximum(first, 1.0) / np.maximum(base_first, 1.0))
        d_second = -np.log10(np.maximum(second, 1.0) / np.maximum(base_second, 1.0))
    else:
        d_first = first - base_first
        d_second = second - base_second
    d_first = zero_phase_filter(sos, d_first)
    d_second = zero_phase_filter(sos, d_second)
    if kind == KIND_CONC:
        return d_first, d_second
    return (matrix[0, 0] * d_first + matrix[0, 1] * d_second,
            matrix[1, 0] * d_first + matrix[1, 1] * d_second)


def scalp_coupling_index(first, second, sos_cardiac, min_samples):
    """Per-channel correlation of the cardiac pulse across wavelengths.

    0 = flat or uncoupled, NaN = cannot be measured (too little data, or no scipy).
    """
    sci = np.full(first.shape[1], np.nan)
    if sos_cardiac is None or first.shape[0] < max(min_samples, 31):
        return sci
    card_first = zero_phase_filter(sos_cardiac, first)
    card_second = zero_phase_filter(sos_cardiac, second)
    for ch in range(first.shape[1]):
        if np.std(card_first[:, ch]) < 1e-6 or np.std(card_second[:, ch]) < 1e-6:
            sci[ch] = 0.0
        else:
            sci[ch] = max(0.0, float(np.corrcoef(card_first[:, ch], card_second[:, ch])[0, 1]))
    return sci


# ==============================================================================
# Optode Montages (g.SENSOR 8 fNIRS: 2 Receivers, 8 Transmitters)
# ==============================================================================
# Layouts follow the g.SENSOR 8 fNIRS manual v1.20.01, section 6.4 (pages 11 and 15):
# R1 and T1..T4 sit on the RIGHT hemisphere, R2 and T5..T8 on the LEFT. T1..T4 run in order
# around the pod, and T5..T8 mirror them, so T1 and T5 are the same position on each side.
#
# Every view keeps the wearer's left on screen-left. Coordinates are in head radii, x right /
# y down from head center. "offsets" are T1..T4 relative to R1; the left pod is the mirror image.
# "sites" are the nearest 10-10 positions in T1..T8 order, as a placement aid: the manual allows
# the pads to be moved. The manual has no occipital layout, so that one extends the central pad.
ZONES = ["FRONTAL", "CENTRAL", "OCCIPITAL"]

MONTAGES = {
    "FRONTAL": {
        "combo": "Frontal (Front View - Forehead)",
        "title": "FRONT VIEW - FOREHEAD",
        "caption": "mirror view: wearer's left on screen-left",
        "pods": ("Right Prefrontal", "Left Prefrontal"),
        "default_power": "LP",
        "distance_cm": 3.2,
        "rx": ((0.49, -0.38), (-0.49, -0.38)),
        "rx_sites": ("AF6", "AF5"),
        # The manual draws this from above; a front view turns it upside down, so T1 (nearest the nose) is lowest
        "offsets": ((-0.16, 0.33), (0.37, -0.02), (0.17, -0.31), (-0.34, 0.02)),
        "roles": ("Inferior", "Lateral", "Superior", "Medial"),
        "sites": ("Fp2", "AF8", "F6", "AF4", "Fp1", "AF7", "F5", "AF3"),
    },
    "CENTRAL": {
        "combo": "Central / Motor (Top-Down View - Central Scalp)",
        "title": "TOP-DOWN VIEW - CENTRAL / MOTOR",
        "caption": "from above, nose up",
        "pods": ("Right Motor", "Left Motor"),
        "default_power": "HP",
        "distance_cm": 3.0,
        "rx": ((0.48, 0.0), (-0.48, 0.0)),
        "rx_sites": ("C4", "C3"),
        "offsets": ((-0.28, -0.28), (0.28, -0.28), (0.28, 0.28), (-0.28, 0.28)),
        "roles": ("Anterior-Medial", "Anterior-Lateral", "Posterior-Lateral", "Posterior-Medial"),
        "sites": ("FC2-FC4", "FC4-FC6", "CP4-CP6", "CP2-CP4", "FC1-FC3", "FC3-FC5", "CP3-CP5", "CP1-CP3"),
    },
    "OCCIPITAL": {
        "combo": "Occipital (Rear View - Back of Head)",
        "title": "REAR VIEW - OCCIPITAL",
        "caption": "from behind; not in manual, central pad reused",
        "pods": ("Right Occipital", "Left Occipital"),
        "default_power": "HP",
        "distance_cm": 3.0,
        "rx": ((0.42, 0.20), (-0.42, 0.20)),
        "rx_sites": ("PO4", "PO3"),
        "offsets": ((-0.28, -0.28), (0.28, -0.28), (0.28, 0.28), (-0.28, 0.28)),
        "roles": ("Superior-Medial", "Superior-Lateral", "Inferior-Lateral", "Inferior-Medial"),
        "sites": ("P2", "P6", "PO8", "O2", "P1", "P5", "PO7", "O1"),
    },
}
MONTAGES["MOTOR"] = MONTAGES["CENTRAL"]


def load_optode_powers(zone):
    """Saved LP/HP type for T1..T8 in this zone, or the zone's default if none is saved."""
    key = "CENTRAL" if zone == "MOTOR" else zone
    saved = QtCore.QSettings("nautilus_bci", "fnirs_visualizer").value(f"optode_power/{key}", "")
    powers = str(saved).split(",")
    if len(powers) == 8 and all(p in ("LP", "HP") for p in powers):
        return powers
    return [MONTAGES[zone]["default_power"]] * 8


def save_optode_powers(zone, powers):
    key = "CENTRAL" if zone == "MOTOR" else zone
    QtCore.QSettings("nautilus_bci", "fnirs_visualizer").setValue(f"optode_power/{key}", ",".join(powers))


def placement_guide_html(zone, powers):
    """Hardware socket legend for one montage, with the LP/HP optode type on each transmitter."""
    montage = MONTAGES[zone]
    parts = []
    for pod in range(2):
        parts.append(f"<b>{montage['pods'][pod]} Pod:</b><br>")
        parts.append(f"• <b>[R{pod + 1}] Center:</b> Receiver (~{montage['rx_sites'][pod]})<br>")
        for t in range(pod * 4, pod * 4 + 4):
            parts.append(f"• <b>[T{t + 1}] {montage['roles'][t % 4]}:</b> ~{montage['sites'][t]} [{powers[t]}]<br>")
        if pod == 0:
            parts.append("<br>")
    parts.append(f"<br><i>{montage['distance_cm'] * 10:.0f} mm spacing. Default for this area: {montage['default_power']}.")
    if "HP" in powers:
        parts.append(" HP battery: ~30-60 min.")
    parts.append("</i>")
    return "".join(parts)


# ==============================================================================
# Interactive 2D Optode Head Map Widget (2 Receivers, 8 Transmitters)
# ==============================================================================
class OptodeHeadMapWidget(QtWidgets.QFrame):
    """
    2D Head Map supporting 3 view perspectives with 2 Central Receivers (R1, R2) and 8 Transmitters (T1..T8).
    Arranged as 2 Sensor Pods (4 Transmitters around each Central Receiver), as in the g.SENSOR manual:
    - Right Pod: 1 Center Receiver (R1) + 4 Transmitters (T1..T4, in order around the pod) -> 4 Channels (C1..C4)
    - Left Pod: 1 Center Receiver (R2) + 4 Transmitters (T5..T8, mirroring T1..T4) -> 4 Channels (C5..C8)
    Supports Frontal (Forehead), Central/Motor (Top-Down), and Occipital (Rear) view perspectives.
    """

    # Tx -> Rx for C1..C8: fixed by the hardware, independent of the view.
    ch_pairings = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 1)]
    COLOR_SCALE_UM = 1.0

    powers_changed = QtCore.Signal()

    def __init__(self, active_zone="FRONTAL", parent=None):
        super().__init__(parent)
        self.active_zone = active_zone
        self.channel_values = np.zeros(8)
        self.channel_sci = np.full(8, np.nan)  # Unknown until measured
        self.hover_tx = None
        self.setMinimumSize(300, 330)
        self.setMouseTracking(True)
        self.setToolTip("Click a transmitter to switch it between low power (LP) and high power (HP)")
        self.recompute_montage()

    def set_zone(self, zone):
        self.active_zone = zone
        self.recompute_montage()
        self.update()

    def recompute_montage(self):
        montage = MONTAGES[self.active_zone]
        self.montage = montage
        self.tx_powers = load_optode_powers(self.active_zone)
        self.rx_positions = list(montage["rx"])
        self.tx_positions = [
            (x + side * dx, y + dy)
            for (x, y), side in zip(montage["rx"], (1, -1))  # The left pod mirrors the right pod
            for dx, dy in montage["offsets"]
        ]
        self.channel_labels = []
        for ch in range(8):
            hemi = "R" if ch < 4 else "L"
            role = "-".join(word[:3] for word in montage["roles"][ch % 4].split("-"))
            self.channel_labels.append(
                f"C{ch + 1}: {hemi}-{role} (T{ch + 1}-R{ch // 4 + 1} ~{montage['sites'][ch]})"
            )

    def update_metrics(self, values, sci_scores):
        """values and sci_scores hold one entry per available channel; the rest stay unknown."""
        self.channel_values = np.zeros(8)
        self.channel_sci = np.full(8, np.nan)
        self.channel_values[:len(values)] = values[:8]
        self.channel_sci[:len(sci_scores)] = sci_scores[:8]
        self.update()

    def get_channel_label(self, idx):
        if idx < len(self.channel_labels):
            return self.channel_labels[idx]
        return f"Ch {idx+1}"

    def _head_frame(self):
        """Pixel center and radius of the head, shared by painting and click hit-testing."""
        top, bottom = 34, self.height() - 44  # Header above, color bar and hint below
        return self.width() / 2, (top + bottom) / 2, min(self.width(), bottom - top) / 2.3

    def _tx_at(self, pos):
        cx, cy, radius = self._head_frame()
        for i, (x_norm, y_norm) in enumerate(self.tx_positions):
            if math.hypot(pos.x() - (cx + x_norm * radius), pos.y() - (cy + y_norm * radius)) <= 11:
                return i
        return None

    def mouseMoveEvent(self, event):
        hover = self._tx_at(event.position())
        if hover != self.hover_tx:
            self.hover_tx = hover
            self.setCursor(QtCore.Qt.PointingHandCursor if hover is not None else QtCore.Qt.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.hover_tx = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        tx = self._tx_at(event.position())
        if tx is None:
            super().mousePressEvent(event)
            return
        self.tx_powers[tx] = "HP" if self.tx_powers[tx] == "LP" else "LP"
        save_optode_powers(self.active_zone, self.tx_powers)
        self.update()
        self.powers_changed.emit()

    def _draw_ears(self, painter, cx, ear_y, half_gap):
        painter.drawEllipse(QtCore.QPointF(cx - half_gap, ear_y), 7, 15)
        painter.drawEllipse(QtCore.QPointF(cx + half_gap, ear_y), 7, 15)

    def _draw_midline(self, painter, cx, y_top, y_bottom):
        painter.setPen(QtGui.QPen(QtGui.QColor("#58a6ff"), 1.0, QtCore.Qt.DashLine))
        painter.drawLine(QtCore.QPointF(cx, y_top), QtCore.QPointF(cx, y_bottom))

    def _draw_head_front(self, painter, cx, cy, r):
        """Forehead outline, ears, eyebrows and eyes, split by the median line."""
        self._draw_ears(painter, cx, cy + 0.25 * r, 0.95 * r)
        painter.drawEllipse(QtCore.QPointF(cx, cy), 0.95 * r, 1.05 * r)
        for sign in (-1, 1):
            brow = QtGui.QPainterPath()
            brow.moveTo(cx + sign * 0.14 * r, cy + 0.27 * r)
            brow.quadTo(cx + sign * 0.36 * r, cy + 0.17 * r, cx + sign * 0.60 * r, cy + 0.29 * r)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(brow)
            painter.drawEllipse(QtCore.QPointF(cx + sign * 0.37 * r, cy + 0.42 * r), 0.15 * r, 0.06 * r)
        self._draw_midline(painter, cx, cy - 1.05 * r, cy + 0.35 * r)

    def _draw_head_top(self, painter, cx, cy, r):
        """Oval scalp seen from above, nose pointing up, ears on the sides, sagittal midline."""
        self._draw_ears(painter, cx, cy, 0.90 * r)
        nose = QtGui.QPainterPath()
        nose.moveTo(cx - 0.12 * r, cy - 0.97 * r)
        nose.lineTo(cx, cy - 1.14 * r)
        nose.lineTo(cx + 0.12 * r, cy - 0.97 * r)
        painter.drawPath(nose)
        painter.drawEllipse(QtCore.QPointF(cx, cy), 0.90 * r, 1.02 * r)
        self._draw_midline(painter, cx, cy - 1.02 * r, cy + 1.02 * r)

    def _draw_head_rear(self, painter, cx, cy, r):
        """Back of the head above the neck baseline, ears on the sides, occipital midline."""
        head_cy = cy - 0.12 * r
        self._draw_ears(painter, cx, head_cy + 0.05 * r, 0.95 * r)
        neck = QtGui.QPainterPath()
        neck.moveTo(cx - 0.42 * r, head_cy + 0.80 * r)
        neck.lineTo(cx - 0.50 * r, cy + 1.12 * r)
        neck.lineTo(cx + 0.50 * r, cy + 1.12 * r)
        neck.lineTo(cx + 0.42 * r, head_cy + 0.80 * r)
        painter.drawPath(neck)
        painter.drawEllipse(QtCore.QPointF(cx, head_cy), 0.95 * r, 0.95 * r)
        self._draw_midline(painter, cx, head_cy - 0.95 * r, cy + 1.12 * r)

    def _draw_head(self, painter, cx, cy, r):
        fill = QtGui.QRadialGradient(QtCore.QPointF(cx, cy - 0.2 * r), 1.3 * r)
        fill.setColorAt(0.0, QtGui.QColor("#232b38"))
        fill.setColorAt(1.0, QtGui.QColor("#151a22"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#55627c"), 2.2))
        painter.setBrush(QtGui.QBrush(fill))
        if self.active_zone == "FRONTAL":
            self._draw_head_front(painter, cx, cy, r)
        elif self.active_zone == "OCCIPITAL":
            self._draw_head_rear(painter, cx, cy, r)
        else:
            self._draw_head_top(painter, cx, cy, r)

    def _set_font(self, painter, size, bold=False):
        font = painter.font()
        font.setPointSize(size)
        font.setBold(bold)
        painter.setFont(font)

    def _draw_header(self, painter, w):
        self._set_font(painter, 9, bold=True)
        painter.setPen(QtGui.QColor("#58a6ff"))
        painter.drawText(0, 5, w, 14, QtCore.Qt.AlignCenter, self.montage["title"])
        self._set_font(painter, 7)
        painter.setPen(QtGui.QColor("#8b949e"))
        painter.drawText(0, 20, w, 12, QtCore.Qt.AlignCenter, self.montage["caption"])

    @staticmethod
    def _activation_color(value_um):
        """Diverging scale: blue (decrease), neutral grey (zero), red (increase)."""
        amount = float(np.clip(value_um / OptodeHeadMapWidget.COLOR_SCALE_UM, -1.0, 1.0))
        neutral, end = (61, 68, 80), (255, 71, 87) if amount >= 0 else (30, 144, 255)
        return QtGui.QColor(*[int(n + abs(amount) * (e - n)) for n, e in zip(neutral, end)])

    def _draw_footer(self, painter, w, h):
        bar = QtCore.QRectF(w / 2 - 55, h - 36, 110, 7)
        gradient = QtGui.QLinearGradient(bar.topLeft(), bar.topRight())
        gradient.setColorAt(0.0, self._activation_color(-self.COLOR_SCALE_UM))
        gradient.setColorAt(0.5, self._activation_color(0.0))
        gradient.setColorAt(1.0, self._activation_color(self.COLOR_SCALE_UM))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawRoundedRect(bar, 3, 3)

        self._set_font(painter, 7)
        painter.setPen(QtGui.QColor("#8b949e"))
        scale = f"{self.COLOR_SCALE_UM:g}"
        painter.drawText(int(bar.left()) - 62, h - 39, 58, 13, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"-{scale}")
        painter.drawText(int(bar.right()) + 4, h - 39, 70, 13, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, f"+{scale} µM HbO")
        painter.drawText(24, h - 22, w - 48, 16, QtCore.Qt.AlignCenter, "click a T to switch LP / HP")

        self._set_font(painter, 10, bold=True)
        painter.setPen(QtGui.QColor("#c9d1d9"))
        painter.drawText(8, h - 24, 18, 18, QtCore.Qt.AlignCenter, "L")
        painter.drawText(w - 26, h - 24, 18, 18, QtCore.Qt.AlignCenter, "R")

    def _draw_pod_outlines(self, painter, cx, cy, r):
        """Connect T1-T2-T3-T4 and T5-T6-T7-T8 so each pod reads as one shape around its receiver."""
        painter.setPen(QtGui.QPen(QtGui.QColor("#8b949e"), 1.4))
        painter.setBrush(QtGui.QColor(13, 17, 23, 150))
        for first in (0, 4):
            painter.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(cx + x * r, cy + y * r) for x, y in self.tx_positions[first:first + 4]
            ]))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx, cy, radius = self._head_frame()

        painter.setPen(QtGui.QPen(QtGui.QColor("#30363d"), 1))
        painter.setBrush(QtGui.QColor("#0d1117"))
        painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, w - 1, h - 1), 8, 8)

        self._draw_head(painter, cx, cy, radius)
        self._draw_header(painter, w)
        self._draw_footer(painter, w, h)
        self._draw_pod_outlines(painter, cx, cy, radius)

        # Draw Channels (Lines connecting Tx -> Rx)
        for i, (tx_i, rx_i) in enumerate(self.ch_pairings):
            tx_x, tx_y = cx + self.tx_positions[tx_i][0] * radius, cy + self.tx_positions[tx_i][1] * radius
            rx_x, rx_y = cx + self.rx_positions[rx_i][0] * radius, cy + self.rx_positions[rx_i][1] * radius
            mx, my = (tx_x + rx_x) / 2.0, (tx_y + rx_y) / 2.0

            sci = self.channel_sci[i]
            if np.isnan(sci):
                sci_color = QtGui.QColor("#6e7681")  # Grey (Not Measured)
            elif sci >= 0.70:
                sci_color = QtGui.QColor("#3fb950")  # Green (Good Contact)
            elif sci >= 0.45:
                sci_color = QtGui.QColor("#d29922")  # Yellow (Moderate)
            else:
                sci_color = QtGui.QColor("#f85149")  # Red (Noisy / Loose)

            painter.setPen(QtGui.QPen(sci_color, 2.0))
            painter.drawLine(QtCore.QPointF(tx_x, tx_y), QtCore.QPointF(rx_x, rx_y))

            # Channel Activation Node
            painter.setBrush(self._activation_color(self.channel_values[i]))
            painter.setPen(QtGui.QPen(sci_color, 1.5))
            painter.drawEllipse(QtCore.QPointF(mx, my), 6, 6)

            painter.setPen(QtGui.QColor("#e6edf3"))
            self._set_font(painter, 6, bold=True)
            # Label beside the line, clear of the node and the central receiver
            length = math.hypot(tx_x - rx_x, tx_y - rx_y)
            perp_x, perp_y = -(tx_y - rx_y) / length, (tx_x - rx_x) / length
            painter.drawText(int(mx + perp_x * 13 - 8), int(my + perp_y * 12 - 5), 16, 10, QtCore.Qt.AlignCenter, f"C{i+1}")

        # Draw 8 Transmitters: yellow = low power, orange-red = high power
        for i, (x_norm, y_norm) in enumerate(self.tx_positions):
            px = cx + x_norm * radius
            py = cy + y_norm * radius
            if i == self.hover_tx:
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
                painter.drawEllipse(QtCore.QPointF(px, py), 10, 10)
            painter.setBrush(QtGui.QColor("#e3b341" if self.tx_powers[i] == "LP" else "#f0562e"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.2))
            painter.drawEllipse(QtCore.QPointF(px, py), 6.5, 6.5)

            painter.setPen(QtGui.QColor("#c9d1d9"))
            self._set_font(painter, 6)
            # Above or below the optode, whichever is away from the receiver; side optodes label above
            below = y_norm - self.rx_positions[i // 4][1] > 0.1
            painter.drawText(int(px - 16), int(py + 9 if below else py - 19), 32, 10, QtCore.Qt.AlignCenter, f"T{i+1} {self.tx_powers[i]}")

        # Draw 2 Receivers (Blue Center Photodetectors R1, R2)
        for i, (x_norm, y_norm) in enumerate(self.rx_positions):
            px = cx + x_norm * radius
            py = cy + y_norm * radius
            painter.setBrush(QtGui.QColor("#58a6ff"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.8))
            painter.drawEllipse(QtCore.QPointF(px, py), 10, 10)

            painter.setPen(QtGui.QColor("#0d1117"))
            self._set_font(painter, 7, bold=True)
            painter.drawText(int(px - 10), int(py - 10), 20, 20, QtCore.Qt.AlignCenter, f"R{i+1}")


APP_STYLE = """
QFrame#card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
QLabel { background: transparent; border: none; color: #c9d1d9; }
QLabel#title { font-size: 15px; font-weight: bold; color: #58a6ff; }
QLabel#strong { font-weight: bold; }
QLabel#guide { font-size: 10px; }
QLabel#hint { color: #e3b341; font-size: 10px; font-weight: bold; }
QGroupBox { font-weight: bold; color: #8b949e; border: 1px solid #30363d; border-radius: 6px; margin-top: 8px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QComboBox, QDoubleSpinBox { background-color: #21262d; color: #79c0ff; font-weight: bold; padding: 4px 8px; border: 1px solid #30363d; border-radius: 4px; }
QPushButton { background-color: #21262d; color: #c9d1d9; font-weight: bold; padding: 5px 10px; border: 1px solid #30363d; border-radius: 6px; }
QPushButton:hover { border-color: #8b949e; }
QPushButton#connect { background-color: #238636; color: white; }
QPushButton#zero { background-color: #8957e5; color: white; }
QCheckBox { font-weight: bold; }
"""

WEAK_SCI = 0.45
GOOD_SCI = 0.70
WEAK_HOLD_SEC = 30.0  # Longer than the display window, so one bad stretch can't trigger a hint
STALE_STREAM_SEC = 3.0
DEFAULT_SCALE_UM = 1.0  # The manual gives no scale; task responses are typically around 1 uM


# ==============================================================================
# Main Real-Time fNIRS Application
# ==============================================================================
class fNIRSVisualizerWindow(QtWidgets.QMainWindow):
    """fNIRS Cortical Visualizer for 2 Receivers / 8 Transmitters Montage."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("fNIRS Cortical Monitor (2 Receivers / 8 Transmitters Setup)")
        self.resize(1360, 880)

        self.srate = 10.0
        self.window_sec = 20.0
        self.active_zone = "FRONTAL"
        self.num_channels = 16
        self.mode_mbll = True
        self.filter_enabled = True

        self.inlet = None
        self.stream_layout = None
        self.last_sample_time = None
        self.needs_redraw = False
        self.weak_since = [None] * 8
        self.reset_buffers()
        self.mbll_matrix = compute_mbll_matrix(MONTAGES[self.active_zone]["distance_cm"])

        self.fixed_scale = DEFAULT_SCALE_UM
        self.is_auto_scale = False

        self.init_filters()
        self.init_dark_theme()
        self.setup_ui()

        # Auto-connect to LSL stream on launch
        QtCore.QTimer.singleShot(600, lambda: self.connect_lsl(silent=True))

        # Update Timer (15 Hz: hemodynamics are slow and a filtered 8-channel frame takes ~30 ms)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll_and_update)
        self.timer.start(66)

    def reset_buffers(self):
        self.max_samples = int(self.srate * 60.0)
        self.raw_buffer = np.zeros((self.max_samples, self.num_channels))
        self.baseline = np.zeros(self.num_channels)
        self.sample_count = 0
        self.is_calibrated = False

    def apply_y_range(self, plot):
        """Fixed ± range for concentrations; raw signals sit far from zero, so they always auto-range."""
        if self.is_auto_scale or not self.mode_mbll:
            plot.enableAutoRange(axis='y', enable=True)
        else:
            plot.enableAutoRange(axis='y', enable=False)
            plot.setYRange(-self.fixed_scale, self.fixed_scale, padding=0.02)

    def on_spin_scale_changed(self, val):
        self.is_auto_scale = False
        self.fixed_scale = max(float(val), 0.0001)
        for p in self.plots:
            self.apply_y_range(p)

    def zoom_in(self):
        cur = self.spin_scale.value()
        new_val = max(cur * 0.5, 0.0001)
        self.spin_scale.setValue(new_val)

    def zoom_out(self):
        cur = self.spin_scale.value()
        new_val = min(cur * 2.0, 100.0)
        self.spin_scale.setValue(new_val)

    def on_preset_changed(self, index):
        presets = {
            0: 0.010,
            1: 0.020,
            2: 0.050,
            3: 0.100,
            4: 0.500,
            5: 1.0,
            6: 2.0,
            7: 5.0,
            8: None
        }
        val = presets.get(index, DEFAULT_SCALE_UM)
        if val is None:
            self.is_auto_scale = True
            for p in self.plots:
                self.apply_y_range(p)
        else:
            self.spin_scale.setValue(val)
            self.on_spin_scale_changed(val)  # setValue is silent when the value is unchanged

    def init_filters(self):
        self.sos_hrf = None
        self.sos_cardiac = None
        if not HAS_SCIPY:
            return
        nyq = 0.5 * self.srate
        self.sos_hrf = signal.butter(3, [0.01 / nyq, min(0.20 / nyq, 0.49)], btype='bandpass', output='sos')
        if nyq > 2.0:  # The cardiac band is unmeasurable at lower rates
            self.sos_cardiac = signal.butter(2, [0.8 / nyq, min(2.0 / nyq, 0.95)], btype='bandpass', output='sos')

    def init_dark_theme(self):
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyle("Fusion")
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0d1117"))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#161b22"))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#21262d"))
            palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#21262d"))
            palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#c9d1d9"))
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#1f6beb"))
            app.setPalette(palette)
        self.setStyleSheet(APP_STYLE)

    def setup_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Toolbar (two rows, so it never forces the window wider than the screen)
        toolbar_card = QtWidgets.QFrame(objectName="card")
        toolbar_rows = QtWidgets.QVBoxLayout(toolbar_card)
        toolbar_rows.setContentsMargins(12, 8, 12, 8)
        toolbar_layout = QtWidgets.QHBoxLayout()
        controls_layout = QtWidgets.QHBoxLayout()
        toolbar_rows.addLayout(toolbar_layout)
        toolbar_rows.addLayout(controls_layout)

        toolbar_layout.addWidget(QtWidgets.QLabel("🧠 2-Receiver / 8-Transmitter fNIRS Monitor", objectName="title"))
        toolbar_layout.addSpacing(15)
        toolbar_layout.addWidget(QtWidgets.QLabel("Select Head Placement:", objectName="strong"))

        self.combo_zone = QtWidgets.QComboBox()
        self.combo_zone.addItems([MONTAGES[zone]["combo"] for zone in ZONES])
        self.combo_zone.currentIndexChanged.connect(self.on_zone_changed)
        toolbar_layout.addWidget(self.combo_zone)
        toolbar_layout.addStretch()

        self.btn_lsl = QtWidgets.QPushButton("🔗 Connect LSL Stream", objectName="connect")
        self.btn_lsl.clicked.connect(lambda: self.connect_lsl())
        toolbar_layout.addWidget(self.btn_lsl)

        self.btn_zero = QtWidgets.QPushButton("🎯 Zero Baseline (I₀)", objectName="zero")
        self.btn_zero.clicked.connect(self.zero_baseline)
        toolbar_layout.addWidget(self.btn_zero)

        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.addItems([
            "View: 2 Receiver Pods (Right R1 & Left R2)",
            "View: 8 Individual Channels (C1 - C8)"
        ])
        self.combo_view.currentIndexChanged.connect(self.on_view_mode_changed)
        controls_layout.addWidget(self.combo_view)
        controls_layout.addSpacing(10)

        # Interactive Amplitude Scale Controls
        controls_layout.addWidget(QtWidgets.QLabel("Amplitude (±):", objectName="strong"))

        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(0.0001, 100.0)
        self.spin_scale.setValue(DEFAULT_SCALE_UM)
        self.spin_scale.setSingleStep(0.0050)
        self.spin_scale.setDecimals(4)
        self.spin_scale.valueChanged.connect(self.on_spin_scale_changed)
        controls_layout.addWidget(self.spin_scale)

        self.btn_zoom_in = QtWidgets.QPushButton("➕")
        self.btn_zoom_in.setToolTip("Zoom In (Increase Amplitude Sensitivity)")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        controls_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QtWidgets.QPushButton("➖")
        self.btn_zoom_out.setToolTip("Zoom Out (Decrease Amplitude Sensitivity)")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        controls_layout.addWidget(self.btn_zoom_out)

        self.combo_presets = QtWidgets.QComboBox()
        self.combo_presets.addItems([
            "Preset: ±0.010",
            "Preset: ±0.020",
            "Preset: ±0.050",
            "Preset: ±0.100",
            "Preset: ±0.500",
            "Preset: ±1.0 µM (Default)",
            "Preset: ±2.0 µM",
            "Preset: ±5.0 µM",
            "Preset: Auto-Scale"
        ])
        self.combo_presets.setCurrentIndex(5)  # Default: ±1.0 µM
        self.combo_presets.currentIndexChanged.connect(self.on_preset_changed)
        controls_layout.addWidget(self.combo_presets)

        controls_layout.addStretch()

        self.chk_mbll = QtWidgets.QCheckBox("MBLL (HbO / HbR / HbT)")
        self.chk_mbll.setChecked(True)
        self.chk_mbll.stateChanged.connect(self.on_mbll_toggled)
        controls_layout.addWidget(self.chk_mbll)

        self.chk_filter = QtWidgets.QCheckBox("Filter (0.01 - 0.20 Hz)")
        self.chk_filter.setChecked(True)
        self.chk_filter.stateChanged.connect(self.on_filter_toggled)
        controls_layout.addWidget(self.chk_filter)

        main_layout.addWidget(toolbar_card)

        # Split Layout
        content_layout = QtWidgets.QHBoxLayout()

        # Left Sidebar
        sidebar_card = QtWidgets.QFrame(objectName="card")
        sidebar_card.setFixedWidth(340)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_card)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)

        # Optode Configuration Legend
        legend_box = QtWidgets.QGroupBox("Hardware Socket Placement Guide")
        leg_layout = QtWidgets.QVBoxLayout(legend_box)

        self.lbl_guide = QtWidgets.QLabel(objectName="guide")
        self.lbl_traces = QtWidgets.QLabel()
        lbl_sci = QtWidgets.QLabel(
            "Contact: <span style='color:#3fb950'>good</span> · <span style='color:#d29922'>moderate</span> · "
            "<span style='color:#f85149'>loose</span> · <span style='color:#6e7681'>not measured</span>"
        )
        lbl_sci.setStyleSheet("font-size: 10px;")

        leg_layout.addWidget(self.lbl_guide)
        leg_layout.addWidget(self.lbl_traces)
        leg_layout.addWidget(lbl_sci)
        sidebar_layout.addWidget(legend_box)

        # Head Map
        self.headmap_widget = OptodeHeadMapWidget(active_zone=self.active_zone)
        self.headmap_widget.powers_changed.connect(self.on_powers_changed)
        self.refresh_placement_guide()
        sidebar_layout.addWidget(self.headmap_widget, 1)

        self.lbl_power_hint = QtWidgets.QLabel(objectName="hint")
        self.lbl_power_hint.setWordWrap(True)
        self.lbl_power_hint.hide()
        sidebar_layout.addWidget(self.lbl_power_hint)

        # Telemetry
        metrics_box = QtWidgets.QGroupBox("Live Channel Telemetry")
        m_layout = QtWidgets.QVBoxLayout(metrics_box)
        m_layout.setSpacing(2)

        self.lbl_srate = QtWidgets.QLabel("Sampling Rate: -- Hz")
        self.lbl_hbo_peak = QtWidgets.QLabel("Peak ΔHbO: -- µM")
        self.lbl_hbo_peak.setStyleSheet("color: #ff4757; font-weight: bold;")
        self.lbl_sci_avg = QtWidgets.QLabel("Mean Optode Contact (SCI): --")
        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setWordWrap(True)
        self.set_status("Stream: Idle (Press Connect LSL)", "#e3b341")

        m_layout.addWidget(self.lbl_srate)
        m_layout.addWidget(self.lbl_hbo_peak)
        m_layout.addWidget(self.lbl_sci_avg)
        m_layout.addWidget(self.lbl_status)
        sidebar_layout.addWidget(metrics_box)

        content_layout.addWidget(sidebar_card)

        # Right Charts
        plot_card = QtWidgets.QFrame(objectName="card")
        plot_layout = QtWidgets.QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(6, 6, 6, 6)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('#0d1117')
        plot_layout.addWidget(self.plot_widget)

        content_layout.addWidget(plot_card, 1)
        main_layout.addLayout(content_layout, 1)

        self.view_mode = "PODS"  # "PODS" (2 plots) or "CHANNELS" (8 plots)
        self.rebuild_plots()

    def set_status(self, text, color):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def refresh_placement_guide(self):
        self.lbl_guide.setText(placement_guide_html(self.active_zone, self.headmap_widget.tx_powers))

    def on_powers_changed(self):
        self.refresh_placement_guide()
        self.weak_since = [None] * 8  # Re-evaluate the swapped optodes from scratch

    def on_view_mode_changed(self, index):
        self.view_mode = "PODS" if index == 0 else "CHANNELS"
        self.rebuild_plots()

    def zero_baseline(self):
        if self.sample_count >= 10:
            window_size = min(int(self.srate * 3.0), self.sample_count)
            self.baseline = np.mean(self.raw_buffer[-window_size:, :], axis=0)
            self.is_calibrated = True
            self.needs_redraw = True
            self.set_status("Stream: Baseline Calibrated (I₀ set)", "#3fb950")

    def on_zone_changed(self, index):
        self.active_zone = ZONES[index] if index < len(ZONES) else "FRONTAL"
        self.headmap_widget.set_zone(self.active_zone)
        self.mbll_matrix = compute_mbll_matrix(MONTAGES[self.active_zone]["distance_cm"])
        self.refresh_placement_guide()
        self.weak_since = [None] * 8
        self.rebuild_plots()

    def style_plot(self, p):
        p.setMouseEnabled(x=True, y=True)
        p.setXRange(-self.window_sec, 0, padding=0)
        self.apply_y_range(p)
        p.showGrid(x=True, y=True, alpha=0.15)
        for side in ('left', 'bottom'):
            p.getAxis(side).setPen(pg.mkPen('#8b949e', width=1))
            p.getAxis(side).setTextPen('#8b949e')
            p.getAxis(side).enableAutoSIPrefix(False)  # Otherwise µM gets rendered as "mµM"
        p.getAxis('left').setWidth(52)
        if self.mode_mbll:
            p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen('#6e7681', width=1)))

    def add_curves(self, p, width):
        if self.mode_mbll:
            return (
                p.plot(pen=pg.mkPen('#ff4757', width=width), name="HbO"),
                p.plot(pen=pg.mkPen('#1e90ff', width=width), name="HbR"),
                p.plot(pen=pg.mkPen('#2ed573', width=max(width - 0.8, 1), style=QtCore.Qt.DashLine), name="HbT"),
            )
        return (
            p.plot(pen=pg.mkPen('#79c0ff', width=width), name="760 nm"),
            p.plot(pen=pg.mkPen('#ffa657', width=width), name="850 nm"),
        )

    def signal_axis_label(self):
        if self.mode_mbll:
            return "Concentration (µM)"
        if self.stream_layout is not None and self.stream_layout.kind == KIND_OD:
            return "Optical Density"
        return "Optical Intensity (counts)"

    def rebuild_plots(self):
        self.plot_widget.clear()
        self.plots = []
        self.curves = []
        self.needs_redraw = True

        if self.mode_mbll:
            self.lbl_traces.setText("<b><span style='color:#ff4757'>ΔHbO</span> · <span style='color:#1e90ff'>ΔHbR</span> · "
                                    "<span style='color:#2ed573'>ΔHbT</span> (µM)</b>")
        else:
            self.lbl_traces.setText("<b><span style='color:#79c0ff'>760 nm</span> · <span style='color:#ffa657'>850 nm</span></b>")

        if self.view_mode == "PODS":
            # 2 Large Receiver Pod Plots: Right Pod (R1) and Left Pod (R2)
            pods = MONTAGES[self.active_zone]["pods"]
            pod_titles = [
                f"{pods[0]} Pod (Receiver R1: T1..T4 Transmitters)",
                f"{pods[1]} Pod (Receiver R2: T5..T8 Transmitters)"
            ]
            for pod_i in range(2):
                p = self.plot_widget.addPlot(row=pod_i, col=0)
                p.setTitle(f"<span style='color: #58a6ff; font-weight: bold; font-size: 13px;'>{pod_titles[pod_i]}</span>")
                self.style_plot(p)
                p.setLabel('left', self.signal_axis_label())
                legend = p.addLegend(offset=(10, 10))
                legend.setBrush(pg.mkBrush('#161b2299'))
                self.curves.append(self.add_curves(p, 2.4))
                self.plots.append(p)
        else:
            # 8 Separate Channels: columns match the head map, so the left pod (C5..C8) is on the left
            for i in range(8):
                p = self.plot_widget.addPlot(row=i % 4, col=1 - i // 4)
                ch_name = self.headmap_widget.get_channel_label(i)
                p.setTitle(f"<span style='color: #8b949e; font-size: 11px;'>{ch_name}</span>")
                self.style_plot(p)
                self.curves.append(self.add_curves(p, 1.6))
                if i % 4 < 3:
                    p.getAxis('bottom').setStyle(showValues=False)
                self.plots.append(p)

        for p in self.plots[1:]:
            p.setXLink(self.plots[0])
        if self.view_mode == "PODS":
            self.plots[0].getAxis('bottom').setStyle(showValues=False)
            self.plots[-1].setLabel('bottom', 'Time (s)')
        else:
            self.plots[3].setLabel('bottom', 'Time (s)')
            self.plots[7].setLabel('bottom', 'Time (s)')

    def on_mbll_toggled(self, state):
        self.mode_mbll = self.chk_mbll.isChecked()
        self.rebuild_plots()

    def on_filter_toggled(self, state):
        self.filter_enabled = self.chk_filter.isChecked()
        self.needs_redraw = True

    def connect_lsl(self, silent=False):
        if not HAS_LSL:
            if not silent:
                QtWidgets.QMessageBox.warning(self, "LSL Not Installed", "pylsl library is not installed.")
            return

        self.set_status("Stream: Searching LSL...", "#e3b341")
        QtWidgets.QApplication.processEvents()

        target_stream = next((s for s in resolve_streams(wait_time=2.0) if is_fnirs_stream(s.name(), s.type())), None)
        if target_stream is None:
            self.set_status("Stream: Idle (Press Connect LSL)", "#e3b341")
            if not silent:
                QtWidgets.QMessageBox.information(self, "LSL Stream", "No active fNIRS LSL stream found.\nPlease run 'uv run scripts/bridges/artinis_to_lsl.py'.")
            return

        self.inlet = StreamInlet(target_stream)
        info = self.inlet.info(timeout=2.0)  # Full info: the resolved one has no channel labels
        labels, units = [], []
        channel = info.desc().child("channels").child("channel")
        while not channel.empty():
            labels.append(channel.child_value("label"))
            units.append(channel.child_value("unit"))
            channel = channel.next_sibling("channel")

        self.attach_stream(parse_stream_layout(labels, units, info.channel_count()),
                           info.channel_count(), info.nominal_srate() or 25.0)
        self.set_status(f"Stream: Connected ({target_stream.name()})" +
                        ("" if self.stream_layout.labelled else " - no channel labels, assuming odd 760 nm / even 850 nm"),
                        "#3fb950")

    def attach_stream(self, layout, channel_count, srate):
        """Prepare buffers, filters and plots for a stream with this layout."""
        self.stream_layout = layout
        self.num_channels = channel_count
        self.srate = srate
        self.init_filters()
        self.reset_buffers()
        self.weak_since = [None] * 8
        self.last_sample_time = time.monotonic()
        self.lbl_srate.setText(f"Sampling Rate: {self.srate:.1f} Hz")

        # A concentration stream has no wavelengths to show as raw traces
        self.chk_mbll.setEnabled(layout.kind != KIND_CONC)
        if layout.kind == KIND_CONC:
            self.chk_mbll.setChecked(True)
        self.rebuild_plots()

    def push_samples(self, samples):
        """Append a (time, column) chunk to the rolling buffer."""
        samples = samples[-self.max_samples:]
        n = samples.shape[0]
        self.raw_buffer = np.roll(self.raw_buffer, -n, axis=0)
        self.raw_buffer[-n:, :] = samples
        self.sample_count = min(self.sample_count + n, self.max_samples)
        self.needs_redraw = True

        if not self.is_calibrated and self.sample_count >= 15:
            self.baseline = np.mean(self.raw_buffer[-15:, :], axis=0)
            self.is_calibrated = True
            if self.stream_layout.kind is None:
                self.stream_layout = self.stream_layout._replace(kind=guess_signal_kind(self.raw_buffer[-15:, :]))
                self.rebuild_plots()

    def pull_lsl(self):
        try:
            samples, _ = self.inlet.pull_chunk(timeout=0.0)
        except Exception as exc:
            self.inlet = None
            self.set_status(f"Stream: Lost ({exc}). Press Connect LSL", "#f85149")
            return
        now = time.monotonic()
        if samples:
            self.push_samples(np.asarray(samples, dtype=np.float64))
            if now - self.last_sample_time > STALE_STREAM_SEC:
                self.set_status("Stream: Connected (data resumed)", "#3fb950")
            self.last_sample_time = now
        elif now - self.last_sample_time > STALE_STREAM_SEC:
            self.set_status(f"Stream: No data for {now - self.last_sample_time:.0f} s", "#f85149")

    def poll_and_update(self):
        if HAS_LSL and self.inlet:
            self.pull_lsl()
        if self.needs_redraw and self.is_calibrated and self.stream_layout is not None and self.stream_layout.first:
            self.needs_redraw = False
            self.redraw()

    def redraw(self):
        window_size = min(int(self.srate * self.window_sec), self.sample_count)
        raw_segment = self.raw_buffer[-window_size:, :]
        time_segment = (np.arange(window_size) - (window_size - 1)) / self.srate

        layout = self.stream_layout
        first = raw_segment[:, layout.first]
        second = raw_segment[:, layout.second]
        hbo, hbr = to_concentrations(
            first, second, self.baseline[layout.first], self.baseline[layout.second], layout.kind,
            self.mbll_matrix, sos=self.sos_hrf if self.filter_enabled else None,
        )
        # HbO and HbR are unrelated in the cardiac band, so SCI needs the wavelength signals
        if layout.kind == KIND_CONC:
            sci = np.full(len(layout.first), np.nan)
        else:
            sci = scalp_coupling_index(first, second, self.sos_cardiac, int(self.srate * 5.0))

        traces = (hbo, hbr, hbo + hbr) if self.mode_mbll else (first, second)
        n_channels = len(layout.first)
        if self.view_mode == "PODS":
            groups = [range(0, min(4, n_channels)), range(4, n_channels)]
        else:
            groups = [range(i, i + 1) if i < n_channels else range(0) for i in range(8)]
        for curve_set, group in zip(self.curves, groups):
            for curve, trace in zip(curve_set, traces):
                if len(group):
                    curve.setData(time_segment, trace[:, list(group)].mean(axis=1))
                else:
                    curve.setData([], [])

        latest_hbo = hbo[-1, :]
        self.headmap_widget.update_metrics(latest_hbo, sci)
        self.lbl_hbo_peak.setText(f"Peak ΔHbO: {np.max(np.abs(latest_hbo)):.2f} µM")
        if np.isfinite(sci).any():
            avg_sci = float(np.nanmean(sci))
            self.lbl_sci_avg.setText(f"Mean Optode Contact (SCI): {avg_sci:.2f}")
            self.lbl_sci_avg.setStyleSheet("color: #3fb950;" if avg_sci >= 0.65 else "color: #d29922;" if avg_sci >= WEAK_SCI else "color: #f85149;")
        else:
            self.lbl_sci_avg.setText("Mean Optode Contact (SCI): not measured")
            self.lbl_sci_avg.setStyleSheet("color: #8b949e;")
        self.update_power_hint(sci)

    def update_power_hint(self, sci):
        """Suggest a physical optode swap when a channel stays weak. Optode type can't be changed from software."""
        now = time.monotonic()
        weak_lp, weak_hp = [], []
        for ch, value in enumerate(sci):
            # SCI wobbles on real data, so only a clearly good reading clears a weak channel.
            # NaN also clears it: an unmeasured channel isn't a weak one.
            if np.isnan(value) or value >= GOOD_SCI:
                self.weak_since[ch] = None
                continue
            if self.weak_since[ch] is None and value < WEAK_SCI:
                self.weak_since[ch] = now
            if self.weak_since[ch] is not None and now - self.weak_since[ch] >= WEAK_HOLD_SEC:
                (weak_lp if self.headmap_widget.tx_powers[ch] == "LP" else weak_hp).append(f"T{ch + 1}")
        lines = []
        if weak_lp:
            lines.append(f"Weak on LP: {', '.join(weak_lp)}. Check skin contact and hair, then try an HP optode.")
        if weak_hp:
            lines.append(f"Weak on HP: {', '.join(weak_hp)}. Check skin contact and hair.")
        self.lbl_power_hint.setText("\n".join(lines))
        self.lbl_power_hint.setVisible(bool(lines))


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = fNIRSVisualizerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
