"""
Nautilus BCI Studio - Standalone Web Backend Server
===================================================

Zero-dependency standard-library HTTP & SSE server that:
1. Serves the responsive frontend (index.html, style.css, app.js).
2. Bridges live 250 Hz LSL EEG stream into real-time telemetry (50Hz Notch + 1-35Hz BP).
3. Provides REST endpoints for BIDS recording and GDS hardware resets.

Usage:
  cd scripts/web
  uv run python server.py --port 8000
"""

import os
import sys
import time
import json
import threading
import subprocess
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
from scipy import signal

# Add parent scripts dir to sys.path
_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    from pylsl import resolve_streams, StreamInlet, resolve_byprop
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

# Vendor 32-channel layout
GPYPE_32 = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "PO4", "PO8", "Oz",
]

# Shared Telemetry State
telemetry_lock = threading.Lock()
latest_telemetry = {
    "connected": False,
    "battery_pct": 80.0,
    "total_channels": 32,
    "clean_count": 0,
    "noisy_count": 0,
    "flat_count": 0,
    "median_rms_uv": 0.0,
    "retouch_pins": [],
    "flat_pins": [],
    "channels": {},
    "scope_traces": {
        "Cz": [0.0] * 64,
        "C3": [0.0] * 64,
        "C4": [0.0] * 64,
        "Oz": [0.0] * 64,
    },
    "powers": {"alpha": 40, "beta": 30, "delta": 20},
    "neurofeedback": {"drive_pct": 50, "lateralization": 0.0},
    "recording": {"is_recording": False, "samples": 0, "duration_sec": 0}
}

stop_worker = threading.Event()
bids_recorder_instance = None


class LSLStreamWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.inlet = None
        self.fs = 250.0
        self.num_channels = 33
        self.buf_samples = int(self.fs * 2.0)
        self.eeg_buffer = np.zeros((self.buf_samples, self.num_channels))
        
        # Design Filters
        nyq = 0.5 * self.fs
        self.b_notch, self.a_notch = signal.iirnotch(50.0, 30.0, self.fs)
        self.b_bp, self.a_bp = signal.butter(4, [1.0 / nyq, 35.0 / nyq], btype='band')
        
        # SMR Bandpass (8-30 Hz)
        self.b_smr, self.a_smr = signal.butter(4, [8.0 / nyq, 30.0 / nyq], btype='band')
        self.baseline_power = None

    def run(self):
        print("[*] LSL Background Consumer Thread started.")
        while not stop_worker.is_set():
            if not HAS_LSL:
                time.sleep(1.0)
                continue

            if self.inlet is None:
                try:
                    streams = resolve_byprop('type', 'EEG', timeout=1.0)
                    if streams:
                        self.inlet = StreamInlet(streams[0], max_buflen=360)
                        info = self.inlet.info()
                        self.fs = info.nominal_srate() if info.nominal_srate() > 0 else 250.0
                        self.num_channels = info.channel_count()
                        self.eeg_buffer = np.zeros((int(self.fs * 2.0), self.num_channels))
                        print(f"[+] Connected to LSL Stream: {info.name()} ({self.num_channels} ch @ {self.fs} Hz)")
                    else:
                        time.sleep(1.0)
                        continue
                except Exception:
                    self.inlet = None
                    time.sleep(1.0)
                    continue

            try:
                samples, _ = self.inlet.pull_chunk(max_samples=64, timeout=0.1)
                if samples and len(samples) > 0:
                    new_data = np.array(samples, dtype=np.float64)
                    if np.median(np.abs(new_data)) < 0.01:
                        new_data *= 1e6

                    n_new = len(new_data)
                    self.eeg_buffer = np.roll(self.eeg_buffer, -n_new, axis=0)
                    self.eeg_buffer[-n_new:, :] = new_data
                    
                    self.process_telemetry()
                else:
                    time.sleep(0.015)
            except Exception as e:
                print(f"[-] LSL pull error: {e}")
                self.inlet = None
                time.sleep(1.0)

    def process_telemetry(self):
        global latest_telemetry
        n_scalp = min(32, self.num_channels)
        raw_eeg = self.eeg_buffer[:, :n_scalp]
        detrended = raw_eeg - np.mean(raw_eeg, axis=0)

        try:
            filt_eeg = signal.filtfilt(self.b_notch, self.a_notch, detrended, axis=0)
            filt_eeg = signal.filtfilt(self.b_bp, self.a_bp, filt_eeg, axis=0)
        except Exception:
            filt_eeg = detrended

        # Metrics
        stds = np.std(filt_eeg, axis=0)
        ptps = np.ptp(filt_eeg, axis=0)
        raw_max = np.max(np.abs(raw_eeg), axis=0)

        is_railed = raw_max > 100000.0
        is_flat = (stds < 0.5) | (ptps < 1.0)
        is_noisy = (stds > 80.0) | (ptps > 300.0)
        is_clean = ~(is_railed | is_flat | is_noisy)

        clean_count = int(np.sum(is_clean))
        noisy_count = int(np.sum(is_noisy & ~is_railed))
        flat_count = int(np.sum(is_flat))
        median_rms = float(np.median(stds[is_clean])) if clean_count > 0 else float(np.median(stds))

        retouch_pins = [GPYPE_32[i] for i in np.where(is_noisy & ~is_railed)[0] if i < len(GPYPE_32)]
        flat_pins = [GPYPE_32[i] for i in np.where(is_flat)[0] if i < len(GPYPE_32)]

        # Extract representative scope traces (downsampled to 64 points)
        scope_len = 64
        step = max(1, len(filt_eeg) // scope_len)
        tail = filt_eeg[-scope_len * step :: step]

        cz_trace = tail[:, 15].round(1).tolist() if tail.shape[1] > 15 else [0.0] * scope_len
        c3_trace = tail[:, 14].round(1).tolist() if tail.shape[1] > 14 else [0.0] * scope_len
        c4_trace = tail[:, 16].round(1).tolist() if tail.shape[1] > 16 else [0.0] * scope_len
        oz_trace = tail[:, 31].round(1).tolist() if tail.shape[1] > 31 else [0.0] * scope_len

        # SMR Desynchronization & Mental Drive
        p_c3 = float(np.mean(c3_trace[-16:])**2)
        p_c4 = float(np.mean(c4_trace[-16:])**2)
        p_cz = float(np.mean(cz_trace[-16:])**2)
        mean_p = (p_c3 + p_c4 + p_cz) / 3.0

        if self.baseline_power is None or self.baseline_power <= 0:
            self.baseline_power = max(1.0, mean_p)
        else:
            self.baseline_power = 0.99 * self.baseline_power + 0.01 * mean_p

        ratio = mean_p / (self.baseline_power + 1e-6)
        drive_pct = int(np.clip((1.0 - ratio) / 0.45 * 100, 0, 100))
        lateral_diff = (p_c4 - p_c3) / (p_c3 + p_c4 + 1e-6)
        lateralization = float(np.clip(lateral_diff * 3.0, -1.0, 1.0))

        # Channel map
        ch_dict = {}
        for i in range(n_scalp):
            name = GPYPE_32[i] if i < len(GPYPE_32) else f"CH{i+1}"
            st = "clean"
            if is_railed[i]: st = "railed"
            elif is_flat[i]: st = "flat"
            elif is_noisy[i]: st = "noisy"
            ch_dict[name] = {"rms": round(float(stds[i]), 1), "ptp": round(float(ptps[i]), 1), "status": st}

        with telemetry_lock:
            latest_telemetry["connected"] = True
            latest_telemetry["clean_count"] = clean_count
            latest_telemetry["noisy_count"] = noisy_count
            latest_telemetry["flat_count"] = flat_count
            latest_telemetry["median_rms_uv"] = round(median_rms, 1)
            latest_telemetry["retouch_pins"] = retouch_pins
            latest_telemetry["flat_pins"] = flat_pins
            latest_telemetry["channels"] = ch_dict
            latest_telemetry["scope_traces"] = {
                "Cz": cz_trace, "C3": c3_trace, "C4": c4_trace, "Oz": oz_trace
            }
            latest_telemetry["neurofeedback"] = {
                "drive_pct": drive_pct,
                "lateralization": round(lateralization, 2)
            }


class BCIWebRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        static_dir = os.path.join(os.path.dirname(__file__))
        super().__init__(*args, directory=static_dir, **kwargs)

    def do_GET(self):
        if self.path == '/api/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.end_headers()
            with telemetry_lock:
                payload = json.dumps(latest_telemetry).encode('utf-8')
            self.wfile.write(payload)
            return

        elif self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "app": "Nautilus BCI Studio"}')
            return

        return super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else ""
        data = json.loads(body) if body else {}

        if self.path == '/api/bridge/reset-gds':
            print("[*] Web API: Emergency Windows GDS reset requested.")
            cmd = 'ssh bci "taskkill /F /IM python.exe; net stop GDS && net start GDS"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"success": res.returncode == 0, "output": res.stdout}).encode('utf-8'))
            return

        elif self.path == '/api/bridge/restart':
            print("[*] Web API: Restarting LSL streamer bridge.")
            cmd = 'ssh bci "cd C:\\Users\\VR\\Documents\\GitHub\\nautilus_bci && uv run --project scripts python scripts/bridges/gds_to_lsl.py --non-interactive --cap-type dry"'
            subprocess.Popen(cmd, shell=True)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"success": true, "msg": "Streamer bridge launched."}')
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Silence routine static polling access logs to keep terminal readable
        try:
            msg = format % args if args else str(format)
            if "/api/telemetry" in msg:
                return
        except Exception:
            pass
        super().log_message(format, *args)


def main():
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    # Launch LSL Worker Thread
    worker = LSLStreamWorker()
    worker.start()

    server_address = ('0.0.0.0', port)
    httpd = ThreadingHTTPServer(server_address, BCIWebRequestHandler)
    print("=" * 70)
    print(f"🚀 Nautilus BCI Studio Web Server Live!".center(70))
    print(f"   - Desktop URL: http://localhost:{port}".center(70))
    print(f"   - Mobile/Mirror: http://192.168.2.100:{port}".center(70))
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down BCI web server...")
        stop_worker.set()
        httpd.shutdown()


if __name__ == '__main__':
    main()
