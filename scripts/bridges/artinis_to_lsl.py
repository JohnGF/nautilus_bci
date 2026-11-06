"""
Artinis OctaMon / OptoMon fNIRS to Lab Streaming Layer (LSL) Bridge
Uses official OctaPortaSDK.dll (g.tec / Artinis C SDK) for direct hardware acquisition without MATLAB.
"""

import os
import sys
import time
import argparse
import ctypes
import json
import subprocess
import threading
import numpy as np
from pylsl import ContinuousResolver, StreamInfo, StreamInlet, StreamOutlet, local_clock

SDK_DIR = r"C:\Program Files\gtec\gHIsys\gSENSORfNIRS"
SDK_DLL = os.path.join(SDK_DIR, "OctaPortaSDK.dll")

# Decadic molar extinction (1/(cm*M)) at [760, 850] nm for [HbO, HbR], used by the mock stream
EXTINCTION = np.array([[1486.586, 3843.707], [2526.391, 1798.643]])
# Sent by fnirs_visualizer.py. The name avoids "fNIRS" so nothing mistakes it for the data stream.
CONTROL_STREAM_NAME = "OptodeBaselineControl"
REBASELINE_COMMAND = "rebaseline"
BT_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bt_preflight.ps1")


def bt_preflight(address, scan_seconds=12, no_pair=False):
    """Check the Bluetooth radio, locate/pair the Artinis device and wake the link."""
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", BT_HELPER,
           "-Address", address, "-ScanSeconds", str(scan_seconds)]
    if no_pair:
        cmd.append("-NoPair")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=scan_seconds + 30).stdout.strip()
        res = json.loads(out.splitlines()[-1])
    except Exception as e:
        print(f"[!] Bluetooth preflight could not run: {e}", flush=True)
        return None
    print(f"[*] Bluetooth: radio={res['radio']} found={res['found']} paired={res['paired']} "
          f"connected={res['connected']} actions={res['action']}", flush=True)
    if not res["ok"]:
        print(f"[-] Bluetooth preflight failed: {res['error']}", flush=True)
    return res

class BaselineSettler:
    """Holds back the frames before the SDK baseline takes effect.

    On real hardware the first ~0.5 s reads about 1.2-1.8 OD on every channel, then drops to
    near 0. Those frames are perfectly flat, so stability can't identify them; the level can.
    A channel that genuinely stays high (a bad optode) must not block the stream, hence the timeout.
    """

    MAX_OD = 0.5
    FRAMES_NEEDED = 10
    TIMEOUT_SEC = 5.0

    def __init__(self):
        self.settled = False
        self.timed_out = False
        self.dropped = 0
        self.first_frame_time = None
        self.good_frames = 0
        self.peak_levels = None

    @property
    def startup_levels(self):
        """Absolute OD per column from the held-back frames. NaN where a column never showed its level, None if none did.

        The first frame alone isn't enough: on real hardware R1 can still read about 1.3 OD while R2 is already at 0.
        """
        if self.peak_levels is None or np.max(self.peak_levels) < self.MAX_OD:
            return None
        return np.where(self.peak_levels >= self.MAX_OD, self.peak_levels, np.nan)

    def accept(self, frame, now):
        """True once frames are safe to publish, including this one."""
        if self.settled:
            return True
        level = np.abs(frame)
        self.peak_levels = level if self.peak_levels is None else np.maximum(self.peak_levels, level)
        if self.first_frame_time is None:
            self.first_frame_time = now
        self.good_frames = self.good_frames + 1 if np.max(np.abs(frame)) < self.MAX_OD else 0
        if self.good_frames >= self.FRAMES_NEEDED:
            self.settled = True
        elif now - self.first_frame_time >= self.TIMEOUT_SEC:
            self.settled = self.timed_out = True
        else:
            self.dropped += 1
        return self.settled


class ControlListener(threading.Thread):
    """Listens for commands from any viewer on the network. A thread, so finding a viewer can't stall device reads."""

    SCAN_PERIOD_SEC = 0.2

    def __init__(self):
        super().__init__(daemon=True, name="fNIRS control")
        self.rebaseline = threading.Event()
        self.stopping = threading.Event()

    def run(self):
        try:
            resolver = ContinuousResolver(prop="name", value=CONTROL_STREAM_NAME)
            inlets = {}
            while not self.stopping.wait(self.SCAN_PERIOD_SEC):
                for info in resolver.results():
                    if info.uid() not in inlets:
                        try:
                            inlet = StreamInlet(info, recover=False)
                            inlet.open_stream(timeout=2.0)
                            inlets[info.uid()] = inlet
                        except Exception:
                            pass  # A viewer that just closed stays listed for a few seconds
                for uid, inlet in list(inlets.items()):
                    try:
                        sample, _ = inlet.pull_sample(timeout=0.0)
                        while sample:
                            if sample[0] == REBASELINE_COMMAND:
                                self.rebaseline.set()
                            sample, _ = inlet.pull_sample(timeout=0.0)
                    except Exception:
                        inlets.pop(uid)  # That viewer closed
        except Exception as exc:
            print(f"[!] Control listener stopped, so Re-baseline won't work until a restart: {exc!r}", flush=True)


def startup_light_text(levels):
    """Worst wavelength per transmitter, with -- where the level is unknown."""
    return " ".join("--" if np.isnan(pair).all() else f"{np.nanmax(pair):.2f}" for pair in levels.reshape(-1, 2))


class ArtinisOctaPortaLSL:
    def __init__(self, bt_address="682719F8B937", sample_rate=50, num_channels=16, dpf=6.0, optode_dist_m=0.030, mode="od", mock=False, preflight=True, no_pair=False, retries=3):
        self.bt_address = bt_address
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.dpf = dpf
        self.optode_dist_m = optode_dist_m
        self.mode = mode
        self.mock = mock
        self.control = None
        self.preflight = preflight
        self.no_pair = no_pair
        self.retries = retries
        self.running = False
        self.sdk = None
        self.outlet = None

    def setup_lsl(self, startup_levels=None):
        """Create the outlet. startup_levels (absolute OD per column) goes into each channel's metadata."""
        if self.outlet is not None:
            return

        info = StreamInfo(
            name='Artinis_fNIRS',
            type='fNIRS',
            channel_count=self.num_channels,
            nominal_srate=float(self.sample_rate),
            channel_format='float32',
            source_id=f'OctaMon_{"SIM" if self.mock else self.bt_address}'
        )
        desc = info.desc().append_child("channels")
        # SDK output is interleaved per transmitter: [T1 a, T1 b, T2 a, T2 b, ...].
        # T1-T4 belong to R1 (right), T5-T8 to R2 (left). Per the g.SENSOR 8 fNIRS manual (v1.20.01):
        # odd channels = 760 nm (OD) / HbO (conc), even channels = 850 nm (OD) / HbR (conc).
        names = ("HbO", "HbR") if self.mode == "conc" else ("760nm", "850nm")
        for tx in range(1, 9):
            rx = 1 if tx <= 4 else 2
            for wl in range(2):
                ch = desc.append_child("channel")
                ch.append_child_value("label", f"R{rx}_T{tx}_{names[wl]}")
                ch.append_child_value("type", names[wl] if self.mode == "conc" else "fNIRS")
                ch.append_child_value("unit", "micromol/L" if self.mode == "conc" else "OD")
                ch.append_child_value("receiver", f"R{rx}")
                ch.append_child_value("transmitter", f"T{tx}")
                ch.append_child_value("side", "right" if rx == 1 else "left")
                if startup_levels is not None:
                    ch.append_child_value("startup_od", f"{startup_levels[2 * (tx - 1) + wl]:.3f}")

        self.outlet = StreamOutlet(info)
        mode_str = f"MODE: {self.mode.upper()} (BT: {self.bt_address})"
        print(f"\n[+] LSL Stream Outlet Active: 'Artinis_fNIRS' ({self.num_channels} ch @ {self.sample_rate} Hz) [{mode_str}]", flush=True)

    def init_sdk(self):
        if not os.path.exists(SDK_DLL):
            raise FileNotFoundError(f"OctaPortaSDK.dll not found at: {SDK_DLL}")

        os.add_dll_directory(SDK_DIR)
        self.sdk = ctypes.CDLL(SDK_DLL)

        self.sdk.artinis_start_portable_acquisition.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.sdk.artinis_start_portable_acquisition.restype = ctypes.c_int

        self.sdk.artinis_stop_portable_acquisition.argtypes = []
        self.sdk.artinis_stop_portable_acquisition.restype = ctypes.c_int

        self.sdk.artinis_set_optode_distance.argtypes = [ctypes.c_int, ctypes.c_double]
        self.sdk.artinis_set_optode_distance.restype = ctypes.c_int

        self.sdk.artinis_set_dpf.argtypes = [ctypes.c_double]
        self.sdk.artinis_set_dpf.restype = ctypes.c_int

        self.sdk.artinis_set_calc_optical_densities_only.argtypes = [ctypes.c_int]
        self.sdk.artinis_set_calc_optical_densities_only.restype = ctypes.c_int

        self.sdk.artinis_set_baseline.argtypes = []
        self.sdk.artinis_set_baseline.restype = ctypes.c_int

        self.sdk.artinis_get_battery_status.argtypes = [ctypes.c_int]
        self.sdk.artinis_get_battery_status.restype = ctypes.c_int

    def mock_frame(self, t):
        """One simulated frame in the live layout: [T1 760nm/HbO, T1 850nm/HbR, T2 ...]."""
        cardiac = 0.5 * np.sin(2 * np.pi * 1.15 * t)
        resp = 0.3 * np.sin(2 * np.pi * 0.25 * t)
        mayer = 0.8 * np.sin(2 * np.pi * 0.10 * t)
        is_active = (int(t) % 16) < 8
        activation = 1.0 / (1.0 + np.exp(-1.5 * ((t % 16) - 3))) if is_active else 0.0
        hbo = 3.5 * activation + mayer + resp * 0.6 + cardiac + np.random.normal(0, 0.05, 8)
        hbr = -1.2 * activation - 0.3 * mayer - 0.2 * resp + cardiac * 0.3 + np.random.normal(0, 0.03, 8)

        if self.mode == "od":
            pathlength_cm = self.optode_dist_m * 100.0 * self.dpf
            first, second = EXTINCTION @ np.vstack([hbo, hbr]) * 1e-6 * pathlength_cm
        else:
            first, second = hbo, hbr
        frame = np.zeros(self.num_channels, dtype=np.float32)
        frame[0::2], frame[1::2] = first, second
        return frame

    def start_mock(self):
        self.running = True
        self.setup_lsl()
        self.control = ControlListener()
        self.control.start()
        mock_baseline = np.zeros(self.num_channels, dtype=np.float32)
        print("[*] Streaming realistic simulated fNIRS hemodynamics... (Ctrl+C to stop)", flush=True)
        dt = 1.0 / self.sample_rate
        t_start = time.time()
        last_stat = time.time()

        while self.running:
            t = time.time() - t_start
            sample = self.mock_frame(t)
            if self.control.rebaseline.is_set():
                self.control.rebaseline.clear()
                mock_baseline = sample.copy()
                print("[+] Baseline re-set from a viewer.", flush=True)
            sample = sample - mock_baseline

            self.outlet.push_sample(sample.tolist(), local_clock())

            if time.time() - last_stat > 1.0:
                print(f"[*] [MOCK fNIRS] Active @ {self.sample_rate}Hz | R1T1: {sample[0]:+.3f} | R2T5: {sample[8]:+.3f}", flush=True)
                last_stat = time.time()

            time.sleep(dt)

    def start(self):
        if self.mock:
            self.start_mock()
            return

        print(f"[*] Initializing Artinis OctaPortaSDK for device BT: {self.bt_address}...", flush=True)
        pre = bt_preflight(self.bt_address, no_pair=self.no_pair) if self.preflight else None
        if pre is not None and not pre["ok"]:
            return
        self.init_sdk()

        p_str = ctypes.c_char_p(self.bt_address.encode('ascii'))
        print(f"[*] Connecting to Artinis device ({self.bt_address})...", flush=True)
        res = -1
        for attempt in range(1, self.retries + 1):
            res = self.sdk.artinis_start_portable_acquisition(ctypes.byref(p_str), 1, 2, self.sample_rate)
            if res == 0:
                break
            print(f"[-] Connect attempt {attempt}/{self.retries} failed (code: {res}); retrying...", flush=True)
            time.sleep(2.0)
        if res != 0:
            print(f"[-] Failed to connect to Artinis device (code: {res}).", flush=True)
            print("    Please ensure device is powered ON and paired in Windows Bluetooth.", flush=True)
            return

        print("[+] Connected successfully to Artinis fNIRS hardware!", flush=True)
        
        for ch in range(8):
            self.sdk.artinis_set_optode_distance(ch, self.optode_dist_m)
        self.sdk.artinis_set_dpf(self.dpf)
        
        calc_od_flag = 1 if self.mode == "od" else 0
        self.sdk.artinis_set_calc_optical_densities_only(calc_od_flag)

        print("[*] Stabilizing optode LEDs (1.0s)...", flush=True)
        time.sleep(1.0)
        self.sdk.artinis_set_baseline()
        print("[+] Optical baseline established!", flush=True)

        battery = self.sdk.artinis_get_battery_status(0)
        print(f"[?] SDK battery value: {battery} (unverified meaning; check device indicator)", flush=True)

        # In OD mode the outlet waits for the settler, because its metadata carries the startup levels
        if self.mode != "od":
            self.setup_lsl()
        self.running = True

        buf_od = (ctypes.c_double * 2000)()
        buf_conc = (ctypes.c_double * 2000)()
        buf_sq = (ctypes.c_double * 2000)()
        n_samples = ctypes.c_int(0)
        n_od = ctypes.c_int(0)
        n_conc = ctypes.c_int(0)
        n_sq = ctypes.c_int(0)

        last_stat = time.time()
        total_pushed = 0
        dropped_zero = 0
        dropped_dup = 0
        errors = 0
        last_new = time.time()
        warned_stale = False
        prev_frame = None
        settler = BaselineSettler() if self.mode == "od" else None
        current_sample = np.zeros(self.num_channels, dtype=np.float32)

        self.control = ControlListener()
        self.control.start()

        print("\n[*] Live fNIRS streaming active on LSL. (Press Ctrl+C to stop)\n", flush=True)

        dt = 1.0 / self.sample_rate

        while self.running:
            try:
                if self.control.rebaseline.is_set():
                    self.control.rebaseline.clear()
                    if settler is None or settler.settled:
                        # Measured live: no gap and no startup plateau, levels re-zero within about 0.2 s.
                        # Its return value is meaningless, so success can't be read from it.
                        self.sdk.artinis_set_baseline()
                        print("[+] Baseline re-set from a viewer. Startup light levels above are now out of date.", flush=True)

                self.sdk.artinis_get_concentrations_tsi(
                    ctypes.byref(n_samples),
                    buf_od,
                    ctypes.c_int(100),
                    ctypes.byref(n_od),
                    buf_conc,
                    ctypes.c_int(100),
                    buf_sq,
                    ctypes.byref(n_sq)
                )

                source = buf_conc if self.mode == "conc" else buf_od
                frame = np.ctypeslib.as_array(source)[:self.num_channels].astype(np.float32).copy()
                if not np.any(frame):
                    # SDK returns all zeros while settling / when nothing arrived; never publish those
                    dropped_zero += 1
                elif prev_frame is not None and np.array_equal(frame, prev_frame):
                    # Same buffer as last call: no new sample from the device
                    dropped_dup += 1
                else:
                    prev_frame = frame
                    last_new = time.time()
                    warned_stale = False
                    was_settled = settler is None or settler.settled
                    if settler is None or settler.accept(frame, last_new):
                        if not was_settled:
                            if settler.timed_out:
                                print(f"[!] A channel is still above {settler.MAX_OD} OD after {settler.TIMEOUT_SEC:.0f} s; publishing anyway. Check optode contact.", flush=True)
                            else:
                                print(f"[+] Baseline settled; held back {settler.dropped} frames.", flush=True)
                            self.setup_lsl(settler.startup_levels)
                            if settler.startup_levels is not None:
                                print("[+] Startup light level (OD, lower = more light), T1..T8:", startup_light_text(settler.startup_levels), flush=True)
                        self.outlet.push_sample(frame.tolist(), local_clock())
                        current_sample[:] = frame
                        total_pushed += 1

                now = time.time()
                if now - last_new > 2.0 and not warned_stale:
                    print(f"[!] No new data from device for {now - last_new:.1f}s (zero frames: {dropped_zero}, duplicate: {dropped_dup}). If the optodes are off the head, room light is probably saturating the receivers; it resumes when they're back on. Otherwise check the Bluetooth link.", flush=True)
                    warned_stale = True
                if now - last_stat >= 1.0:
                    print(f"[*] Stream @ {self.sample_rate}Hz | new: {total_pushed}/s | zero-dropped: {dropped_zero} | dup-dropped: {dropped_dup} | errs: {errors} | R1T1: {current_sample[0]:+.3f} R2T5: {current_sample[8]:+.3f}", flush=True)
                    total_pushed = dropped_zero = dropped_dup = 0
                    last_stat = now

                time.sleep(dt * 0.5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"[-] Read error: {e!r}", flush=True)
                time.sleep(0.02)

    def stop(self):
        self.running = False
        if self.control is not None:
            self.control.stopping.set()
        if self.sdk:
            try:
                self.sdk.artinis_stop_portable_acquisition()
                print("[+] Artinis hardware acquisition stopped.", flush=True)
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Artinis OctaMon/OptoMon fNIRS to LSL Bridge")
    parser.add_argument("--bt", default="682719F8B937", help="Device Bluetooth address (default: 682719F8B937)")
    parser.add_argument("--rate", type=int, default=50, help="Sampling rate in Hz (default: 50)")
    parser.add_argument("--mode", choices=["od", "conc"], default="od", help="Acquisition mode: 'od' for optical densities (760nm & 850nm) or 'conc' for HbO/HbR (default: od)")
    parser.add_argument("--optode-dist-cm", type=float, default=3.0, help="Transmitter-receiver distance for SDK concentrations: 3.0 on the central pads, 3.2 on the forehead pads (manual section 6.4). Default: 3.0")
    parser.add_argument("--no-preflight", action="store_true", help="Skip Bluetooth radio/pair check")
    parser.add_argument("--no-pair", action="store_true", help="Check only; never attempt to pair")
    parser.add_argument("--retries", type=int, default=3, help="SDK connect attempts (default: 3)")
    parser.add_argument("--mock", action="store_true", help="Run simulated fNIRS mock data stream")
    args = parser.parse_args()

    bridge = ArtinisOctaPortaLSL(bt_address=args.bt, sample_rate=args.rate, mode=args.mode, mock=args.mock, optode_dist_m=args.optode_dist_cm / 100.0,
                                preflight=not args.no_preflight, no_pair=args.no_pair, retries=args.retries)
    try:
        bridge.start()
    except KeyboardInterrupt:
        bridge.stop()
        print("\n[!] Bridge terminated by user.", flush=True)

if __name__ == '__main__':
    main()
