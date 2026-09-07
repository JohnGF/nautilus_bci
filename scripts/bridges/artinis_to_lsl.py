"""
Artinis OctaMon / OptoMon fNIRS to Lab Streaming Layer (LSL) Bridge
Uses official OctaPortaSDK.dll (g.tec / Artinis C SDK) for direct hardware acquisition without MATLAB.
"""

import os
import sys
import time
import argparse
import ctypes
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

SDK_DIR = r"C:\Program Files\gtec\gHIsys\gSENSORfNIRS"
SDK_DLL = os.path.join(SDK_DIR, "OctaPortaSDK.dll")

class ArtinisOctaPortaLSL:
    def __init__(self, bt_address="682719F8B937", sample_rate=50, num_channels=16, dpf=6.0, optode_dist_m=0.035, mode="od", mock=False):
        self.bt_address = bt_address
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.dpf = dpf
        self.optode_dist_m = optode_dist_m
        self.mode = mode
        self.mock = mock
        self.running = False
        self.sdk = None
        self.outlet = None

    def setup_lsl(self):
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
        for i in range(8):
            ch = desc.append_child("channel")
            ch.append_child_value("label", f"Tx{i+1}_HbO" if self.mode == "conc" else f"Tx{i+1}_760nm")
            ch.append_child_value("type", "HbO" if self.mode == "conc" else "fNIRS")
            ch.append_child_value("unit", "micromol/L" if self.mode == "conc" else "OD")

        for i in range(8):
            ch = desc.append_child("channel")
            ch.append_child_value("label", f"Tx{i+1}_HbR" if self.mode == "conc" else f"Tx{i+1}_850nm")
            ch.append_child_value("type", "HbR" if self.mode == "conc" else "fNIRS")
            ch.append_child_value("unit", "micromol/L" if self.mode == "conc" else "OD")

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

    def start_mock(self):
        self.running = True
        self.setup_lsl()
        print("[*] Streaming realistic simulated fNIRS hemodynamics... (Ctrl+C to stop)", flush=True)
        dt = 1.0 / self.sample_rate
        t_start = time.time()
        last_stat = time.time()

        while self.running:
            t = time.time() - t_start
            sample = np.zeros(self.num_channels, dtype=np.float32)

            cardiac = 0.5 * np.sin(2 * np.pi * 1.15 * t)
            resp = 0.3 * np.sin(2 * np.pi * 0.25 * t)
            mayer = 0.8 * np.sin(2 * np.pi * 0.10 * t)
            is_active = (int(t) % 16) < 8
            hrf_hbo = 3.5 * (1.0 / (1.0 + np.exp(-1.5 * ((t % 16) - 3)))) if is_active else 0.0
            hrf_hbr = -1.2 * (1.0 / (1.0 + np.exp(-1.5 * ((t % 16) - 3)))) if is_active else 0.0

            for i in range(8):
                noise1 = float(np.random.normal(0, 0.05))
                noise2 = float(np.random.normal(0, 0.03))
                sample[i] = float(hrf_hbo + mayer + resp * 0.6 + cardiac + noise1)
                sample[8 + i] = float(hrf_hbr - 0.3 * mayer - 0.2 * resp + noise2)

            self.outlet.push_sample(sample.tolist(), local_clock())

            if time.time() - last_stat > 1.0:
                print(f"[*] [MOCK fNIRS] Active @ {self.sample_rate}Hz | Tx1: {sample[0]:+.2f} | Tx8: {sample[8]:+.2f}", flush=True)
                last_stat = time.time()

            time.sleep(dt)

    def start(self):
        if self.mock:
            self.start_mock()
            return

        print(f"[*] Initializing Artinis OctaPortaSDK for device BT: {self.bt_address}...", flush=True)
        self.init_sdk()

        p_str = ctypes.c_char_p(self.bt_address.encode('ascii'))
        print(f"[*] Connecting to Artinis device ({self.bt_address})...", flush=True)
        res = self.sdk.artinis_start_portable_acquisition(ctypes.byref(p_str), 1, 2, self.sample_rate)
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
        print(f"[+] Device Battery Level: {battery}%", flush=True)

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
        current_sample = np.zeros(self.num_channels, dtype=np.float32)

        print("\n[*] Live fNIRS streaming active on LSL. (Press Ctrl+C to stop)\n", flush=True)

        dt = 1.0 / self.sample_rate

        while self.running:
            try:
                ret = self.sdk.artinis_get_concentrations_tsi(
                    ctypes.byref(n_samples),
                    buf_od,
                    ctypes.c_int(100),
                    ctypes.byref(n_od),
                    buf_conc,
                    ctypes.c_int(100),
                    buf_sq,
                    ctypes.byref(n_sq)
                )

                s_count = n_samples.value
                
                # Check for samples arriving from hardware
                if s_count > 0:
                    raw_data = np.ctypeslib.as_array(buf_od)
                    raw_sq_data = np.ctypeslib.as_array(buf_sq)
                    
                    if np.count_nonzero(raw_data[:16]) > 0:
                        frame = raw_data[:16].astype(np.float32)
                    elif np.count_nonzero(raw_sq_data[:16]) > 0:
                        frame = raw_sq_data[:16].astype(np.float32)
                    else:
                        frame = raw_data[:16].astype(np.float32)
                        
                    self.outlet.push_sample(frame.tolist(), local_clock())
                    current_sample[:] = frame
                    total_pushed += 1

                now = time.time()
                if now - last_stat >= 1.0:
                    ch1 = current_sample[0]
                    ch8 = current_sample[8] if len(current_sample) > 8 else 0.0
                    print(f"[*] Stream Active @ {self.sample_rate}Hz | Pushed: {total_pushed} smp/s | Ch1: {ch1:+.3f} | Ch8: {ch8:+.3f}", flush=True)
                    total_pushed = 0
                    last_stat = now

                time.sleep(dt * 0.95)

            except KeyboardInterrupt:
                break
            except Exception as e:
                time.sleep(0.02)

    def stop(self):
        self.running = False
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
    parser.add_argument("--mock", action="store_true", help="Run simulated fNIRS mock data stream")
    args = parser.parse_args()

    bridge = ArtinisOctaPortaLSL(bt_address=args.bt, sample_rate=args.rate, mode=args.mode, mock=args.mock)
    try:
        bridge.start()
    except KeyboardInterrupt:
        bridge.stop()
        print("\n[!] Bridge terminated by user.", flush=True)

if __name__ == '__main__':
    main()
