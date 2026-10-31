"""
Smartwatch / Wearable (PPG & IMU) to Lab Streaming Layer (LSL) Bridge
=====================================================================

This script receives wearable sensor data (PPG/HeartRate and 6-DOF IMU)
over WebSockets or UDP and streams them live into LSL.

Supported inputs:
1. W2 multi-device UDP packets (https://github.com/JohnGF/galaxy_watch_lsl >= 1.2, ESP32 boards, ...):
   any number of devices, one set of LSL streams per device, named by body
   location (e.g. 'IMU_L_wrist', 'IMU_R_ankle'). See docs/guides/wearable_protocol.md
2. Legacy single-watch UDP CSV packets -> 'Smartwatch_IMU' / 'Smartwatch_PPG'
3. Sensor Logger app (iOS / Wear OS / Android) - set live export to WebSocket (ws://<PC_IP>:8080)
4. --mock flag: simulates four W2 devices (both wrists, both ankles)

Usage:
    uv run python smartwatch_lsl_bridge.py --mode udp --port 5005
    uv run python smartwatch_lsl_bridge.py --mode websocket --port 8080
    uv run python smartwatch_lsl_bridge.py --mock
"""
import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)


import argparse
import json
import random
import socket
import sys
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

from clock_sync import ECHO_KIND, PING_PREFIX
import threading

from wearable_protocol import (BEACON_PORT, DeviceClockMapper, DeviceRegistry, format_beacon,
                               format_packet, parse_packet)

# Try importing websockets for websocket server mode
try:
    import asyncio
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

STATUS_EVERY_S = 5.0
MOCK_LOCATIONS = ["L_wrist", "R_wrist", "L_ankle", "R_ankle"]
MOCK_DRIFT = 30e-6  # device crystals run 30 ppm fast, so the drift fit has something to find


def create_lsl_outlets():
    """Create LSL outlets for IMU and PPG streams."""
    # 1. IMU Stream (6 channels: Accel X,Y,Z + Gyro X,Y,Z)
    imu_info = StreamInfo(
        name='Smartwatch_IMU',
        type='IMU',
        channel_count=6,
        nominal_srate=50.0,  # 50 Hz default
        channel_format='float32',
        source_id='smartwatch_imu_01'
    )
    imu_channels = imu_info.desc().append_child("channels")
    for name, unit in [('Accel_X', 'm/s2'), ('Accel_Y', 'm/s2'), ('Accel_Z', 'm/s2'),
                        ('Gyro_X', 'rad/s'), ('Gyro_Y', 'rad/s'), ('Gyro_Z', 'rad/s')]:
        ch = imu_channels.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "IMU")

    imu_outlet = StreamOutlet(imu_info)

    # 2. PPG / HR Stream (2 channels: HeartRate/PPG_val, Confidence)
    ppg_info = StreamInfo(
        name='Smartwatch_PPG',
        type='PPG',
        channel_count=2,
        nominal_srate=1.0,  # 1 Hz default for HR / pulse
        channel_format='float32',
        source_id='smartwatch_ppg_01'
    )
    ppg_channels = ppg_info.desc().append_child("channels")
    for name, unit in [('HeartRate', 'bpm'), ('Confidence', 'percentage')]:
        ch = ppg_channels.append_child("channel")
        ch.append_child_value("label", name)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "PPG")

    ppg_outlet = StreamOutlet(ppg_info)

    print("[+] LSL Outlets created:")
    print("    - 'Smartwatch_IMU' (6 channels: Accel XYZ + Gyro XYZ)")
    print("    - 'Smartwatch_PPG' (2 channels: HeartRate BPM, Confidence)")

    return imu_outlet, ppg_outlet


class LegacyOutlets:
    """Creates the single-watch 'Smartwatch_*' outlets only once a legacy packet arrives,
    so a W2-only session does not advertise two empty streams to the recorder."""

    def __init__(self):
        self._outlets = None

    def get(self):
        if self._outlets is None:
            self._outlets = create_lsl_outlets()
        return self._outlets


def run_mock_stream(registry):
    """Simulate four W2 devices through the real parse/registry path. Each has its own clock
    (random origin, 30 ppm drift) and answers pings over a jittery simulated Wi-Fi link."""
    print("\n[+] Running in MOCK mode: " + ", ".join(MOCK_LOCATIONS))
    print("    Press Ctrl+C to stop.")

    sample_rate = 50.0
    batch = 5
    interval = batch / sample_rate
    lsl0 = local_clock()
    devices = [(f"mock-{loc}", loc, random.uniform(0, 1000.0)) for loc in MOCK_LOCATIONS]
    seq = {d[0]: 0 for d in devices}

    def device_clock(clock0, lsl_t):
        return clock0 + (lsl_t - lsl0) * (1 + MOCK_DRIFT)

    def to_ns(t):
        return int(round(t * 1e9))

    start = time.monotonic()
    next_status = start + STATUS_EVERY_S
    n = 0

    try:
        while True:
            for device_id, loc, clock0 in devices:
                samples = []
                for k in range(batch):
                    t = (n * batch + k) / sample_rate
                    phase = MOCK_LOCATIONS.index(loc)
                    samples.append((to_ns(device_clock(clock0, lsl0 + t)), [
                        np.sin(2 * np.pi * 1.0 * t + phase) * 0.5,
                        np.cos(2 * np.pi * 1.0 * t + phase) * 0.5,
                        9.81 + np.sin(2 * np.pi * 0.2 * t) * 0.1,
                        np.sin(2 * np.pi * 2.0 * t) * 0.1,
                        np.cos(2 * np.pi * 2.0 * t) * 0.1,
                        np.sin(2 * np.pi * 0.5 * t) * 0.05,
                    ]))
                packets = [format_packet(device_id, loc, seq[device_id], "IMU", samples, sample_rate)]
                if n % int(sample_rate / batch) == 0:  # once a second
                    t_ns = samples[-1][0]
                    hr = 70.0 + 5.0 * np.sin(2 * np.pi * 0.05 * n * interval)
                    packets.append(format_packet(device_id, loc, seq[device_id] + 1, "HR", [(t_ns, [hr])]))
                    packets.append(format_packet(device_id, loc, seq[device_id] + 2, "HB", [(t_ns, [n, 0, 90])]))
                for p in packets:
                    registry.handle(parse_packet(p), addr=("mock", 0))
                    seq[device_id] += 1

                t1 = local_clock()
                ping = registry.next_ping(device_id, t1)
                if ping is not None:
                    ping_id = int(ping.decode()[len(PING_PREFIX):])
                    # Wi-Fi legs of 1-15 ms each, independently jittered, like a real link
                    up, down, turnaround = random.uniform(1e-3, 15e-3), random.uniform(1e-3, 15e-3), 2e-4
                    t2 = device_clock(clock0, t1 + up)
                    echo = format_packet(device_id, loc, seq[device_id], ECHO_KIND,
                                         [(to_ns(t2), [ping_id, turnaround * 1e6])])
                    seq[device_id] += 1
                    registry.handle(parse_packet(echo), arrival=t1 + up + turnaround + down)
            n += 1
            if time.monotonic() >= next_status:
                print_status(registry)
                next_status += STATUS_EVERY_S
            time.sleep(max(0.0, start + n * interval - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[*] Mock stream stopped.")


def print_status(registry):
    lines = registry.status_lines()
    if lines:
        print(f"[status] {len(lines)} device(s)")
        for line in lines:
            print(line)


def handle_legacy_udp(text, legacy, clock):
    """Single-watch JSON / CSV packets from apps that predate the W2 protocol."""
    # 1. Try JSON parsing
    if text.startswith('{'):
        try:
            payload = json.loads(text)
            imu_outlet, ppg_outlet = legacy.get()
            if 'accel' in payload and 'gyro' in payload:
                ax, ay, az = payload['accel']
                gx, gy, gz = payload['gyro']
                imu_outlet.push_sample([ax, ay, az, gx, gy, gz])
            if 'heart_rate' in payload:
                hr = payload['heart_rate']
                conf = payload.get('confidence', 100.0)
                ppg_outlet.push_sample([hr, conf])
            return
        except (json.JSONDecodeError, KeyError):
            pass

    # 2. Try CSV parsing
    #    accX,accY,accZ,gyroX,gyroY,gyroZ,hr[,touchX,touchY,isTouched,timestampNs]
    parts = text.split(',')
    if len(parts) >= 6:
        try:
            vals = [float(p.strip()) for p in parts if p.strip()]
            if len(vals) >= 6:
                imu_outlet, ppg_outlet = legacy.get()
                # Newer watch app sends the sensor timestamp; older builds fall back to arrival time
                stamp = clock.map(vals[10] * 1e-9) if len(vals) >= 11 else local_clock()
                imu_outlet.push_sample(vals[:6], stamp)
                if len(vals) >= 7:
                    ppg_outlet.push_sample([vals[6], 100.0], stamp)
        except ValueError:
            pass


def beacon_targets(local_ip):
    """Limited broadcast leaves through the default interface only, so also hit this host's /24."""
    targets = ["255.255.255.255"]
    parts = local_ip.split(".")
    if len(parts) == 4 and local_ip != "127.0.0.1":
        targets.append(".".join(parts[:3] + ["255"]))
    return targets


def start_beacon(port, local_ip, period_s=1.0):
    """Announce this bridge once a second on BEACON_PORT so devices set to 'auto' find it."""
    payload = format_beacon(socket.gethostname(), port).encode()
    targets = beacon_targets(local_ip)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def loop():
        while True:
            for addr in targets:
                try:
                    sock.sendto(payload, (addr, BEACON_PORT))
                except OSError:
                    pass  # no route right now (Wi-Fi down); try again next period
            time.sleep(period_s)

    threading.Thread(target=loop, name="beacon", daemon=True).start()
    print(f"    Discovery beacon: {payload.decode()} -> {', '.join(targets)} :{BEACON_PORT} every {period_s:g}s")


def run_udp_server(host, port, registry, legacy, beacon=True):
    """Listen for W2 packets from any number of wearables, plus legacy single-watch packets."""
    print(f"\n[+] Starting UDP server on {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Four devices at 10-20 packets/s each can outrun the default buffer while the GIL is busy
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((host, port))
    sock.settimeout(1.0)  # wake up for the status table even when every device is silent

    if beacon:
        start_beacon(port, get_local_ip())
    print("    Listening for UDP packets from wearables... Press Ctrl+C to stop.")
    legacy_clock = DeviceClockMapper()
    next_status = time.monotonic() + STATUS_EVERY_S
    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                data = None
            if data is not None:
                arrival = local_clock()
                text = data.decode('utf-8', errors='ignore').strip()
                try:
                    packet = parse_packet(text)
                except ValueError as e:
                    print(f"[-] Bad W2 packet from {addr[0]}: {e}")
                    packet = None
                else:
                    if packet is None:
                        handle_legacy_udp(text, legacy, legacy_clock)
                if packet is not None:
                    try:
                        registry.handle(packet, arrival, addr)
                    except Exception as e:  # one misbehaving device must not stop the others
                        print(f"[-] Dropped {packet.kind} packet from {packet.device_id}: {e}")
                        continue
                    # Ping right after the device's own packet: its radio is still awake
                    ping = registry.next_ping(packet.device_id, local_clock())
                    if ping is not None:
                        sock.sendto(ping, addr)

            if time.monotonic() >= next_status:
                print_status(registry)
                next_status = time.monotonic() + STATUS_EVERY_S

    except KeyboardInterrupt:
        print("\n[*] UDP server stopped.")
    finally:
        sock.close()


async def handle_ws_client(websocket, legacy):
    """Handle incoming WebSocket messages (e.g. from Sensor Logger app)."""
    print(f"[+] Client connected from {websocket.remote_address}")
    imu_outlet, ppg_outlet = legacy.get()
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                # Sensor Logger packet parsing structure:
                if 'payload' in data:
                    for item in data['payload']:
                        name = item.get('name')
                        values = item.get('values', {})

                        if name in ('accelerometer', 'accelerometeruncalibrated'):
                            ax, ay, az = values.get('x', 0), values.get('y', 0), values.get('z', 0)
                            # Push buffer sample
                            imu_outlet.push_sample([ax, ay, az, 0, 0, 0])
                        elif name in ('gyroscope', 'gyroscopeuncalibrated'):
                            gx, gy, gz = values.get('x', 0), values.get('y', 0), values.get('z', 0)
                            imu_outlet.push_sample([0, 0, 0, gx, gy, gz])
                        elif name in ('heartrate', 'ppg'):
                            hr = values.get('bpm', values.get('value', 0))
                            ppg_outlet.push_sample([float(hr), 100.0])

            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        print(f"[*] Client disconnected: {websocket.remote_address}")


async def main_ws(host, port, legacy):
    print(f"\n[+] Starting WebSocket server on ws://{host}:{port}...")
    print("    Connect your Sensor Logger app to this endpoint.")
    async with websockets.serve(lambda ws: handle_ws_client(ws, legacy), host, port):
        await asyncio.Future()  # run forever


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description="Smartwatch / wearable PPG & IMU to LSL Bridge")
    parser.add_argument('--mode', choices=['websocket', 'udp'], default='websocket', help="Transport protocol")
    parser.add_argument('--host', default='0.0.0.0', help="Host IP to bind")
    parser.add_argument('--port', type=int, default=8080, help="Port to listen on")
    parser.add_argument('--mock', action='store_true', help="Simulate four W2 devices for testing")
    parser.add_argument('--no-beacon', action='store_true',
                        help=f"UDP mode: do not broadcast the discovery beacon on port {BEACON_PORT}")

    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 70)
    print(" Smartwatch PPG & IMU -> LSL Bridge ".center(70, "="))
    print("=" * 70)
    print(f"Local IP for Watch Connection: {local_ip}")
    print("=" * 70)

    registry = DeviceRegistry()
    legacy = LegacyOutlets()

    if args.mock:
        run_mock_stream(registry)
    elif args.mode == 'udp':
        run_udp_server(args.host, args.port, registry, legacy, beacon=not args.no_beacon)
    elif args.mode == 'websocket':
        if not HAS_WEBSOCKETS:
            print("[-] 'websockets' package not installed. Run: `pip install websockets` or `uv add websockets`.")
            sys.exit(1)
        try:
            asyncio.run(main_ws(args.host, args.port, legacy))
        except KeyboardInterrupt:
            print("\n[*] WebSocket server stopped.")


if __name__ == '__main__':
    main()
