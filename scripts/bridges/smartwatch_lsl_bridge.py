"""
Smartwatch (PPG & IMU) to Lab Streaming Layer (LSL) Bridge
==========================================================

This script receives smartwatch sensor data (PPG/HeartRate and 6-DOF IMU) 
over WebSockets or UDP and streams them live into LSL.

Supported inputs:
1. Sensor Logger app (iOS / Wear OS / Android) - set live export to WebSocket (ws://<PC_IP>:8080)
2. Custom Wear OS / Android UDP packet streams
3. --mock flag for testing LSL streams without a watch connected

Usage:
    uv run python smartwatch_lsl_bridge.py --mode websocket --port 8080
    uv run python smartwatch_lsl_bridge.py --mode udp --port 5005
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
import socket
import sys
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

# Try importing websockets for websocket server mode
try:
    import asyncio
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


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


def run_mock_stream(imu_outlet, ppg_outlet):
    """Broadcasting mock smartwatch signals for pipeline verification."""
    print("\n[+] Running in MOCK mode. Broadcasting synthetic smartwatch PPG & IMU...")
    print("    Press Ctrl+C to stop.")
    
    sample_rate = 50.0  # 50 Hz IMU
    interval = 1.0 / sample_rate
    t = 0.0

    try:
        while True:
            t += interval
            # Simulate IMU data
            accel_x = np.sin(2 * np.pi * 1.0 * t) * 0.5
            accel_y = np.cos(2 * np.pi * 1.0 * t) * 0.5
            accel_z = 9.81 + np.sin(2 * np.pi * 0.2 * t) * 0.1
            gyro_x = np.sin(2 * np.pi * 2.0 * t) * 0.1
            gyro_y = np.cos(2 * np.pi * 2.0 * t) * 0.1
            gyro_z = np.sin(2 * np.pi * 0.5 * t) * 0.05

            imu_sample = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            imu_outlet.push_sample(imu_sample)

            # Push PPG/HR sample once every second
            if int(t * 50) % 50 == 0:
                simulated_hr = 70.0 + 5.0 * np.sin(2 * np.pi * 0.05 * t)
                ppg_sample = [simulated_hr, 95.0]
                ppg_outlet.push_sample(ppg_sample)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[*] Mock stream stopped.")


def run_udp_server(host, port, imu_outlet, ppg_outlet):
    """Listen for incoming JSON or CSV sensor packets from free smartwatch apps (e.g. HyperIMU)."""
    print(f"\n[+] Starting UDP server on {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    print("    Listening for UDP packets from Galaxy Watch... Press Ctrl+C to stop.")
    count = 0
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            count += 1
            text = data.decode('utf-8', errors='ignore').strip()
            
            # Debug output (commented out for silent streaming)
            # if count % 50 == 1:
            #     print(f"[UDP Packet #{count}] Received from {addr[0]}: {text[:60]}")

            # 1. Try JSON parsing
            if text.startswith('{'):
                try:
                    payload = json.loads(text)
                    if 'accel' in payload and 'gyro' in payload:
                        ax, ay, az = payload['accel']
                        gx, gy, gz = payload['gyro']
                        imu_outlet.push_sample([ax, ay, az, gx, gy, gz])
                    if 'heart_rate' in payload:
                        hr = payload['heart_rate']
                        conf = payload.get('confidence', 100.0)
                        ppg_outlet.push_sample([hr, conf])
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass

            # 2. Try CSV parsing (e.g. format: accX,accY,accZ,gyroX,gyroY,gyroZ,hr)
            parts = text.split(',')
            if len(parts) >= 6:
                try:
                    vals = [float(p.strip()) for p in parts if p.strip()]
                    if len(vals) >= 6:
                        imu_outlet.push_sample(vals[:6])
                        if len(vals) >= 7:
                            ppg_outlet.push_sample([vals[6], 100.0])
                except ValueError:
                    pass

    except KeyboardInterrupt:
        print("\n[*] UDP server stopped.")
    finally:
        sock.close()


async def handle_ws_client(websocket, imu_outlet, ppg_outlet):
    """Handle incoming WebSocket messages (e.g. from Sensor Logger app)."""
    print(f"[+] Client connected from {websocket.remote_address}")
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


async def main_ws(host, port, imu_outlet, ppg_outlet):
    print(f"\n[+] Starting WebSocket server on ws://{host}:{port}...")
    print("    Connect your Sensor Logger app to this endpoint.")
    async with websockets.serve(lambda ws: handle_ws_client(ws, imu_outlet, ppg_outlet), host, port):
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
    parser = argparse.ArgumentParser(description="Smartwatch PPG & IMU to LSL Bridge")
    parser.add_argument('--mode', choices=['websocket', 'udp'], default='websocket', help="Transport protocol")
    parser.add_argument('--host', default='0.0.0.0', help="Host IP to bind")
    parser.add_argument('--port', type=int, default=8080, help="Port to listen on")
    parser.add_argument('--mock', action='store_true', help="Run synthetic mock stream for testing")

    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 70)
    print(" Smartwatch PPG & IMU -> LSL Bridge ".center(70, "="))
    print("=" * 70)
    print(f"Local IP for Watch Connection: {local_ip}")
    print("=" * 70)

    imu_outlet, ppg_outlet = create_lsl_outlets()

    if args.mock:
        run_mock_stream(imu_outlet, ppg_outlet)
    elif args.mode == 'udp':
        run_udp_server(args.host, args.port, imu_outlet, ppg_outlet)
    elif args.mode == 'websocket':
        if not HAS_WEBSOCKETS:
            print("[-] 'websockets' package not installed. Run: `pip install websockets` or `uv add websockets`.")
            sys.exit(1)
        try:
            asyncio.run(main_ws(args.host, args.port, imu_outlet, ppg_outlet))
        except KeyboardInterrupt:
            print("\n[*] WebSocket server stopped.")


if __name__ == '__main__':
    main()
