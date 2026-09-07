"""
Standalone Galaxy Watch 7 UDP Sensor Diagnostic Monitor
=========================================================

Isolates and tests ONLY the smartwatch connection.
Prints every incoming raw UDP packet, packet rate (Hz), and parsed values.

Usage:
  uv run python watch_diagnostic_monitor.py --port 5005
"""
import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)


import sys
import socket
import time
import json


def get_local_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    return ips if ips else ["192.168.137.1"]


def main():
    port = 5005
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    local_ips = get_local_ips()
    hotspot_ip = next((ip for ip in local_ips if ip.startswith("192.168.137.")), local_ips[0])

    print("=" * 75)
    print(" GALAXY WATCH 7 SENSOR DIAGNOSTIC MONITOR ".center(75, "="))
    print("=" * 75)
    print(f"Target PC Hotspot IP to put in Watch App:  {hotspot_ip}")
    print(f"Target UDP Port in Watch App            :  {port}")
    print(f"All Detected PC IPs                      :  {', '.join(local_ips)}")
    print("=" * 75)
    print(f"[+] Binding UDP socket on 0.0.0.0:{port}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)

    print(f"[*] Listening for Watch UDP packets on port {port}...")
    print("    Press Ctrl+C to stop.\n" + "-" * 75)

    packet_count = 0
    start_time = time.time()
    last_print_time = time.time()

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
                packet_count += 1
                now = time.time()
                text = data.decode('utf-8', errors='ignore').strip()

                # Calculate live rate (Hz)
                elapsed = now - start_time
                hz = packet_count / elapsed if elapsed > 0 else 0.0

                # Print every packet or rate sample
                if packet_count <= 5 or packet_count % 25 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Packet #{packet_count:<5} | From: {addr[0]}:{addr[1]} | Rate: {hz:4.1f} Hz")
                    print(f"   Raw Content: {text}")

                    # Attempt CSV parsing
                    parts = [p.strip() for p in text.split(',')]
                    if len(parts) >= 6:
                        try:
                            vals = [float(p) for p in parts if p]
                            ax, ay, az = vals[0], vals[1], vals[2]
                            gx, gy, gz = vals[3], vals[4], vals[5]
                            hr = vals[6] if len(vals) >= 7 else 0.0
                            print(f"   --> Accel XYZ: ({ax:6.2f}, {ay:6.2f}, {az:6.2f}) m/s² | Gyro XYZ: ({gx:5.2f}, {gy:5.2f}, {gz:5.2f}) rad/s | HR: {hr:.0f} BPM")
                        except ValueError:
                            pass
                    print("-" * 75)

            except socket.timeout:
                if packet_count == 0 and (time.time() - last_print_time) >= 4.0:
                    last_print_time = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] ⏳ Waiting for watch data... (Make sure Watch App has IP: {hotspot_ip} and Port: {port})")

    except KeyboardInterrupt:
        print(f"\n[*] Diagnostic monitor stopped. Total packets captured: {packet_count}")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
