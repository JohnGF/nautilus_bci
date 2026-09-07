# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pylsl>=1.18.2",
# ]
# ///
import argparse
import socket
import json
import time
from pylsl import StreamInfo, StreamOutlet, local_clock

def main():
    parser = argparse.ArgumentParser(description="UDP to LSL Bridge for External Game Engines")
    parser.add_argument("--port", type=int, default=9000, help="UDP port to listen on (default: 9000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind to (default: 127.0.0.1)")
    parser.add_argument("--marker-name", type=str, default="ExternalGameMarkers", help="LSL Stream Name")
    parser.add_argument("--source-id", type=str, default="External_Game_Bridge", help="LSL Source ID")
    args = parser.parse_args()

    # Set up LSL Outlet
    print(f"Setting up LSL Marker Outlet ('{args.marker_name}')")
    info = StreamInfo(
        name=args.marker_name,
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id=args.source_id
    )
    outlet = StreamOutlet(info)
    print("LSL Outlet created successfully.")

    # Set up UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    print(f"Listening for UDP messages on {args.host}:{args.port}...")
    print("Expected JSON format: {\"name\": \"MarkerName\", \"duration\": 0.1}")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            timestamp = local_clock()
            try:
                msg = data.decode('utf-8').strip()
                # Parse JSON
                payload = json.loads(msg)

                # Extract name and duration
                marker_name = payload.get("name")
                duration = payload.get("duration", 0.0)

                if marker_name:
                    # Append duration suffix for BIDS recorder parsing
                    # Same format used in base_task.py: {marker_str}_dur_{duration}
                    lsl_str = f"{marker_name}_dur_{duration}"
                    outlet.push_sample([lsl_str], timestamp)
                    print(f"[{time.strftime('%H:%M:%S')}] Forwarded LSL Marker: {marker_name} (Duration: {duration}s) from {addr}")
                else:
                    print(f"[Warning] Received JSON missing 'name' field: {payload}")

            except json.JSONDecodeError:
                print(f"[Warning] Received invalid JSON from {addr}: {data.decode('utf-8', errors='ignore')}")
            except Exception as e:
                print(f"[Error] Error processing message from {addr}: {e}")

    except KeyboardInterrupt:
        print("\nShutting down bridge...")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
