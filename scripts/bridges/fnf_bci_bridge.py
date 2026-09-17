import sys
import time
import argparse
import json
import socket
import numpy as np
import joblib

try:
    from pylsl import resolve_stream, StreamInlet
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

def send_to_godot(direction, ip="127.0.0.1", port=5005):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(direction.encode('utf-8'), (ip, port))
    except Exception as e:
        print(f"[UDP] Failed to send to Godot: {e}")

def main():
    parser = argparse.ArgumentParser(description="FNF Real-Time BCI Bridge")
    parser.add_argument("--source", type=str, choices=["simulator", "lsl"], default="simulator")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--sub", type=str, default="01")
    parser.add_argument("--ses", type=str, default="07")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=5005, help="Godot UDP port")
    parser.add_argument("--auto-send", action="store_true", default=True)
    parser.add_argument("--no-auto-send", dest="auto_send", action="store_false")
    
    args = parser.parse_args()

    print(f"==================================================")
    print(f" FNF Real-Time BCI Godot Bridge ")
    print(f"==================================================")
    print(f" Source:    {args.source}")
    print(f" Model:     {args.model}")
    print(f" Auto-send: {args.auto_send}")
    print(f"==================================================")

    # 1. Load the Model
    try:
        print(f"[*] Loading model from {args.model}...")
        data = joblib.load(args.model)
        clf = data['model']
        classes = data.get('classes', ['Left', 'Right', 'Up', 'Down', 'Rest'])
        print(f"[+] Model loaded successfully. Classes: {classes}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        sys.exit(1)

    # 2. Setup Input Stream
    inlet = None
    if args.source == "lsl":
        if not HAS_LSL:
            print("[ERROR] pylsl is not installed. Falling back to simulator.")
            args.source = "simulator"
        else:
            print("[*] Looking for an EEG LSL stream...")
            streams = resolve_stream('type', 'EEG')
            if not streams:
                print("[ERROR] No EEG stream found. Is the streamer running?")
                sys.exit(1)
            inlet = StreamInlet(streams[0], max_buflen=360)
            info = inlet.info()
            print(f"[+] Connected to {info.name()} at {info.nominal_srate()} Hz")

    if args.source == "simulator":
        print("[*] Running in Simulator Mode (Generating synthetic data)...")
        fs = 250.0
    else:
        fs = info.nominal_srate() if info.nominal_srate() > 0 else 250.0

    buf_seconds = 3.0
    buf_samples = int(fs * buf_seconds)
    num_channels = 32

    # Continuous decoding loop
    try:
        while True:
            if args.source == "lsl" and inlet is not None:
                # Wait to collect a 3-second chunk (this is a simplified blocking wait)
                chunk, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=buf_samples)
                if len(chunk) < buf_samples:
                    # Not enough data yet, in a real pipeline we use a rolling buffer
                    time.sleep(0.1)
                    continue
                # Keep first 32 channels (ignore Battery or Aux)
                X_live = np.array(chunk)[-buf_samples:, :32].T
            else:
                # Simulator mode: emit a random valid direction every 2 seconds
                time.sleep(2.0)
                random_idx = np.random.randint(0, len(classes))
                direction = classes[random_idx]
                if direction != 'Rest':
                    print(f"[*] SIMULATOR DECODED -> {direction}")
                    if args.auto_send:
                        send_to_godot(direction)
                continue

            # Formatting for model: (1, 32, n_samples)
            X_input = np.expand_dims(X_live, axis=0)

            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_input)[0]
                    max_prob = np.max(probs)
                    pred_idx = np.argmax(probs)
                    direction = classes[pred_idx]
                    
                    if max_prob >= args.threshold:
                        if direction != 'Rest':
                            print(f"[DECISION] {direction} (Confidence: {max_prob*100:.1f}%)")
                            if args.auto_send:
                                send_to_godot(direction)
                        else:
                            print(f"[DECISION] Rest (Confidence: {max_prob*100:.1f}%)")
                    else:
                        print(f"[UNCERTAIN] Max confidence {max_prob*100:.1f}% below threshold {args.threshold}")
                else:
                    # Model doesn't support proba, just predict
                    pred_idx = clf.predict(X_input)[0]
                    direction = classes[pred_idx]
                    if direction != 'Rest':
                        print(f"[DECISION] {direction}")
                        if args.auto_send:
                            send_to_godot(direction)
            except Exception as e:
                print(f"[WARNING] Inference error: {e}")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] Stopping pipeline.")
        sys.exit(0)

if __name__ == "__main__":
    main()
