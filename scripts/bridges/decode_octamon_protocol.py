"""
Artinis OctaMon Protocol Decoder
Based on diagnostic: device is REQUEST-RESPONSE, not continuous stream.
Polls at 25Hz and decodes the 46-byte response frames.
"""
import sys
import time
import serial
import numpy as np

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM14'

ser = serial.Serial(PORT, 115200, timeout=0.05)
ser.dtr = True
ser.rts = True
ser.reset_input_buffer()
time.sleep(0.2)

def send_cmd(cmd_id, payload=b""):
    frame = bytearray([0x1B, 0x02, cmd_id, len(payload)])
    frame.extend(payload)
    chk = sum(frame[2:]) & 0xFF
    frame.append(chk)
    frame.extend([0x1B, 0x03])
    ser.write(frame)
    ser.flush()

# Activation
send_cmd(0x01)
time.sleep(0.05)
send_cmd(0x05, payload=bytes([0xFF, 0xFF, 0x19]))
time.sleep(0.05)
send_cmd(0x06)
time.sleep(0.05)

print("Polling 0x08 at 10Hz for 5 seconds, analyzing response structure...")
print("=" * 90)

all_samples = []
for i in range(50):
    ser.reset_input_buffer()
    send_cmd(0x08)
    time.sleep(0.08)
    
    resp = ser.read(ser.in_waiting) if ser.in_waiting > 0 else b''
    if len(resp) < 10:
        continue
    
    # Strip ESC-STX / ESC-ETX framing
    if resp[:2] == b'\x1b\x02' and resp[-2:] == b'\x1b\x03':
        payload = resp[2:-2]  # Everything between markers
    else:
        payload = resp
    
    # Decode all uint16 words from the full payload
    words = np.frombuffer(payload[:(len(payload)//2)*2], dtype=np.uint16)
    all_samples.append(words.copy())
    
    if i < 10 or i % 10 == 0:
        print(f"Sample {i+1:3d}: resp_len={len(resp):3d}, words({len(words):2d}): {words.tolist()}")

ser.close()

if not all_samples:
    print("No samples captured!")
    sys.exit(1)

# Find max word count
max_words = max(len(s) for s in all_samples)
print(f"\n{'='*90}")
print(f"Captured {len(all_samples)} samples, max {max_words} words per sample")
print(f"\nPer-word statistics (identifying which words are live channels):")
print(f"{'Word':>6s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'Status':>12s}")
print("-" * 65)

for w_idx in range(max_words):
    vals = [float(s[w_idx]) for s in all_samples if w_idx < len(s)]
    if not vals:
        continue
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    min_v = np.min(vals)
    max_v = np.max(vals)
    status = "LIVE (varying)" if std_v > 0.5 else "STATIC"
    print(f"  [{w_idx:2d}]  {mean_v:10.1f} {std_v:10.2f} {min_v:10.0f} {max_v:10.0f}  {status}")
