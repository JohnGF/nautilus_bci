"""
Artinis OctaMon Raw Stream Diagnostic
Captures ALL raw bytes from the device with a clean serial buffer,
analyzes the byte stream structure, and identifies the actual protocol.
"""
import sys
import time
import serial
import numpy as np
from collections import Counter

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM14'
BAUD = 115200

print(f"{'='*70}")
print(f" Artinis OctaMon Raw Stream Diagnostic")
print(f" Port: {PORT} | Baud: {BAUD}")
print(f"{'='*70}")

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.5, write_timeout=0.5)
    ser.dtr = True
    ser.rts = True
except Exception as e:
    print(f"[-] Cannot open {PORT}: {e}")
    sys.exit(1)

# Step 1: Drain any stale buffer
ser.reset_input_buffer()
ser.reset_output_buffer()
time.sleep(0.3)
stale = ser.read(ser.in_waiting) if ser.in_waiting else b''
print(f"\n[1] Drained {len(stale)} stale bytes from buffer")

# Step 2: Check if device sends anything unprompted
print(f"\n[2] Listening for unprompted data (2 seconds)...")
t0 = time.time()
unprompted = bytearray()
while time.time() - t0 < 2.0:
    if ser.in_waiting > 0:
        unprompted.extend(ser.read(ser.in_waiting))
    time.sleep(0.05)
print(f"    Received {len(unprompted)} unprompted bytes")
if unprompted:
    print(f"    First 64 bytes hex: {unprompted[:64].hex(' ')}")

# Step 3: Send activation sequence and capture raw response
def send_cmd(cmd_id, payload=b""):
    frame = bytearray([0x1B, 0x02, cmd_id, len(payload)])
    frame.extend(payload)
    chk = sum(frame[2:]) & 0xFF
    frame.append(chk)
    frame.extend([0x1B, 0x03])
    ser.write(frame)
    ser.flush()
    return frame

print(f"\n[3] Sending activation handshake...")
ser.reset_input_buffer()
time.sleep(0.1)

cmds = [
    ("0x01 Info/Wake",   0x01, b""),
    ("0x05 Config",      0x05, bytes([0xFF, 0xFF, 0x19])),
    ("0x06 Start Meas",  0x06, b""),
    ("0x08 Start Stream", 0x08, b""),
]

for label, cmd, payload in cmds:
    frame = send_cmd(cmd, payload)
    print(f"    Sent {label}: {frame.hex(' ')}")
    time.sleep(0.15)
    if ser.in_waiting > 0:
        resp = ser.read(ser.in_waiting)
        print(f"    -> Response ({len(resp)} bytes): {resp[:48].hex(' ')}{'...' if len(resp) > 48 else ''}")
    else:
        print(f"    -> No immediate response")

# Step 4: Capture raw stream for 5 seconds
print(f"\n[4] Capturing raw byte stream for 5 seconds...")
all_bytes = bytearray()
chunk_sizes = []
t0 = time.time()
while time.time() - t0 < 5.0:
    if ser.in_waiting > 0:
        chunk = ser.read(ser.in_waiting)
        all_bytes.extend(chunk)
        chunk_sizes.append((time.time() - t0, len(chunk)))
    time.sleep(0.01)

print(f"    Total captured: {len(all_bytes)} bytes in {len(chunk_sizes)} chunks")
if chunk_sizes:
    sizes = [s for _, s in chunk_sizes]
    print(f"    Chunk sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
    print(f"    Chunks/sec: {len(chunk_sizes) / 5.0:.1f}")

# Step 5: Analyze byte distribution
print(f"\n[5] Byte frequency analysis (top 20):")
byte_counts = Counter(all_bytes)
for byte_val, count in byte_counts.most_common(20):
    char_repr = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
    print(f"    Byte 0x{byte_val:02X} ({char_repr:>4s}): {count:5d} times ({100*count/len(all_bytes):.1f}%)")

# Step 6: Find 0x1B 0x02 / 0x1B 0x03 frame markers
esc_stx_count = 0
esc_etx_count = 0
for i in range(len(all_bytes) - 1):
    if all_bytes[i] == 0x1B and all_bytes[i+1] == 0x02:
        esc_stx_count += 1
    if all_bytes[i] == 0x1B and all_bytes[i+1] == 0x03:
        esc_etx_count += 1
print(f"\n[6] Frame marker analysis:")
print(f"    ESC-STX (0x1B 0x02) occurrences: {esc_stx_count}")
print(f"    ESC-ETX (0x1B 0x03) occurrences: {esc_etx_count}")

# Step 7: Try to extract frames and dump them
print(f"\n[7] Extracted frames (ESC-STX to ESC-ETX):")
frames = []
pos = 0
while pos < len(all_bytes) - 1:
    idx_start = all_bytes.find(b'\x1b\x02', pos)
    if idx_start == -1:
        break
    idx_end = all_bytes.find(b'\x1b\x03', idx_start + 2)
    if idx_end == -1:
        break
    frame_data = all_bytes[idx_start+2:idx_end]
    frames.append((idx_start, frame_data))
    pos = idx_end + 2

print(f"    Found {len(frames)} frames total")
for i, (offset, frame) in enumerate(frames[:10]):
    cmd = frame[0] if len(frame) > 0 else None
    plen = frame[1] if len(frame) > 1 else None
    words = []
    if len(frame) > 3:
        payload = frame[2:-1]
        words = np.frombuffer(payload[:(len(payload)//2)*2], dtype=np.uint16).tolist()
    print(f"    Frame {i+1}: offset={offset}, len={len(frame)}, cmd=0x{cmd:02X}, payload_len={plen}, words({len(words)})={words[:8]}{'...' if len(words)>8 else ''}")

# Step 8: Look for alternative framing - maybe the device uses \r\n or other delimiters
print(f"\n[8] Alternative delimiter analysis:")
newline_count = all_bytes.count(b'\r\n'[0])  # 0x0D
stx_raw = all_bytes.count(0x02)
etx_raw = all_bytes.count(0x03)
print(f"    0x0D (\\r) count: {newline_count}")
print(f"    0x0A (\\n) count: {all_bytes.count(0x0A)}")
print(f"    0x02 (STX) count: {stx_raw}")
print(f"    0x03 (ETX) count: {etx_raw}")

# Step 9: Dump first 256 raw bytes for manual inspection
print(f"\n[9] First 256 raw bytes (hex dump):")
for row in range(min(16, (len(all_bytes)+15)//16)):
    start = row * 16
    end = min(start + 16, len(all_bytes))
    hex_part = ' '.join(f'{b:02x}' for b in all_bytes[start:end])
    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in all_bytes[start:end])
    print(f"    {start:04x}: {hex_part:<48s}  {ascii_part}")

# Step 10: Also try polling (send 0x08 repeatedly and see if data changes)
print(f"\n[10] Polling test: sending 0x08 five times at 100ms intervals...")
ser.reset_input_buffer()
for poll_i in range(5):
    send_cmd(0x08)
    time.sleep(0.1)
    if ser.in_waiting > 0:
        resp = ser.read(ser.in_waiting)
        words = np.frombuffer(resp[:(len(resp)//2)*2], dtype=np.uint16)
        print(f"    Poll {poll_i+1}: {len(resp)} bytes, first_words={words[:6].tolist() if len(words)>=6 else words.tolist()}")
    else:
        print(f"    Poll {poll_i+1}: no response")

ser.close()
print(f"\n{'='*70}")
print(f" Diagnostic complete.")
print(f"{'='*70}")
