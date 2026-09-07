"""
Artinis OctaMon - Deep Command Probe
The device uses request-response protocol. We need to find the correct
command that triggers actual optical measurement data (not just echoed config).

Key finding: Response frame is always 46 bytes:
  [0x1B 0x02] [0x19 0xFF] [42 payload bytes] [0x1B 0x03]
  
  0x19 = device ID or cmd-response marker (25 = sample rate?)
  0xFF = payload length marker
  
  Payload breakdown (42 bytes = 21 uint16 words):
    word[0]  = 65305 = 0xFF19 (header/status - always same)
    word[1]  = LIVE COUNTER (increments by 16 each poll, wraps at 2290->2050)
    word[2]  = 2048  (static - possibly an offset/dark reference)
    word[3]  = 6404  (static)
    word[4]  = 31    (static)
    word[5]  = 0     (static)
    word[6..18] = static values (likely frozen optical config or last measurement)
    word[19] = 513   (static)
    word[20] = LIVE (checksum or timestamp)

Strategy: Try every plausible command 0x00..0x20 and look for responses
that return DIFFERENT data patterns, especially ones where words 2..18 become live.
"""
import sys
import time
import serial
import numpy as np

PORT = sys.argv[1] if len(sys.argv) > 1 else 'COM14'

ser = serial.Serial(PORT, 115200, timeout=0.15)
ser.dtr = True
ser.rts = True
ser.reset_input_buffer()
time.sleep(0.3)

def send_cmd(cmd_id, payload=b""):
    frame = bytearray([0x1B, 0x02, cmd_id, len(payload)])
    frame.extend(payload)
    chk = sum(frame[2:]) & 0xFF
    frame.append(chk)
    frame.extend([0x1B, 0x03])
    ser.write(frame)
    ser.flush()
    return frame

print("=" * 80)
print(" OctaMon Deep Command Probe")
print("=" * 80)

# Step 1: Try all single-byte commands 0x00 to 0x30
print("\n[1] Probing commands 0x00 to 0x30...")
for cmd in range(0x31):
    ser.reset_input_buffer()
    send_cmd(cmd)
    time.sleep(0.12)
    
    if ser.in_waiting > 0:
        resp = ser.read(ser.in_waiting)
        # Check if response is different from the standard 46-byte echo
        is_standard_46 = (len(resp) == 46)
        words = np.frombuffer(resp[:(len(resp)//2)*2], dtype=np.uint16)
        
        marker = "STANDARD" if is_standard_46 else f"DIFFERENT ({len(resp)} bytes)"
        # Check if any words beyond [1] and [20] differ from known static
        static_ref = [65305, 0, 2048, 6404, 31, 0, 3328, 30218, 2573, 3391, 29450, 2573, 21587, 21057, 3412, 19722, 2573, 3393, 10, 513, 0]
        diffs = []
        for w_i in range(min(len(words), len(static_ref))):
            if w_i in [0, 1, 20]:  # skip header, counter, checksum
                continue
            if len(words) > w_i and words[w_i] != static_ref[w_i]:
                diffs.append(w_i)
        
        if diffs or not is_standard_46:
            print(f"  Cmd 0x{cmd:02X}: {marker} | Changed words: {diffs} | words={words[:8].tolist()}")

# Step 2: Try different payload configurations for cmd 0x05 (config)
print("\n[2] Probing config command 0x05 with different payloads...")
configs = [
    bytes([0x01, 0x00, 0x19]),  # Enable channel 1 only
    bytes([0xFF, 0x00, 0x19]),  # Different mask
    bytes([0x01, 0x01, 0x19]),  # Different config
    bytes([0xFF, 0xFF, 0x32]),  # 50 Hz
    bytes([0xFF, 0xFF, 0x0A]),  # 10 Hz
    bytes([0x01, 0x00, 0x00]),  # Minimal
    bytes([0x00, 0x01]),        # 2-byte payload
    bytes([0x01]),              # 1-byte payload
]

for cfg in configs:
    ser.reset_input_buffer()
    send_cmd(0x05, cfg)
    time.sleep(0.12)
    if ser.in_waiting > 0:
        resp = ser.read(ser.in_waiting)
        words = np.frombuffer(resp[:(len(resp)//2)*2], dtype=np.uint16)
        print(f"  Config {cfg.hex(' ')}: {len(resp)} bytes, words[1:4]={words[1:4].tolist() if len(words)>3 else words.tolist()}")

# Step 3: Try the sequence: 0x01, 0x05 config, 0x06, then rapid-poll with 0x0A/0x0B/0x0C
print("\n[3] Testing alternative data-read commands after activation...")
send_cmd(0x01)
time.sleep(0.05)
send_cmd(0x05, payload=bytes([0xFF, 0xFF, 0x19]))
time.sleep(0.05)
send_cmd(0x06)
time.sleep(0.1)

for data_cmd in [0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x20, 0x53]:
    samples = []
    for _ in range(3):
        ser.reset_input_buffer()
        send_cmd(data_cmd)
        time.sleep(0.06)
        if ser.in_waiting > 0:
            resp = ser.read(ser.in_waiting)
            words = np.frombuffer(resp[:(len(resp)//2)*2], dtype=np.uint16)
            samples.append(words.copy())
    
    if samples:
        # Check word-by-word variance
        max_len = max(len(s) for s in samples)
        live_words = []
        for w_i in range(max_len):
            vals = [float(s[w_i]) for s in samples if w_i < len(s)]
            if len(vals) > 1 and np.std(vals) > 0.5:
                live_words.append(w_i)
        
        print(f"  Cmd 0x{data_cmd:02X}: {len(samples[0])} words, live_indices={live_words}, w[1:5]={samples[0][1:5].tolist() if len(samples[0])>4 else 'short'}")

ser.close()
print("\n" + "=" * 80)
print(" Probe complete.")
print("=" * 80)
