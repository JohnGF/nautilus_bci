"""
Artinis Binary Protocol Probe & Stream Initiator
Tests the official Artinis STX/ETX (0x1B 0x02 ... 0x1B 0x03) framing.
"""
import time
import serial

port = 'COM14'
baud = 115200

print(f"Connecting to {port} @ {baud}...")
ser = serial.Serial(port, baud, timeout=1.0)
ser.dtr = True
ser.rts = True

# Artinis standard binary frame structure: [ESC STX cmd len data ... CRC ESC ETX]
# ESC = 0x1B, STX = 0x02, ETX = 0x03
def send_artinis_packet(cmd_id, payload=b""):
    frame = bytearray([0x1B, 0x02, cmd_id, len(payload)])
    frame.extend(payload)
    # Checksum: sum of bytes modulo 256
    chk = sum(frame[2:]) & 0xFF
    frame.append(chk)
    frame.extend([0x1B, 0x03])
    ser.write(frame)
    ser.flush()
    print(f"[*] Sent Artinis Packet (Cmd 0x{cmd_id:02X}): {frame.hex(' ')}")
    time.sleep(0.3)
    if ser.in_waiting > 0:
        resp = ser.read(ser.in_waiting)
        print(f"    [RESPONSE] ({len(resp)} bytes): {resp.hex(' ')}")
        return resp
    return None

print("\nProbing Artinis Command Set...")
# 0x01 = Get Info, 0x02 = Get Status, 0x05 = Set Config, 0x06 = Start Stream, 0x08 = Start Measurement
for cmd in [0x01, 0x02, 0x03, 0x05, 0x06, 0x08, 0x10, 0x20, 0x53]:
    send_artinis_packet(cmd)

print("\n[*] Listening for continuous incoming data for 4 seconds...")
t0 = time.time()
total_bytes = 0
while time.time() - t0 < 4.0:
    if ser.in_waiting > 0:
        d = ser.read(ser.in_waiting)
        total_bytes += len(d)
        print(f"Live stream ({len(d)} bytes): {d.hex(' ')}")
    time.sleep(0.1)

print(f"\nTotal live bytes captured: {total_bytes}")
ser.close()
