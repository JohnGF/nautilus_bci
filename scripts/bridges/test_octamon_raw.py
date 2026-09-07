"""
Artinis OctaMon Hardware Sniffer & Diagnostic Logger
Connects to COM14 and prints any raw bytes received from the device.
"""
import time
import serial

port = 'COM14'
baud = 115200

print(f"Connecting to {port} @ {baud}...")
try:
    ser = serial.Serial(port, baud, timeout=1.0)
    ser.dtr = True
    ser.rts = True
    print("[+] Port opened successfully!")
except Exception as e:
    print(f"[-] Error opening {port}: {e}")
    exit(1)

commands = [
    ("Identify (i)", b"i\r\n"),
    ("Version (v)", b"v\r\n"),
    ("Status (?)", b"?\r\n"),
    ("Start (s)", b"s\r\n"),
    ("Start (START)", b"START\r\n"),
    ("Start (M)", b"M\r\n"),
    ("Acquire (A)", b"A\r\n"),
    ("Binary Sync", bytes([0x00, 0x01, 0x02, 0x03])),
]

print("\nSending probe commands to OctaMon...")
for label, cmd in commands:
    ser.reset_input_buffer()
    print(f"\n[*] Sending: {label} -> {cmd}")
    ser.write(cmd)
    ser.flush()
    time.sleep(0.4)
    
    in_wait = ser.in_waiting
    print(f"    Bytes waiting: {in_wait}")
    if in_wait > 0:
        data = ser.read(in_wait)
        print(f"    [DATA RECEIVED] Hex: {data.hex(' ')}")
        try:
            print(f"    [DATA RECEIVED] ASCII: {repr(data.decode('ascii', errors='replace'))}")
        except Exception:
            pass

print("\n[*] Listening for incoming data stream for 3 seconds...")
start_t = time.time()
while time.time() - start_t < 3.0:
    if ser.in_waiting > 0:
        d = ser.read(ser.in_waiting)
        print(f"Received ({len(d)} bytes): {d.hex(' ')}")
    time.sleep(0.1)

ser.close()
print("\nScan completed.")
