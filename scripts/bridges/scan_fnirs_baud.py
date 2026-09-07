"""
fNIRS Battery Stream Deep Sniffer & Protocol Analyzer
Tests DTR/RTS hardware flow control, extended wait times, and protocol handshake variants.
"""
import sys
import time
import serial
import serial.tools.list_ports

def deep_scan_port(port):
    bauds = [115200, 57600, 38400, 9600, 19200, 230400, 460800, 921600]
    
    # Extended commands for g.SENSOR fNIRS / Artinis controllers
    probe_commands = [
        ("Passive Listen (1.5s)", None),
        ("CRLF", b"\r\n"),
        ("Wake 'a'", b"a\r\n"),
        ("Wake 's'", b"s\r\n"),
        ("Start Command", b"START\r\n"),
        ("start Command (lowercase)", b"start\r\n"),
        ("ASCII 'G'", b"G\r\n"),
        ("ASCII 'v' (version)", b"v\r\n"),
        ("Binary 0x01", bytes([0x01])),
        ("Binary START Header", bytes([0x02, 0x53, 0x54, 0x41, 0x52, 0x54, 0x03])),
        ("Brite/OctaMon Sync", bytes([0x00, 0x01, 0x02, 0x03])),
        ("Artinis DTR Toggle", b"DTR"),
    ]

    print(f"\n" + "=" * 65)
    print(f" Deep Scanning Active Port: {port}")
    print("=" * 65)

    for b in bauds:
        try:
            # Open with both RTS and DTR asserted (standard for Bluetooth serial peripherals)
            ser = serial.Serial(
                port=port,
                baudrate=b,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.6,
                write_timeout=0.6,
                rtscts=False,
                dsrdtr=False
            )
            ser.dtr = True
            ser.rts = True
        except Exception as e:
            print(f"[-] Cannot open {port} @ {b}: {e}")
            return False

        print(f"[*] Probing {port} @ {b} baud (DTR/RTS asserted)...")
        for desc, cmd in probe_commands:
            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()

                if cmd == b"DTR":
                    ser.dtr = False
                    time.sleep(0.05)
                    ser.dtr = True
                elif cmd is not None:
                    ser.write(cmd)
                    ser.flush()

                time.sleep(0.3)
                incoming = ser.read(128)

                if len(incoming) > 0:
                    print(f"\n[🎉 SUCCESS!] RECEIVED {len(incoming)} BYTES on {port} @ {b} baud!")
                    print(f"     Trigger: '{desc}'")
                    print(f"     Hex Dump: {incoming[:48].hex(' ')}")
                    try:
                        print(f"     ASCII String: {incoming[:48].decode('ascii', errors='ignore')}")
                    except Exception:
                        pass
                    ser.close()
                    return (port, b, cmd, incoming)
            except Exception:
                pass

        ser.close()
    return False

if __name__ == '__main__':
    print("fNIRS Battery Protocol Analyzer Running...")
    ports = ['COM7', 'COM8']
    
    found = None
    for p in ports:
        res = deep_scan_port(p)
        if res:
            found = res
            break

    if not found:
        print("\n[!] No responses received with standard triggers.")
        print("[*] Next diagnostic: verifying if Bluetooth incoming vs outgoing port pairing is needed.")
