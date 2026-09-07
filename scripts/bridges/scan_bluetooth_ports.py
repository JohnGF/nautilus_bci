import serial
import serial.tools.list_ports
import time

ports = [p.device for p in serial.tools.list_ports.comports()]
print(f"Available COM ports: {ports}")

for port in ['COM14', 'COM15', 'COM7', 'COM8']:
    print(f"\n--- Trying {port} ---")
    try:
        ser = serial.Serial(port, 115200, timeout=1.0, write_timeout=1.0)
        ser.dtr = True
        ser.rts = True
        print(f"[+] Successfully connected to {port}!")
        time.sleep(0.5)
        in_w = ser.in_waiting
        print(f"    Bytes in buffer: {in_w}")
        if in_w > 0:
            print("    Data:", ser.read(in_w).hex(' '))
        ser.close()
    except Exception as e:
        print(f"[-] {port} failed: {e}")
