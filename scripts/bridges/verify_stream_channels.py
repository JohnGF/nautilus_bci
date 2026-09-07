import time
from pylsl import resolve_streams, StreamInlet

print("Resolving fNIRS / Artinis LSL streams...")
streams = resolve_streams(wait_time=2.5)
target = None
for s in streams:
    if 'FNIRS' in s.type().upper() or 'ARTINIS' in s.name().upper():
        target = s
        break

if not target:
    print("[-] No active fNIRS LSL stream found on network.")
    exit(0)

inlet = StreamInlet(target)
info = inlet.info()

print(f"\n=======================================================")
print(f" [+] LSL Stream Metadata Validation")
print(f"=======================================================")
print(f" Stream Name:    {info.name()}")
print(f" Stream Type:    {info.type()}")
print(f" Total Channels: {info.channel_count()}")
print(f" Sampling Rate:  {info.nominal_srate()} Hz")
print(f" Source ID:      {info.source_id()}")
print(f"\nChannel Breakdown (2 Receivers x 4 Transmitters x 2 Wavelengths = 16 Channels):")
print("-" * 55)

ch_elem = info.desc().child("channels").child("channel")
ch_idx = 1
while not ch_elem.empty():
    label = ch_elem.child_value("label")
    unit = ch_elem.child_value("unit")
    ch_type = ch_elem.child_value("type")
    
    receiver = "Receiver 1 (Left Pod)" if ch_idx <= 8 else "Receiver 2 (Right Pod)"
    print(f" [{ch_idx:02d}] {label:20s} | {receiver} | Unit: {unit}")
    ch_elem = ch_elem.next_sibling()
    ch_idx += 1

print("\nSampling real data chunk (10 samples)...")
chunk, timestamps = inlet.pull_chunk(timeout=1.5, max_samples=10)
if chunk:
    print(f"[+] Successfully captured chunk with {len(chunk)} samples x {len(chunk[0])} channels!")
    print(f"    Left Receiver (R1) Optical Counts:  {chunk[-1][:4]}")
    print(f"    Right Receiver (R2) Optical Counts: {chunk[-1][8:12]}")
else:
    print("[-] No data samples received within timeout.")
