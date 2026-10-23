# LSL Network & Firewall Setup

Every device in the suite (g.Nautilus EEG, smartwatch, fNIRS, thermal camera, task markers) publishes a Lab Streaming Layer (LSL) stream, and the recorders find these streams automatically. Discovery uses UDP multicast and broadcast. Most firewalls block both by default, **and the failure is silent**: the bridges run, but the recorder only sees some streams or none.

Do this once per machine before your first recording session.

---

## Ports LSL uses

| Protocol | Ports | Purpose |
|---|---|---|
| UDP | 16571 | Stream discovery (multicast `239.255.172.215`, IPv6 `ff02::/ff05::`, and broadcast) |
| UDP | 16572–16636 | Time synchronisation |
| TCP | 16572–16636 | Data transfer |

Allow **inbound** traffic on these ports for **IPv4 and IPv6**. Rules limited to a source subnet (e.g. `from 192.168.2.0/24`) are not enough. Discovery packets are addressed to multicast groups, and LSL also sends them over IPv6.

## Linux

**ufw** (Ubuntu, CachyOS, Arch). Replace `enp6s0` with your network interface (`ip -br addr`), and repeat for `wlan0` if devices connect over Wi-Fi:

```bash
sudo ufw allow in on enp6s0 proto udp to any port 16571:16636
sudo ufw allow in on enp6s0 proto tcp to any port 16572:16636
```

**firewalld** (Fedora):

```bash
sudo firewall-cmd --permanent --add-port=16571-16636/udp --add-port=16572-16636/tcp
sudo firewall-cmd --reload
```

To confirm the firewall was the problem, look for blocked LSL packets in the kernel log:

```bash
journalctl -k --since "-10min" | grep "BLOCK.*DPT=16571"
```

## Windows

The first time Python opens an LSL stream, Windows asks whether to allow it. Tick **Private networks** and click Allow. Also make sure the lab network is set to *Private*, not *Public*: Settings → Network & Internet → your connection → Network profile type.

To add the rules up front instead (run in an Administrator PowerShell):

```powershell
netsh advfirewall firewall add rule name="LSL UDP" dir=in action=allow protocol=UDP localport=16571-16636 profile=private
netsh advfirewall firewall add rule name="LSL TCP" dir=in action=allow protocol=TCP localport=16572-16636 profile=private
```

## macOS

When asked "Do you want the application python to accept incoming network connections?", click **Allow**. If you clicked Deny earlier, change it under System Settings → Network → Firewall → Options.

---

## Check that it works

The two outlets below are created in the same process. A blocked firewall typically lets only the last one through, so this catches the problem even on a single machine:

```bash
cd scripts
uv run python -c "
from pylsl import StreamInfo, StreamOutlet, resolve_streams
import time
a = StreamOutlet(StreamInfo('Check_A', 'Test', 1, 0, 'float32', 'check_a'))
b = StreamOutlet(StreamInfo('Check_B', 'Test', 1, 0, 'float32', 'check_b'))
time.sleep(1)
print(sorted(s.name() for s in resolve_streams(2) if s.name().startswith('Check_')))"
```

Expected output: `['Check_A', 'Check_B']`. If you only see `['Check_B']`, the firewall is still blocking discovery.

For a real session, start the bridges and check that the recorder lists every stream, e.g. both `Smartwatch_IMU` and `Smartwatch_PPG` for the watch:

```bash
uv run python bridges/smartwatch_lsl_bridge.py --mock   # terminal 1
uv run python -c "
import sys; sys.path.insert(0, 'recorders')
from multimodal_bids_recorder import MultimodalBIDSRecorder
MultimodalBIDSRecorder().discover_and_connect_streams(timeout=3)"   # terminal 2
```

## Other network gotchas

- **Same network:** all machines, the watch and the phone must be on the same LAN/subnet. Guest Wi-Fi and most university networks (eduroam) isolate clients and block multicast, so use a dedicated router or a phone hotspot.
- **VPNs** (WireGuard, corporate VPNs) can capture multicast routes. If discovery fails with the firewall open, check `ip route get 239.255.172.215` (Linux): it should go through your LAN interface.
- **Watch timestamps:** the smartwatch bridge timestamps samples with the watch's own sensor clock. That only works with a watch app build that sends the 11th CSV field (`timestampNs`); older builds fall back to arrival time, which is up to tens of milliseconds late and varies from packet to packet. Redeploy the app from `integrations/galaxy_watch_lsl` if in doubt.
- **Recorders apply clock sync:** `multimodal_bids_recorder.py` and `bids_recorder.py` open inlets with `proc_clocksync`, so streams from other machines are converted to the recording PC's clock. If you write your own recorder, do the same.
