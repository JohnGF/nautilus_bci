"""
Wearable wire protocol v2 (W2) and the per-device LSL fan-out
==============================================================

Any number of wearables (Wear OS watches, ESP32 boards, ...) send UDP packets to
one bridge. Each device gets its own LSL outlets and its own clock mapping, so
both wrists and both ankles can be recorded at once. Spec: docs/guides/wearable_protocol.md

    W2|<device_id>|<location>|<seq>|<kind>[@<rate_hz>]|<t_ns>,<v1>,<v2>,...;<t_ns>,...

This module has no socket code so it can be unit tested without a network.
"""
import re
import time
from collections import deque
from dataclasses import dataclass, field

from pylsl import StreamInfo, StreamOutlet, local_clock

from clock_sync import ECHO_KIND, DeviceSync, format_ping

TAG = "W2"
STATUS_KIND = "HB"  # heartbeat: sent, dropped, battery_pct; status only, not an LSL stream
# Appended to every data stream: the sample's raw device timestamp, so a recording can redo the
# clock fit offline against the ClockSync_* exchanges. double64, since float32 would round it to ~10 ms.
DEVICE_TIME_CHANNEL = ("DeviceTime", "s_device", "DeviceClock")
SYNC_CHANNELS = [("t1_ping_sent", "s_lsl"), ("t2_ping_received", "s_device"),
                 ("t3_echo_sent", "s_device"), ("t4_echo_received", "s_lsl")]
STALE_AFTER_S = 3.0
SEQ_MOD = 2 ** 32
SEQ_RESTART_GAP = 1000  # a jump this large means the device restarted, not that packets were lost
SEQ_MISSING_MAX = 256  # remember this many recently missed seqs, so late UDP arrivals can be un-counted

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

# kind -> (LSL type, [(label, unit)]). Kinds not listed get generic ch1..chN channels.
KNOWN_KINDS = {
    "IMU": ("IMU", [("Accel_X", "m/s2"), ("Accel_Y", "m/s2"), ("Accel_Z", "m/s2"),
                    ("Gyro_X", "rad/s"), ("Gyro_Y", "rad/s"), ("Gyro_Z", "rad/s")]),
    "HR": ("PPG", [("HeartRate", "bpm")]),
    "TOUCH": ("Touch", [("Touch_X", "px"), ("Touch_Y", "px"), ("Touched", "bool")]),
}


class DeviceClockMapper:
    """Maps device timestamps (seconds) onto LSL local_clock().

    offset = arrival - device_time is transport delay plus the clock difference.
    The minimum over a sliding window is the fastest packet seen, so it strips
    network jitter; the window lets it follow slow clock drift between devices.
    What remains is the constant minimum one-way latency, which a per-device
    sync check (e.g. a tap seen in both EEG and IMU) can measure.
    """

    def __init__(self, window_s=10.0):
        self.window_s = window_s
        self.offsets = deque()  # (arrival, offset), offsets kept increasing

    def update(self, device_s, arrival=None):
        """Feed one (device time, arrival) pair; returns the current device-to-LSL offset."""
        arrival = local_clock() if arrival is None else arrival
        offset = arrival - device_s
        # Monotonic deque: anything with a larger offset can never be the window minimum again
        while self.offsets and self.offsets[-1][1] >= offset:
            self.offsets.pop()
        self.offsets.append((arrival, offset))
        while arrival - self.offsets[0][0] > self.window_s:
            self.offsets.popleft()
        return self.offsets[0][1]

    def map(self, device_s, arrival=None):
        return device_s + self.update(device_s, arrival)


@dataclass
class Packet:
    device_id: str
    location: str
    seq: int
    kind: str
    srate: float
    samples: list  # [(device_time_s, [values])], oldest first


def parse_packet(text):
    """Parse one W2 packet. Returns None if it is not W2; raises ValueError if it is W2 but malformed."""
    if not text.startswith(TAG + "|"):
        return None
    parts = text.strip().split("|")
    if len(parts) != 6:
        raise ValueError(f"expected 6 '|' fields, got {len(parts)}")
    _, device_id, location, seq, kind, body = parts
    if not _ID_RE.match(device_id) or not _ID_RE.match(location):
        raise ValueError("device_id and location must be 1-32 chars of [A-Za-z0-9_-]")

    srate = 0.0
    if "@" in kind:
        kind, rate = kind.split("@", 1)
        srate = float(rate)
    if not _ID_RE.match(kind):
        raise ValueError(f"bad kind {kind!r}")

    samples = []
    width = None
    for chunk in body.split(";"):
        if not chunk:
            continue
        fields = chunk.split(",")
        if width is None:
            width = len(fields)
        elif len(fields) != width:
            raise ValueError("samples in one packet must have the same channel count")
        samples.append((int(fields[0]) * 1e-9, [float(v) for v in fields[1:]]))
    if not samples or width < 2:
        raise ValueError("packet has no samples")
    samples.sort(key=lambda s: s[0])
    return Packet(device_id, location, int(seq), kind, srate, samples)


def format_packet(device_id, location, seq, kind, samples, srate=0.0):
    """Inverse of parse_packet; samples are [(t_ns, [values])]. Used by the mock and tests."""
    head = f"{kind}@{srate:g}" if srate else kind
    body = ";".join(",".join([str(t)] + [f"{v:g}" for v in values]) for t, values in samples)
    return f"{TAG}|{device_id}|{location}|{seq % SEQ_MOD}|{head}|{body}"


def make_stream_info(name, stream_type, source_id, srate, channels, device_id, location, channel_format="float32"):
    info = StreamInfo(name=name, type=stream_type, channel_count=len(channels),
                      nominal_srate=srate, channel_format=channel_format, source_id=source_id)
    desc = info.desc()
    desc.append_child_value("device_id", device_id)
    desc.append_child_value("location", location)
    desc.append_child_value("protocol", TAG)
    desc.append_child_value("clock_sync", "ping/echo offset+drift fit (see ClockSync_* stream); "
                                          "min one-way offset until the first burst or for devices without echo")
    chans = desc.append_child("channels")
    for label, unit, *ch_type in channels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", ch_type[0] if ch_type else stream_type)
    return info


def make_outlet(*args, **kwargs):
    return StreamOutlet(make_stream_info(*args, **kwargs))


@dataclass
class Device:
    device_id: str
    location: str
    stream_suffix: str
    clock: DeviceClockMapper = field(default_factory=DeviceClockMapper)  # fallback without echo
    sync: DeviceSync = field(default_factory=DeviceSync)
    sync_outlet: object = None
    addr: tuple = None
    outlets: dict = field(default_factory=dict)
    last_seq: int = None
    missing: set = field(default_factory=set)  # recently missed seqs (late arrivals un-count)
    packets: int = 0
    lost: int = 0
    battery: int = None
    last_seen: float = 0.0
    samples: dict = field(default_factory=dict)  # kind -> count since last status


class DeviceRegistry:
    """Creates outlets per (device, kind) on first sight and keeps one clock mapper per device."""

    def __init__(self, outlet_factory=make_outlet, log=print):
        self.outlet_factory = outlet_factory
        self.log = log
        self.devices = {}
        self._last_status = time.monotonic()

    def handle(self, packet, arrival=None, addr=None):
        arrival = local_clock() if arrival is None else arrival
        dev = self.devices.get(packet.device_id)
        if dev is None:
            dev = self._add_device(packet)
        elif packet.location != dev.location:
            self.log(f"[!] {dev.device_id} now reports location {packet.location}; "
                     f"its streams keep the name {dev.stream_suffix}")

        self._track_seq(dev, packet.seq)
        dev.packets += 1
        dev.last_seen = time.monotonic()
        if addr is not None:
            dev.addr = addr

        if packet.kind == ECHO_KIND:
            self._handle_echo(dev, packet, arrival)
            return

        # Offset from the newest sample: the one with the least time sitting in the batch
        fallback = dev.clock.update(packet.samples[-1][0], arrival)

        if packet.kind == STATUS_KIND:
            values = packet.samples[-1][1]
            if len(values) >= 3:
                dev.battery = int(values[2])
            return

        outlet = dev.outlets.get(packet.kind)
        if outlet is None:
            outlet = self._add_outlet(dev, packet)
        for t, values in packet.samples:
            offset = dev.sync.offset(t)
            outlet.push_sample(values + [t], t + (fallback if offset is None else offset))
        dev.samples[packet.kind] = dev.samples.get(packet.kind, 0) + len(packet.samples)

    def next_ping(self, device_id, now):
        """Ping payload to send to the device right now, or None. Call immediately before sendto,
        with now = local_clock(), since now becomes t1."""
        dev = self.devices.get(device_id)
        if dev is None or dev.addr is None:
            return None
        ping_id = dev.sync.next_ping(now)
        return None if ping_id is None else format_ping(ping_id).encode()

    def _handle_echo(self, dev, packet, arrival):
        t2, values = packet.samples[0]
        if len(values) < 2:
            return
        t3 = t2 + values[1] * 1e-6  # device turnaround in microseconds
        ex = dev.sync.add_echo(int(values[0]), t2, t3, arrival)
        if ex is None:
            return
        if dev.sync_outlet is None:
            # Raw exchanges, so the offset/drift fit can be redone offline over the whole session
            dev.sync_outlet = self.outlet_factory(
                f"ClockSync_{dev.stream_suffix}", "ClockSync", f"{dev.device_id}_ClockSync", 0.0,
                SYNC_CHANNELS, dev.device_id, dev.location, channel_format="double64")
            self.log(f"    LSL outlet 'ClockSync_{dev.stream_suffix}' (ping/echo exchanges)")
        dev.sync_outlet.push_sample([ex.t1, ex.t2, ex.t3, ex.t4], ex.t4)

    def _add_device(self, packet):
        taken = {d.location for d in self.devices.values()}
        suffix = packet.location
        if packet.location in taken:
            # Two devices configured for the same body site; keep both, but name them apart
            suffix = f"{packet.location}_{packet.device_id}"
            self.log(f"[!] Location {packet.location} already in use; streams for "
                     f"{packet.device_id} are named *_{suffix}")
        dev = Device(packet.device_id, packet.location, suffix)
        self.devices[packet.device_id] = dev
        self.log(f"[+] New device {packet.device_id} at {packet.location}")
        return dev

    def _add_outlet(self, dev, packet):
        n = len(packet.samples[0][1])
        stream_type, channels = KNOWN_KINDS.get(packet.kind, (packet.kind, []))
        if len(channels) != n:
            channels = [(f"ch{i + 1}", "") for i in range(n)]
        name = f"{packet.kind}_{dev.stream_suffix}"
        outlet = self.outlet_factory(name, stream_type, f"{dev.device_id}_{packet.kind}", packet.srate,
                                     channels + [DEVICE_TIME_CHANNEL], dev.device_id, dev.location,
                                     channel_format="double64")
        dev.outlets[packet.kind] = outlet
        self.log(f"    LSL outlet '{name}' ({stream_type}, {n} ch, {packet.srate:g} Hz)")
        return outlet

    @staticmethod
    def _track_seq(dev, seq):
        if seq in dev.missing:
            # Late UDP arrival: it was counted lost when a newer packet overtook it
            dev.missing.discard(seq)
            dev.lost -= 1
            return
        if dev.last_seq is not None:
            gap = (seq - dev.last_seq - 1) % SEQ_MOD
            if 0 < gap < SEQ_RESTART_GAP:
                dev.lost += gap
                if gap <= SEQ_MISSING_MAX:
                    dev.missing.update((dev.last_seq + 1 + k) % SEQ_MOD for k in range(gap))
                if len(dev.missing) > SEQ_MISSING_MAX:  # keep only the most recent gaps
                    dev.missing = set(sorted(dev.missing, key=lambda s: (seq - s) % SEQ_MOD)[:SEQ_MISSING_MAX])
        dev.last_seq = seq

    def status_lines(self):
        """One line per device with rates since the previous call; resets the rate counters."""
        now = time.monotonic()
        elapsed = max(now - self._last_status, 1e-6)
        self._last_status = now
        lines = []
        for dev in sorted(self.devices.values(), key=lambda d: d.location):
            rates = " ".join(f"{k} {c / elapsed:5.1f}Hz" for k, c in sorted(dev.samples.items()))
            dev.samples = {}
            total = dev.packets + dev.lost
            loss = 100.0 * dev.lost / total if total else 0.0
            batt = f"{dev.battery}%" if dev.battery is not None else "?"
            age = now - dev.last_seen
            state = "STALE" if age > STALE_AFTER_S else "ok"
            if dev.sync.synced:
                # Offset shrinks when the device clock runs fast; show fast as positive
                sync = f"sync ±{dev.sync.error_bound() * 1e3:.1f}ms drift {-dev.sync.slope * 1e6:+.0f}ppm"
            elif dev.sync.gave_up:
                sync = "sync: no echo (fallback)"
            else:
                sync = "sync: pending"
            lines.append(f"  {dev.location:<10} {dev.device_id:<14} {state:<5} {rates:<28} "
                         f"loss {loss:4.1f}%  batt {batt:>4}  seen {age:4.1f}s ago  {sync}")
        return lines
