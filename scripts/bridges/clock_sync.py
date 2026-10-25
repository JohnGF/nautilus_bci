"""
Ping/echo clock synchronization for wearables that do not run liblsl
====================================================================

LSL synchronizes clocks between machines running liblsl, but a watch or an
ESP32 talks plain UDP to the bridge, so the device -> bridge hop is outside
LSL. This closes that gap the way NTP (and LSL's own time_correction) does:

    bridge  t1 --- "W2P|<id>" --->  t2  device
    bridge  t4 <--- ECHO(t2,t3) ---  t3  device

    offset = ((t1 - t2) + (t4 - t3)) / 2     (LSL time minus device time)
    delay  = (t4 - t1) - (t3 - t2)           (network round trip)

The offset is exact if the two legs take equally long, and wrong by at most
delay / 2 if they do not. Wi-Fi delay varies a lot (power save, queueing), so
each burst of pings keeps only its fastest exchange. Device and PC crystals
also disagree by tens of ppm (36-180 ms/hour), so the bursts repeat every few
seconds and a straight line (offset + drift) is fitted through the recent ones.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

PING_PREFIX = "W2P|"
ECHO_KIND = "ECHO"
INITIAL_PINGS = 40      # first burst: a good offset within a few seconds
BURST_PINGS = 5
BURST_PERIOD_S = 10.0
PING_TIMEOUT_S = 2.0
FIT_WINDOW = 30         # bursts in the drift fit, ~5 min
MAX_SILENT_BURSTS = 3   # stop pinging devices that never answer (older firmware)
PING_ID_MOD = 1 << 24   # ids travel as float32 on the device side; exact below 2^24


@dataclass
class Exchange:
    t1: float  # LSL time the ping left the bridge
    t2: float  # device time the ping arrived
    t3: float  # device time the echo left
    t4: float  # LSL time the echo arrived

    @property
    def offset(self):
        return ((self.t1 - self.t2) + (self.t4 - self.t3)) / 2

    @property
    def delay(self):
        return (self.t4 - self.t1) - (self.t3 - self.t2)


def format_ping(ping_id):
    return f"{PING_PREFIX}{ping_id}"


class DeviceSync:
    """Offset and drift between one device's clock and LSL time, from ping/echo bursts."""

    def __init__(self):
        self.next_id = 0
        self.burst = 0
        self.pings_left = INITIAL_PINGS
        self.next_burst_at = None
        self.pending = {}       # ping id -> (t1, burst)
        self.closed_upto = -1   # bursts up to this one are folded into points
        self.burst_best = {}    # burst -> fastest Exchange so far
        self.points = deque(maxlen=FIT_WINDOW)  # fastest Exchange of each closed burst
        self.silent_bursts = 0
        self.slope = 0.0        # drift, seconds per second (1e-6 = 1 ppm)
        self.intercept = None   # offset at x_ref
        self.x_ref = 0.0

    @property
    def synced(self):
        return self.intercept is not None

    @property
    def gave_up(self):
        return not self.synced and self.silent_bursts >= MAX_SILENT_BURSTS

    def offset(self, device_s):
        """LSL time minus device time at device time device_s; None until the first burst closes."""
        if self.intercept is None:
            return None
        return self.intercept + self.slope * (device_s - self.x_ref)

    def error_bound(self):
        """Worst case for the newest burst: half its fastest round trip."""
        return self.points[-1].delay / 2 if self.points else None

    def next_ping(self, now):
        """Return a ping id to send now, or None. Call right after a packet from the device
        arrives: its radio has just transmitted and is awake, so the ping is not held back
        until the next Wi-Fi power-save beacon."""
        self._close_finished_bursts(now)
        if self.gave_up:
            return None
        if self.pings_left == 0:
            if self.next_burst_at is None or now < self.next_burst_at:
                return None
            self.burst += 1
            self.pings_left = BURST_PINGS
        ping_id = self.next_id
        self.next_id = (self.next_id + 1) % PING_ID_MOD
        self.pending[ping_id] = (now, self.burst)
        self.pings_left -= 1
        if self.pings_left == 0:
            self.next_burst_at = now + BURST_PERIOD_S
        return ping_id

    def add_echo(self, ping_id, t2, t3, t4):
        """Record one echo; returns the Exchange, or None if the ping is unknown or timed out."""
        sent = self.pending.pop(ping_id, None)
        if sent is None:
            return None
        t1, burst = sent
        ex = Exchange(t1, t2, t3, t4)
        best = self.burst_best.get(burst)
        if best is None or ex.delay < best.delay:
            self.burst_best[burst] = ex
        self._close_finished_bursts(t4)
        return ex

    def _close_finished_bursts(self, now):
        """Close bursts in order once all their pings are sent and each is answered or timed out."""
        self.pending = {i: (t1, b) for i, (t1, b) in self.pending.items() if now - t1 <= PING_TIMEOUT_S}
        waiting = {b for _, b in self.pending.values()}
        fully_sent = self.burst if self.pings_left == 0 else self.burst - 1
        while self.closed_upto < fully_sent and self.closed_upto + 1 not in waiting:
            self.closed_upto += 1
            best = self.burst_best.pop(self.closed_upto, None)
            if best is None:
                self.silent_bursts += 1
                continue
            self.silent_bursts = 0
            self.points.append(best)
            self._fit()

    def _fit(self):
        x = np.array([p.t2 for p in self.points])
        y = np.array([p.offset for p in self.points])
        self.x_ref = x[-1]
        if len(x) < 3 or x[-1] - x[0] < 1.0:
            # Too little spread for a slope; keep the previous drift estimate
            self.intercept = y[-1]
            return
        self.slope, self.intercept = np.polyfit(x - self.x_ref, y, 1)
