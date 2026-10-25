"""Unit tests for ping/echo clock sync: uv run pytest bridges/test_clock_sync.py"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from clock_sync import (BURST_PERIOD_S, MAX_SILENT_BURSTS, DeviceSync,  # noqa: E402
                        Exchange)


def test_symmetric_legs_give_exact_offset():
    # Device clock = LSL - 50 s; 3 ms each way, 0.2 ms turnaround
    ex = Exchange(t1=100.0, t2=50.003, t3=50.0032, t4=100.0062)
    assert ex.offset == pytest.approx(50.0)
    assert ex.delay == pytest.approx(0.006)


def test_asymmetric_legs_stay_within_half_delay():
    ex = Exchange(t1=100.0, t2=50.010, t3=50.010, t4=100.012)  # 10 ms up, 2 ms down
    assert abs(ex.offset - 50.0) <= ex.delay / 2 + 1e-12


def simulate(sync, seconds, true_offset, drift, rng, answer=True):
    """Device sends 10 packets/s; each may trigger a ping over a jittery link (1-15 ms per leg).
    drift > 0 makes the device clock slow, so the offset (LSL - device) grows: slope = +drift."""
    now = 0.0
    while now < seconds:
        ping_id = sync.next_ping(now)
        if ping_id is not None and answer:
            up, down = rng.uniform(1e-3, 15e-3), rng.uniform(1e-3, 15e-3)
            device = lambda lsl: (lsl - true_offset) / (1 + drift)  # noqa: E731
            t2 = device(now + up)
            sync.add_echo(ping_id, t2, t2 + 2e-4, now + up + 2e-4 * (1 + drift) + down)
        now += 0.1
    return lambda lsl: (lsl - true_offset) / (1 + drift)


def test_recovers_offset_and_drift_through_jitter():
    rng = random.Random(1)
    sync = DeviceSync()
    device = simulate(sync, 300, true_offset=1234.5, drift=50e-6, rng=rng)
    assert sync.synced
    # Each burst point is off by ~1 ms on this link, so drift is only good to ~10 ppm over 5 min;
    # between 10 s refits that is 0.1 ms, far below the offset error itself
    assert sync.slope * 1e6 == pytest.approx(50, abs=15)
    # Mapping a device timestamp back to LSL lands within the reported (worst-case) bound
    for lsl in (100.0, 200.0, 299.0):
        err = device(lsl) + sync.offset(device(lsl)) - lsl
        assert abs(err) < 2e-3
        assert abs(err) <= sync.error_bound()


def test_first_burst_gives_offset_within_seconds():
    sync = DeviceSync()
    simulate(sync, 8, true_offset=10.0, drift=0.0, rng=random.Random(2))
    assert sync.synced


def test_silent_device_is_given_up_on():
    sync = DeviceSync()
    simulate(sync, 5 + (MAX_SILENT_BURSTS + 1) * BURST_PERIOD_S, 0.0, 0.0, random.Random(3), answer=False)
    assert sync.gave_up and not sync.synced
    assert sync.next_ping(10_000.0) is None


def test_unknown_echo_is_ignored():
    assert DeviceSync().add_echo(12345, 0.0, 0.0, 1.0) is None


def test_short_sessions_do_not_invent_drift():
    sync = DeviceSync()
    simulate(sync, 40, true_offset=5.0, drift=30e-6, rng=random.Random(4))
    assert sync.synced and sync.slope == 0.0  # 4 bursts over 30 s: too short to trust a slope
    assert abs(sync.offset(0.0) - 5.0) < sync.error_bound() + 1e-9
