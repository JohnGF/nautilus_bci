"""Unit tests for the W2 wearable protocol: uv run pytest bridges/test_wearable_protocol.py"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from wearable_protocol import (DeviceClockMapper, DeviceRegistry, format_packet,  # noqa: E402
                               parse_packet)


class FakeOutlet:
    def __init__(self, name, stream_type, source_id, srate, channels, device_id, location, channel_format="float32"):
        self.name, self.type, self.source_id, self.srate = name, stream_type, source_id, srate
        self.channel_format = channel_format
        self.channels = channels
        self.pushed = []

    def push_sample(self, values, stamp):
        self.pushed.append((values, stamp))


def make_registry():
    outlets = []

    def factory(*args, **kwargs):
        outlets.append(FakeOutlet(*args, **kwargs))
        return outlets[-1]

    return DeviceRegistry(outlet_factory=factory, log=lambda *_: None), outlets


def test_parse_imu_batch():
    p = parse_packet("W2|wear-ab12|L_wrist|7|IMU@50|2000000000,1,2,3,4,5,6;1980000000,0,0,9.8,0,0,0")
    assert (p.device_id, p.location, p.seq, p.kind, p.srate) == ("wear-ab12", "L_wrist", 7, "IMU", 50.0)
    # Samples come back oldest first even if the device sent them out of order
    assert [t for t, _ in p.samples] == pytest.approx([1.98, 2.0])
    assert p.samples[1][1] == [1, 2, 3, 4, 5, 6]


def test_format_roundtrip():
    text = format_packet("esp-01", "R_ankle", 3, "IMU", [(10, [0.5, -1.0, 9.81, 0, 0, 0.25])], 100.0)
    p = parse_packet(text)
    assert p.kind == "IMU" and p.srate == 100.0
    assert p.samples[0][1] == pytest.approx([0.5, -1.0, 9.81, 0, 0, 0.25])


def test_non_w2_is_left_to_legacy_parser():
    assert parse_packet("1.0,2.0,3.0,4.0,5.0,6.0,70.0,-1.0,-1.0,0,123") is None
    assert parse_packet("HB,1200,3,87,123456789012345") is None


@pytest.mark.parametrize("bad", [
    "W2|dev|L_wrist|1|IMU",                      # missing body
    "W2|dev with space|L_wrist|1|IMU|1,2",       # bad id
    "W2|dev|L_wrist|x|IMU|1,2",                  # bad seq
    "W2|dev|L_wrist|1|IMU|1,2,3;4,5",            # ragged samples
    "W2|dev|L_wrist|1|IMU|1",                    # timestamp without values
])
def test_malformed_w2_raises(bad):
    with pytest.raises(ValueError):
        parse_packet(bad)


def test_each_device_gets_its_own_streams_and_clock():
    registry, outlets = make_registry()
    # Same device time, very different clocks: each must map to its own arrival
    registry.handle(parse_packet("W2|a|L_wrist|0|IMU@50|1000000000,0,0,0,0,0,0"), arrival=100.0)
    registry.handle(parse_packet("W2|b|L_ankle|0|IMU@50|1000000000,0,0,0,0,0,0"), arrival=500.0)
    assert [o.name for o in outlets] == ["IMU_L_wrist", "IMU_L_ankle"]
    assert [o.source_id for o in outlets] == ["a_IMU", "b_IMU"]
    assert outlets[0].pushed[0][1] == pytest.approx(100.0)
    assert outlets[1].pushed[0][1] == pytest.approx(500.0)


def test_batch_keeps_sample_spacing():
    registry, outlets = make_registry()
    registry.handle(parse_packet("W2|a|L_wrist|0|IMU|0,1,1,1,1,1,1;20000000,2,2,2,2,2,2"), arrival=10.0)
    stamps = [s for _, s in outlets[0].pushed]
    assert stamps == pytest.approx([9.98, 10.0])


def test_kinds_and_generic_channels():
    registry, outlets = make_registry()
    registry.handle(parse_packet("W2|a|L_wrist|0|HR|0,72"), arrival=1.0)
    registry.handle(parse_packet("W2|a|L_wrist|1|EMG@1000|0,0.1,0.2"), arrival=1.0)
    assert (outlets[0].name, outlets[0].type) == ("HR_L_wrist", "PPG")
    assert (outlets[1].type, [c[0] for c in outlets[1].channels]) == ("EMG", ["ch1", "ch2", "DeviceTime"])


def test_data_streams_carry_raw_device_time():
    registry, outlets = make_registry()
    registry.handle(parse_packet("W2|a|L_wrist|0|IMU|1500000000,1,2,3,4,5,6"), arrival=10.0)
    assert outlets[0].channel_format == "double64"
    values, stamp = outlets[0].pushed[0]
    assert values == [1, 2, 3, 4, 5, 6, 1.5]
    assert stamp == pytest.approx(10.0)


def test_heartbeat_is_status_not_stream():
    registry, outlets = make_registry()
    registry.handle(parse_packet("W2|a|L_wrist|0|HB|0,100,0,87"), arrival=1.0)
    assert outlets == []
    assert registry.devices["a"].battery == 87


def test_duplicate_location_gets_distinct_names():
    registry, outlets = make_registry()
    registry.handle(parse_packet("W2|a|L_wrist|0|HR|0,72"), arrival=1.0)
    registry.handle(parse_packet("W2|b|L_wrist|0|HR|0,72"), arrival=1.0)
    assert [o.name for o in outlets] == ["HR_L_wrist", "HR_L_wrist_b"]


def test_sequence_gaps_count_as_loss_but_restarts_do_not():
    registry, _ = make_registry()
    for seq in (0, 1, 4):  # 2 and 3 lost
        registry.handle(parse_packet(f"W2|a|L_wrist|{seq}|HR|0,72"), arrival=1.0)
    assert registry.devices["a"].lost == 2
    registry.handle(parse_packet("W2|a|L_wrist|0|HR|0,72"), arrival=1.0)  # device restarted
    assert registry.devices["a"].lost == 2
    registry.handle(parse_packet(f"W2|a|L_wrist|{2**32 - 1}|HR|0,72"), arrival=1.0)
    registry.handle(parse_packet("W2|a|L_wrist|0|HR|0,72"), arrival=1.0)  # wraparound
    assert registry.devices["a"].lost == 2


def test_clock_mapper_takes_fastest_packet_in_window():
    clock = DeviceClockMapper(window_s=10.0)
    assert clock.update(0.0, arrival=1.05) == pytest.approx(1.05)
    assert clock.update(1.0, arrival=2.01) == pytest.approx(1.01)   # faster packet wins
    assert clock.update(2.0, arrival=3.20) == pytest.approx(1.01)   # slow packet ignored
    assert clock.update(20.0, arrival=21.03) == pytest.approx(1.03)  # old minimum aged out


def test_echo_publishes_raw_exchanges_and_takes_over_timing():
    registry, outlets = make_registry()
    addr = ("10.0.0.5", 40000)
    registry.handle(parse_packet("W2|a|L_wrist|0|HR|0,72"), arrival=100.0, addr=addr)
    dev = registry.devices["a"]
    # Device clock reads 0 when LSL reads 1000; every leg takes exactly 2 ms
    now = 1000.0
    for _ in range(40):
        ping_id = int(registry.next_ping("a", now).decode().split("|")[1])
        t2 = now + 0.002 - 1000.0
        registry.handle(parse_packet(f"W2|a|L_wrist|1|ECHO|{round(t2 * 1e9)},{ping_id},100"),
                        arrival=now + 0.0041, addr=addr)
        now += 0.1
    sync_outlet = next(o for o in outlets if o.type == "ClockSync")
    assert (sync_outlet.name, sync_outlet.channel_format) == ("ClockSync_L_wrist", "double64")
    assert len(sync_outlet.pushed) == 40
    assert dev.sync.synced
    # A data packet with a terrible arrival time is still placed by the ping/echo fit
    registry.handle(parse_packet("W2|a|L_wrist|2|IMU|5000000000,0,0,0,0,0,0"), arrival=2000.0)
    imu = next(o for o in outlets if o.type == "IMU")
    assert imu.pushed[0][1] == pytest.approx(1005.0, abs=1e-6)


def test_late_packets_are_not_counted_as_lost():
    registry, _ = make_registry()
    for seq in (5, 7, 6, 8):  # 6 arrives after 7: reordered, nothing lost
        registry.handle(parse_packet(format_packet("a", "L_wrist", seq, "IMU", [(seq, [0.0] * 6)])))
    assert registry.devices["a"].lost == 0
    registry.handle(parse_packet(format_packet("a", "L_wrist", 10, "IMU", [(10, [0.0] * 6)])))
    assert registry.devices["a"].lost == 1  # 9 really is missing


def test_beacon_roundtrip_and_hostname_sanitising():
    from wearable_protocol import BEACON_PORT, format_beacon, parse_beacon
    assert BEACON_PORT == 5006
    assert format_beacon("lab-pc", 5005) == "W2B|lab-pc|5005|1"
    assert parse_beacon(format_beacon("my|odd host", 5005)) == ("my-odd-host", 5005)
    assert parse_beacon("W2P|3") is None and parse_beacon("W2B|pc|x") is None and parse_beacon("W2B|pc|70000") is None
