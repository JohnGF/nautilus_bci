"""fNIRS visualizer: stream layout, MBLL physics, contact quality and the live display path."""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visualizers"))

from PySide6 import QtWidgets  # noqa: E402

import fnirs_visualizer as fv  # noqa: E402

BRIDGE_LABELS = [
    f"R{1 if tx <= 4 else 2}_T{tx}_{wavelength}"
    for tx in range(1, 9) for wavelength in ("760nm", "850nm")
]
SRATE = 50.0
DISTANCE_CM = fv.MONTAGES["FRONTAL"]["distance_cm"]


def planted_delta_od(hbo_um, hbr_um):
    """Forward Beer-Lambert (decadic) for a known concentration change."""
    path = DISTANCE_CM * fv.DEFAULT_DPF
    return tuple(
        (fv.EPSILON[wl]["hbo"] * hbo_um + fv.EPSILON[wl]["hbr"] * hbr_um) * 1e-6 * path
        for wl in (760, 850)
    )


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    win = fv.fNIRSVisualizerWindow()
    win.timer.stop()
    yield win
    win.close()


def test_bridge_labels_pair_each_transmitter_across_wavelengths():
    layout = fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16)
    assert layout.kind == fv.KIND_OD and layout.labelled
    assert layout.first == list(range(0, 16, 2))
    assert layout.second == list(range(1, 16, 2))


def test_older_block_labels_still_pair_correctly():
    labels = [f"Tx{i}_760nm" for i in range(1, 9)] + [f"Tx{i}_850nm" for i in range(1, 9)]
    layout = fv.parse_stream_layout(labels, ["OD"] * 16, 16)
    assert layout.first == list(range(8))
    assert layout.second == list(range(8, 16))


def test_montage_matches_manual_diagram():
    for zone in fv.ZONES:
        widget = fv.MONTAGES[zone]
        assert widget["rx"][0][0] > 0 > widget["rx"][1][0]  # R1 on the wearer's right, screen-right
    assert fv.MONTAGES["FRONTAL"]["distance_cm"] == 3.2
    assert fv.MONTAGES["CENTRAL"]["distance_cm"] == 3.0


def test_central_transmitters_sit_where_the_manual_draws_them(app):
    hm = fv.OptodeHeadMapWidget("CENTRAL")
    tx = np.array(hm.tx_positions)
    assert (tx[0] < hm.rx_positions[0]).all()  # T1: anterior-medial of R1
    assert tx[1][0] > hm.rx_positions[0][0] and tx[1][1] < 0  # T2: anterior-lateral
    assert (tx[2] > hm.rx_positions[0]).all()  # T3: posterior-lateral
    assert tx[5][0] < hm.rx_positions[1][0] and tx[5][1] < 0  # T6: anterior-lateral of R2
    assert tx[4][0] > hm.rx_positions[1][0] and tx[4][1] < 0  # T5: anterior-medial


def test_front_view_puts_t1_nearest_the_nose_at_the_bottom(app):
    hm = fv.OptodeHeadMapWidget("FRONTAL")
    assert hm.tx_positions[0][1] > hm.rx_positions[0][1] > hm.tx_positions[2][1]


def test_concentration_labels_are_recognised():
    labels = [f"Tx{i}_HbO" for i in range(1, 9)] + [f"Tx{i}_HbR" for i in range(1, 9)]
    layout = fv.parse_stream_layout(labels, ["micromol/L"] * 16, 16)
    assert layout.kind == fv.KIND_CONC
    assert layout.first[2] == 2 and layout.second[2] == 10


def test_unlabelled_stream_is_interleaved_and_skips_aux_columns():
    labels = [f"Ch{i}" for i in range(16)] + ["Battery"]
    layout = fv.parse_stream_layout(labels, [""] * 17, 17)
    assert not layout.labelled and layout.kind is None
    assert layout.first == list(range(0, 16, 2))
    assert layout.second == list(range(1, 16, 2))


@pytest.mark.parametrize("name,stream_type,expected", [
    ("Artinis_fNIRS", "fNIRS", True),
    ("gNautilus_fNIRS", "fNIRS", True),
    ("gNautilus_EEG", "EEG", False),
    ("gNautilus_fNIRS_Link", "Diagnostics", False),
])
def test_stream_selection(name, stream_type, expected):
    assert fv.is_fnirs_stream(name, stream_type) is expected


@pytest.mark.parametrize("kind", [fv.KIND_OD, fv.KIND_INTENSITY])
def test_mbll_recovers_planted_concentration(kind):
    d760, d850 = planted_delta_od(1.5, -0.5)
    if kind == fv.KIND_OD:
        base = np.array([[1.8]])
        first, second = base + d760, base + d850
    else:
        base = np.array([[2000.0]])
        first, second = base * 10 ** -d760, base * 10 ** -d850
    hbo, hbr = fv.to_concentrations(first, second, base[0], base[0], kind, fv.compute_mbll_matrix(DISTANCE_CM))
    assert hbo[0, 0] == pytest.approx(1.5, rel=1e-6)
    assert hbr[0, 0] == pytest.approx(-0.5, rel=1e-6)


def test_scalp_coupling_index_separates_pulse_from_noise(app):
    win_filters = fv.fNIRSVisualizerWindow()
    win_filters.timer.stop()
    win_filters.srate = SRATE
    win_filters.init_filters()
    sos = win_filters.sos_cardiac
    rng = np.random.default_rng(0)
    t = np.arange(1000) / SRATE
    pulse = np.sin(2 * np.pi * 1.2 * t)
    first = np.column_stack([pulse, rng.normal(size=1000), np.full(1000, 1.8)])
    second = np.column_stack([0.8 * pulse, rng.normal(size=1000), np.full(1000, 1.8)])
    sci = fv.scalp_coupling_index(first, second, sos, 250)
    assert sci[0] > 0.95
    assert sci[1] < 0.45
    assert sci[2] == 0.0
    assert np.isnan(fv.scalp_coupling_index(first[:100], second[:100], sos, 250)).all()
    win_filters.close()


def test_filter_does_not_pin_the_newest_sample_to_zero():
    sos = fv.signal.butter(3, [0.01 / 25, 0.20 / 25], btype='bandpass', output='sos')
    t = np.arange(1000) / SRATE
    peak_at_end = np.cos(2 * np.pi * 0.08 * (t - t[-1]))[:, None] + 0.3
    assert fv.zero_phase_filter(sos, peak_at_end)[-1, 0] > 0.6


def bridge_chunk(n_samples, active_channel=None, delta=(0.0, 0.0)):
    chunk = np.full((n_samples, 16), 0.001)
    if active_channel is not None:
        chunk[:, 2 * active_channel] += delta[0]
        chunk[:, 2 * active_channel + 1] += delta[1]
    return chunk


@pytest.mark.parametrize("view_index", [0, 1])
def test_activity_on_one_transmitter_reaches_only_its_channel(window, view_index):
    window.combo_view.setCurrentIndex(view_index)
    window.chk_filter.setChecked(False)
    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    window.push_samples(bridge_chunk(30))
    window.push_samples(bridge_chunk(30, active_channel=2, delta=planted_delta_od(1.0, 0.0)))
    window.poll_and_update()

    values = window.headmap_widget.channel_values
    assert values[2] == pytest.approx(1.0, rel=1e-6)
    assert np.delete(values, 2) == pytest.approx(np.zeros(7), abs=1e-9)


def test_reconnecting_recalibrates_the_baseline(window):
    layout = fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16)
    window.attach_stream(layout, 16, SRATE)
    window.push_samples(bridge_chunk(30))
    assert window.is_calibrated
    window.attach_stream(layout, 16, SRATE)
    assert not window.is_calibrated


def test_baseline_ignores_the_sdk_startup_plateau_even_within_one_chunk(window):
    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    window.push_samples(np.full((22, 16), 1.8))
    assert window.is_calibrated  # The plateau is flat, so it does look calibrated at first
    window.push_samples(bridge_chunk(30))
    assert window.baseline == pytest.approx(np.full(16, 0.001))
    assert window.sample_count == 30

    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    window.push_samples(np.vstack([np.full((10, 16), 1.8), bridge_chunk(20)]))
    assert window.baseline == pytest.approx(np.full(16, 0.001))


def test_level_jump_mid_stream_recalibrates_instead_of_offsetting_the_trace(window):
    window.chk_filter.setChecked(False)
    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    window.push_samples(bridge_chunk(60))
    window.push_samples(bridge_chunk(60) - 1.8)
    window.poll_and_update()
    assert window.headmap_widget.channel_values == pytest.approx(np.zeros(8), abs=1e-9)
    assert "baseline reset" in window.lbl_status.text()


def test_normal_pulse_steps_do_not_reset_the_baseline(window):
    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    window.push_samples(bridge_chunk(60))
    window.push_samples(bridge_chunk(60) + 0.1)  # Beyond any step measured on the head
    assert window.sample_count == 120


def test_concentration_streams_are_not_reset_by_large_values(window):
    labels = [f"Tx{i}_HbO" for i in range(1, 9)] + [f"Tx{i}_HbR" for i in range(1, 9)]
    window.attach_stream(fv.parse_stream_layout(labels, ["micromol/L"] * 16, 16), 16, SRATE)
    window.push_samples(np.zeros((20, 16)))
    window.push_samples(np.full((20, 16), 3.0))
    assert window.sample_count == 40


def test_bridge_holds_back_frames_until_the_baseline_takes_effect():
    from bridges.artinis_to_lsl import BaselineSettler

    settler = BaselineSettler()
    now = 100.0
    plateau = [settler.accept(np.full(16, 1.8), now + i * 0.02) for i in range(22)]
    assert not any(plateau)
    settled = [settler.accept(np.full(16, 0.01), now + 0.5 + i * 0.02) for i in range(BaselineSettler.FRAMES_NEEDED)]
    assert settled == [False] * (BaselineSettler.FRAMES_NEEDED - 1) + [True]
    assert not settler.timed_out
    assert settler.accept(np.full(16, 1.8), now + 2.0)  # Once settled, later frames always pass


def test_bridge_publishes_anyway_if_one_channel_never_settles():
    from bridges.artinis_to_lsl import BaselineSettler

    settler = BaselineSettler()
    bad_channel = np.zeros(16)
    bad_channel[5] = 1.0
    assert not settler.accept(bad_channel, 100.0)
    assert settler.accept(bad_channel, 100.0 + BaselineSettler.TIMEOUT_SEC)
    assert settler.timed_out


def test_startup_levels_keep_both_wavelengths():
    layout = fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16)
    values = [f"{1.0 + i / 100:.3f}" for i in range(16)]
    first, second = fv.startup_levels_per_channel(values, layout)
    assert first[0] == pytest.approx(1.00) and second[0] == pytest.approx(1.01)
    assert first[7] == pytest.approx(1.14) and second[7] == pytest.approx(1.15)


@pytest.mark.parametrize("values", [[""] * 16, ["1.0"] * 4, []])
def test_streams_without_startup_levels_report_none(values):
    layout = fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16)
    assert fv.startup_levels_per_channel(values, layout) is None


def test_low_light_is_only_a_note():
    # Levels from the real session after replugging: T5 stayed weak, the others worked
    first = np.array([1.675, 1.396, 1.058, 1.437, 1.783, 1.31, 1.189, 1.292])
    second = np.array([1.652, 1.294, 0.883, 1.338, 1.793, 1.146, 0.969, 1.131])
    notes = fv.channel_notes(first, second, np.full(8, 20.0))
    assert notes[4] == "low light at start"
    assert all(note == "" for ch, note in enumerate(notes) if ch != 4)


def pulse_fixture(sos_seed=0, n=1000):
    """5 channels with a 1.1 Hz pulse; 3 with noise shared across wavelengths but no pulse."""
    rng = np.random.default_rng(sos_seed)
    t = np.arange(n) / SRATE
    first = rng.normal(0, 0.0005, (n, 8))
    second = rng.normal(0, 0.0005, (n, 8))
    first[:, :5] += 0.003 * np.sin(2 * np.pi * 1.1 * t)[:, None]
    second[:, :5] += 0.0025 * np.sin(2 * np.pi * 1.1 * t)[:, None]
    shared = rng.normal(0, 0.002, (n, 3))
    first[:, 5:] += shared
    second[:, 5:] += shared
    return first, second


@pytest.fixture
def cardiac_sos():
    return fv.signal.butter(2, [0.8 / 25, 2.0 / 25], btype="bandpass", output="sos")


def test_pulse_metrics_find_the_shared_heart_rate(cardiac_sos):
    first, second = pulse_fixture()
    pulse = fv.pulse_metrics(first, second, SRATE, cardiac_sos)
    assert pulse.heart_hz == pytest.approx(1.1, abs=0.1)
    assert pulse.channels_with_pulse == 5
    assert (pulse.strength[:5] >= fv.PULSE_CLEAR_STRENGTH).all()
    assert (pulse.strength[5:] < fv.PULSE_CLEAR_STRENGTH).all()


def test_contact_score_discounts_correlated_noise_that_has_no_pulse(cardiac_sos):
    first, second = pulse_fixture()
    sci = fv.scalp_coupling_index(first, second, cardiac_sos, 250)
    assert sci[5:].min() > 0.85  # SCI alone is fooled by noise common to both wavelengths
    contact = fv.contact_score(sci, fv.pulse_metrics(first, second, SRATE, cardiac_sos))
    assert contact[:5].min() > 0.9
    assert contact[5:].max() < 0.8 * sci[5:].min()


def test_pulse_metrics_need_enough_data(cardiac_sos):
    first, second = pulse_fixture(n=100)
    pulse = fv.pulse_metrics(first, second, SRATE, cardiac_sos)
    assert np.isnan(pulse.heart_hz) and np.isnan(pulse.strength).all()


def two_channel_pulse(amplitude, channels=(1, 2), n=1000):
    rng = np.random.default_rng(11)
    first, second = rng.normal(0, 0.001, (n, 8)), rng.normal(0, 0.001, (n, 8))
    wave = amplitude * np.sin(2 * np.pi * 1.4 * np.arange(n) / SRATE)
    for ch in channels:
        first[:, ch] += wave
        second[:, ch] += 0.8 * wave
    return first, second


def test_two_strong_channels_on_one_pod_confirm_a_pulse_despite_a_dark_pod(cardiac_sos):
    first, second = two_channel_pulse(0.003)
    first[:, 4:] = np.round(first[:, 4:] / 4e-4) * 4e-4
    second[:, 4:] = np.round(second[:, 4:] / 4e-4) * 4e-4
    pulse = fv.pulse_metrics(first, second, SRATE, cardiac_sos)
    assert pulse.heart_hz == pytest.approx(1.4, abs=0.1)
    assert pulse.channels_with_pulse == 2
    sci = fv.scalp_coupling_index(first, second, cardiac_sos, 250)
    contact = fv.contact_score(sci, pulse)
    assert contact[1] > 0.9 and contact[2] > 0.9
    assert contact[4:].max() < 0.5


def test_two_weak_channels_do_not_confirm_a_pulse(cardiac_sos):
    first, second = two_channel_pulse(0.0002)
    pulse = fv.pulse_metrics(first, second, SRATE, cardiac_sos)
    assert np.isnan(pulse.heart_hz) and pulse.channels_with_pulse == 0
    sci = fv.scalp_coupling_index(first, second, cardiac_sos, 250)
    assert fv.contact_score(sci, pulse)[1] < fv.WEAK_SCI


def test_two_strong_channels_on_different_pods_confirm_a_pulse(cardiac_sos):
    first, second = two_channel_pulse(0.003, channels=(1, 6))
    pulse = fv.pulse_metrics(first, second, SRATE, cardiac_sos)
    assert pulse.heart_hz == pytest.approx(1.4, abs=0.1)


def test_a_spectrum_falling_at_the_band_edge_is_not_a_pulse(cardiac_sos):
    rng = np.random.default_rng(12)
    n = 1000
    drift = np.cumsum(rng.normal(0, 0.002, (n, 1)), axis=0)
    gains = np.array([1.0, 0.8, 1.2, 0.9, 1.0, 0.7, 1.1, 0.6])
    first = drift * gains + rng.normal(0, 1e-5, (n, 8))
    second = drift * gains + rng.normal(0, 1e-5, (n, 8))
    assert np.isnan(fv.pulse_metrics(first, second, SRATE, cardiac_sos).heart_hz)


def test_notes_say_pulse_unclear_when_no_pulse_is_shared():
    notes = fv.channel_notes(None, None, np.full(8, 20.0), shared_pulse=False)
    assert notes == ["pulse unclear"] * 8


def test_table_shows_measurements_and_keeps_unknowns_blank(window):
    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    first, second = pulse_fixture()
    frames = np.zeros((1000, 16))
    frames[:, 0::2], frames[:, 1::2] = first, second
    window.push_samples(frames)
    window.poll_and_update()
    headers = [name for name, _ in fv.QUALITY_COLUMNS]
    pulse_col, start_col, note_col = headers.index("Pulse"), headers.index("Start 760"), headers.index("Notes")
    assert float(window.quality_table.item(0, pulse_col).text()) >= fv.PULSE_CLEAR_STRENGTH
    assert window.quality_table.item(0, start_col).text() == "--"
    assert window.quality_table.item(7, note_col).text() == "pulse unclear"
    assert "bpm" in window.lbl_heart.text() and "5 of 8" in window.lbl_heart.text()
    assert window.headmap_widget.channel_sci[0] > 0.9 > window.headmap_widget.channel_sci[7]


def test_bridge_publishes_startup_levels_in_stream_metadata():
    from bridges.artinis_to_lsl import ArtinisOctaPortaLSL, BaselineSettler

    settler = BaselineSettler()
    plateau = np.arange(16, dtype=np.float32) / 10 + 0.6
    settler.accept(plateau, 100.0)
    assert settler.startup_levels == pytest.approx(plateau)

    bridge = ArtinisOctaPortaLSL(mode="od", mock=True)
    bridge.setup_lsl(settler.startup_levels)
    channels = bridge.outlet.get_info().desc().child("channels")
    channel, values = channels.child("channel"), []
    while not channel.empty():
        values.append(channel.child_value("startup_od"))
        channel = channel.next_sibling("channel")
    layout = fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16)
    first, second = fv.startup_levels_per_channel(values, layout)
    assert first[1] == pytest.approx(0.8, abs=1e-3) and second[1] == pytest.approx(0.9, abs=1e-3)


def test_startup_levels_keep_each_columns_peak_and_mark_unseen_ones_unknown():
    from bridges.artinis_to_lsl import BaselineSettler, startup_light_text

    settler = BaselineSettler()
    early = np.zeros(16)
    early[:8] = 1.3  # R1 still shows its level while R2 is already baselined
    late = np.zeros(16)
    late[:8] = 1.3
    late[8:14] = 1.8  # T5..T7 show up one frame later, T8 never does
    settler.accept(early, 100.0)
    settler.accept(late, 100.02)
    levels = settler.startup_levels
    assert levels[:14] == pytest.approx([1.3] * 8 + [1.8] * 6)
    assert np.isnan(levels[14:]).all()
    assert startup_light_text(levels) == "1.30 1.30 1.30 1.30 1.80 1.80 1.80 --"

    values = [f"{value:.3f}" for value in levels]
    first, second = fv.startup_levels_per_channel(values, fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16))
    assert first[6] == pytest.approx(1.8) and np.isnan(first[7]) and np.isnan(second[7])
    assert fv.channel_notes(first, second, np.full(8, np.nan))[6] == "low light at start"
    assert fv.channel_notes(first, second, np.full(8, np.nan))[7] == ""


def test_no_startup_levels_when_the_first_frame_is_already_baselined():
    from bridges.artinis_to_lsl import BaselineSettler

    settler = BaselineSettler()
    settler.accept(np.full(16, 0.01), 100.0)
    assert settler.startup_levels is None


def test_time_axis_is_true_while_the_window_is_filling(window):
    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    window.push_samples(bridge_chunk(100))
    window.poll_and_update()
    x, _ = window.curves[0][0].getData()
    assert x[-1] == 0.0
    assert x[0] == pytest.approx(-99 / SRATE)


def test_concentration_stream_never_claims_measured_contact(window):
    labels = [f"Tx{i}_HbO" for i in range(1, 9)] + [f"Tx{i}_HbR" for i in range(1, 9)]
    window.attach_stream(fv.parse_stream_layout(labels, ["micromol/L"] * 16, 16), 16, SRATE)
    window.push_samples(np.zeros((400, 16)))
    window.poll_and_update()
    assert np.isnan(window.headmap_widget.channel_sci).all()
    assert not window.chk_mbll.isEnabled()


def test_preset_leaves_auto_scale_even_when_value_is_unchanged(window):
    window.combo_presets.setCurrentIndex(8)
    assert window.is_auto_scale
    window.combo_presets.setCurrentIndex(5)  # Same value the spin box already holds
    assert not window.is_auto_scale


def test_checkboxes_can_be_turned_back_on(window):
    window.chk_mbll.setChecked(False)
    assert not window.mode_mbll
    window.chk_mbll.setChecked(True)
    assert window.mode_mbll
    window.chk_filter.setChecked(False)
    assert not window.filter_enabled
    window.chk_filter.setChecked(True)
    assert window.filter_enabled


def test_power_hint_waits_then_names_only_weak_lp_transmitter(window):
    window.attach_stream(fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16), 16, SRATE)
    sci = np.array([0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, np.nan])
    window.update_power_hint(sci)
    assert window.lbl_power_hint.text() == ""
    window.weak_since[1] -= fv.WEAK_HOLD_SEC + 1
    window.update_power_hint(sci)
    assert "T2" in window.lbl_power_hint.text() and "T8" not in window.lbl_power_hint.text()
    sci[1] = 0.6  # Wobble inside the moderate range keeps the hint
    window.update_power_hint(sci)
    assert "T2" in window.lbl_power_hint.text()
    sci[1] = 0.9
    window.update_power_hint(sci)
    assert window.lbl_power_hint.text() == ""


def test_mock_bridge_stream_decodes_to_its_planted_concentration():
    from bridges.artinis_to_lsl import ArtinisOctaPortaLSL

    t = 3.7
    conc = ArtinisOctaPortaLSL(mode="conc", mock=True)
    od = ArtinisOctaPortaLSL(mode="od", mock=True)
    np.random.seed(0)
    conc_frame = conc.mock_frame(t)
    np.random.seed(0)
    od_frame = od.mock_frame(t)

    layout = fv.parse_stream_layout(BRIDGE_LABELS, ["OD"] * 16, 16)
    hbo, hbr = fv.to_concentrations(
        od_frame[layout.first][None, :], od_frame[layout.second][None, :], 0.0, 0.0, fv.KIND_OD,
        fv.compute_mbll_matrix(od.optode_dist_m * 100, dpf=od.dpf),
    )
    assert hbo[0] == pytest.approx(conc_frame[layout.first], rel=1e-3)
    assert hbr[0] == pytest.approx(conc_frame[layout.second], rel=1e-3)
