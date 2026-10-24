import pytest

from nautilus_analysis.montage import GPYPE_32, STANDARD_32_LEGACY, resolve_channels


def test_gpype_32_matches_vendor_manual_and_bridge():
    # g.Nautilus PRO manual section 16.2, logical channels 1..32
    assert GPYPE_32[:6] == ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3"]
    assert GPYPE_32[14:17] == ["C3", "Cz", "C4"] and GPYPE_32[-1] == "Oz" and len(GPYPE_32) == 32
    assert STANDARD_32_LEGACY[4] == "C3" and GPYPE_32[4] == "F7"  # the documented disagreement


def test_index_names_get_gpype_layout_and_battery_becomes_misc():
    names = [f"EEG{i:03d}" for i in range(1, 34)]
    res = resolve_channels(names)
    assert res.names[:32] == GPYPE_32 and res.names[32] == "EEG033"
    assert res.types[32] == "misc" and res.extra_misc == ["EEG033"]
    assert res.assumed and res.label == "montage assumed: gpype_32"


def test_real_names_are_kept_and_canonicalised():
    res = resolve_channels(["FP1", "CZ", "c3", "C4", "Battery"], ["eeg"] * 5)
    assert res.names == ["Fp1", "Cz", "C3", "C4", "Battery"]
    assert res.types == ["eeg", "eeg", "eeg", "eeg", "misc"]
    assert not res.assumed and res.label == "montage from channels.tsv"


def test_mostly_numeric_names_fall_back_to_layout():
    # bids_thinking_misc: 'CZ', '2', ..., '32' + a MISC battery column
    names = ["CZ"] + [str(i) for i in range(2, 33)] + ["Battery"]
    res = resolve_channels(names, ["eeg"] * 32 + ["misc"])
    assert res.names[:32] == GPYPE_32 and res.assumed


def test_explicit_layout_and_errors():
    assert resolve_channels([f"E{i}" for i in range(32)], montage="standard_32").names == STANDARD_32_LEGACY
    with pytest.raises(ValueError, match="Cannot infer a channel layout for 20"):
        resolve_channels([f"E{i}" for i in range(20)])
    with pytest.raises(ValueError, match="not 10-20 names"):
        resolve_channels(["Cz", "EEG002"], montage="from_channels_tsv")
    with pytest.raises(ValueError, match="Unknown montage"):
        resolve_channels(["Cz"], montage="biosemi")
