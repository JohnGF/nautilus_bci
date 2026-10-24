import json

import pytest

from nautilus_analysis import SpecError, build_trial_table, discover, infer_spec, load_raw, load_spec, read_spec
from nautilus_analysis.io import read_events_tsv
from nautilus_analysis.spec import spec_from_dict
from conftest import CLASSES, FALSE_START_TRIALS, TRIALS_PER_CLASS, real_dataset

MINIMAL = {"conditions": {"label_regex": "^Task_(\\w+)$"}}


@pytest.fixture(scope="module")
def frames(synthetic_bids):
    return {r.id.rsplit("_task", 1)[0]: read_events_tsv(r.events_path) for r in discover(synthetic_bids)}


def test_inference_recovers_the_paradigm(frames):
    spec = infer_spec(list(frames.values()), name="synthetic_lrud")
    c = spec.conditions
    assert c.label_regex == "^Task_(Bottom|Left|Right|Top)$" and c.anchor_regex is None
    assert c.baseline.event_regex == "^Cue_(Bottom|Left|Right|Top)$"
    assert spec.sham.event_regex == "^Rest$" and spec.sham.must_end_before_regex == c.baseline.event_regex
    assert spec.segments.mode_regex and spec.segments.block_regex and spec.segments.main_block_only
    assert spec.source == "inferred"


def test_loading_order(synthetic_bids, frames, tmp_path):
    specs_dir = tmp_path / "repo_specs"
    specs_dir.mkdir()
    events = list(frames.values())
    # 4. inference, with a draft saved for a human
    spec = load_spec(synthetic_bids, events_frames=events, draft_dir=tmp_path / "out", specs_dir=specs_dir,
                     verbose=False)
    assert spec.source.startswith("inferred") and (tmp_path / "out" / "inferred_spec.json").exists()
    assert read_spec(tmp_path / "out" / "inferred_spec.json").conditions == spec.conditions
    # 3. repo-side spec named after the dataset
    (specs_dir / "synthetic_lrud.json").write_text(json.dumps({**MINIMAL, "notes": "repo"}))
    assert load_spec(synthetic_bids, specs_dir=specs_dir).notes == "repo"
    # 2. code/nautilus_spec.json travels with the data
    code = tmp_path / "ds" / "code"
    code.mkdir(parents=True)
    (code / "nautilus_spec.json").write_text(json.dumps({**MINIMAL, "notes": "code"}))
    (specs_dir / "ds.json").write_text(json.dumps({**MINIMAL, "notes": "repo"}))
    spec = load_spec(tmp_path / "ds", specs_dir=specs_dir)
    assert spec.notes == "code" and spec.name == "ds"
    # 1. explicit --spec beats everything
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({**MINIMAL, "notes": "explicit", "_comment": "ignored"}))
    assert load_spec(tmp_path / "ds", spec_path=explicit, specs_dir=specs_dir).notes == "explicit"
    with pytest.raises(FileNotFoundError):
        load_spec(tmp_path / "ds", spec_path=tmp_path / "missing.json")


@pytest.mark.parametrize("bad, match", [
    ({"conditions": {"label_regex": "x"}, "montag": "gpype_32"}, r"unknown key\(s\) \['montag'\]"),
    ({"conditions": {"label_regex": "x", "window": {"tmin": 1, "tmax": 0}}}, "tmin < tmax"),
    ({"conditions": {"label_regex": "(unclosed"}}, "invalid regex"),
    ({"montage": "biosemi64"}, "montage"),
    ({"units": "nV"}, "units"),
    ({"bands": {"mu": [13, 8]}}, "bands.mu"),
    ({"conditions": {"label_regex": "x", "baseline": {"tmin": -0.5, "tmax": 0.5}}}, "anchor-relative baseline"),
])
def test_invalid_specs_fail_loudly(bad, match):
    with pytest.raises(SpecError, match=match):
        spec_from_dict(bad)


def test_trial_table_uses_only_within_session_events(synthetic_bids, frames):
    spec = infer_spec(list(frames.values()))
    for key, df in frames.items():
        table = build_trial_table(df, spec)
        truth = synthetic_bids.truth[key]
        assert list(table.trials["label"]) == [l for l, _ in truth]
        assert list(table.trials["anchor_onset"].round(3)) == [round(o, 3) for _, o in truth]
        assert (table.trials["anchor_onset"] - table.trials["baseline_onset"]).round(3).eq(1.0).all()  # Cue -> Task
        assert table.class_counts == {c: TRIALS_PER_CLASS for c in sorted(CLASSES)}
        assert table.selected_mode == "AudioEyesClosed" and table.dropped["other_mode"] == FALSE_START_TRIALS
        assert any("mixed-mode session" in w for w in table.warnings)
        # Sham Rest: within the main block, each window ends before the next cue
        assert len(table.sham) == len(truth) and table.sham["prev_label"].notna().all()


def test_explicit_mode_and_anchor_rules(frames):
    df = frames["sub-01_ses-01"]
    spec = spec_from_dict({"conditions": {"label_regex": "^Cue_(\\w+)$", "anchor_regex": "^Task_",
                                          "max_label_lag_s": 2.0},
                           "segments": {"mode_regex": "_Mode_(\\w+)$", "select_mode": "VisualOnly"}})
    table = build_trial_table(df, spec)
    assert len(table.trials) == FALSE_START_TRIALS and not table.warnings
    assert (table.trials["anchor_onset"] - table.trials["label_onset"]).round(3).eq(1.0).all()
    # Anchor-relative baseline when no baseline event is given
    assert (table.trials["baseline_onset"] == table.trials["anchor_onset"]).all()
    with pytest.raises(ValueError, match="no 'conditions'"):
        build_trial_table(df, spec_from_dict({}))


def test_label_after_anchor_at_same_time_and_label_map():
    import pandas as pd
    df = pd.DataFrame({"onset": [1.0, 5.0, 5.0, 9.0, 12.0, 12.0],
                       "trial_type": ["Rest", "Imagine", "FIRE selected", "Rest", "WATER selected", "Imagine"]})
    spec = spec_from_dict({"conditions": {"label_regex": "^(FIRE|WATER) selected$", "anchor_regex": "^Imagine$",
                                          "label_map": {"FIRE": "Fire", "WATER": "Water"}},
                           "sham": {"event_regex": "^Rest$", "tmin": 0.5, "tmax": 3.5,
                                    "must_end_before_regex": "selected$"}})
    table = build_trial_table(df, spec)
    assert list(table.trials["label"]) == ["Fire", "Water"]
    assert list(table.trials["anchor_onset"]) == [5.0, 12.0]
    # Rest at 1.0 ends at 4.5 (< 5.0) -> kept; Rest at 9.0 ends at 12.5 (> 12.0) -> dropped
    assert list(table.sham["onset"]) == [1.0] and table.dropped["sham_overlaps_next_trial"] == 1


@pytest.mark.realdata
def test_real_music_lrud_matches_legacy_parser():
    root = real_dataset("bids/music_lrud")
    import pandas as pd
    from analysis.compare_lrud_subjects import parse_events
    recs = [r for r in discover(root) if r.events_path]
    spec = infer_spec([read_events_tsv(r.events_path) for r in recs])
    for r in recs:
        table = build_trial_table(read_events_tsv(r.events_path), spec)
        tasks, rests, _, _ = parse_events(pd.read_csv(r.events_path, sep="\t"), "AudioEyesClosed")
        assert list(table.trials["label"]) == list(tasks["label"])
        assert list(table.trials["anchor_onset"]) == list(tasks["onset"])
        assert list(table.trials["baseline_onset"]) == list(tasks["cue"])
