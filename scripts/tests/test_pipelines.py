"""Phase 2: epochs, decode, spectral and the run_dataset orchestration on the synthetic fixture."""

import json
import os

import numpy as np
import pytest

from nautilus_analysis import apply_qc, build_trial_table, compute_qc, discover, infer_spec, load_raw
from nautilus_analysis.decode import channel_sets, choose_cv, decode_epochset, model_factories
from nautilus_analysis.epochs import concat, make_epochs
from nautilus_analysis.pipelines import RunOptions, parse_pipelines, run_dataset
from nautilus_analysis.spectral import erd, erd_stats
from conftest import CLASSES, ERD_CHANNEL, TRIALS_PER_CLASS

FAST = model_factories(["Riemann TS+LR"])


@pytest.fixture(scope="module")
def prepared(synthetic_bids):
    """{rec id: (clean raw, trial table)} for the good recordings of sub-01 and sub-02, plus the spec."""
    recs = [r for r in discover(synthetic_bids) if r.sub in ("01", "02")]
    loaded = [load_raw(r) for r in recs]
    spec = infer_spec([L.events for L in loaded])
    out = {L.recording.id: (apply_qc(L.raw, compute_qc(L.raw)), build_trial_table(L.events, spec)) for L in loaded}
    return out, spec


def _epochs(prepared, rid, band=(8.0, 30.0)):
    tables, spec = prepared
    raw, table = tables[rid]
    return make_epochs(raw, table, spec, *band, group=rid)


def test_epochs_shapes_baseline_and_sham(prepared):
    rid = "sub-01_ses-01_task-imagery"
    es = _epochs(prepared, rid, band=(None, None))
    assert es.X.shape == (32, 32, 750) and es.X.dtype == np.float32
    assert es.class_counts == {c: TRIALS_PER_CLASS for c in sorted(CLASSES)}
    assert len(es.X_sham) == 32 and set(es.groups) == {rid}
    assert es.active_task_s == pytest.approx(96.0) and es.wear_time_s > 300
    # Every main-block Rest follows a trial, so carry-over labels are all present.
    assert all(p in CLASSES for p in es.y_sham_prev)


def test_concat_pools_on_common_channels(prepared):
    a = _epochs(prepared, "sub-01_ses-01_task-imagery")
    b = _epochs(prepared, "sub-01_ses-02_task-imagery").pick_channels(["C3", "Cz", "C4", "Pz"])
    pooled = concat([a, b])
    assert pooled.ch_names == ["C3", "Cz", "C4", "Pz"] and len(pooled.y) == 64
    assert sorted(set(pooled.groups)) == ["sub-01_ses-01_task-imagery", "sub-01_ses-02_task-imagery"]


def test_choose_cv():
    y = np.array(list("ABAB" * 5), dtype=object)
    assert choose_cv(y)[2].startswith("StratifiedKFold 5")
    groups = np.array(["r1"] * 10 + ["r2"] * 10)
    assert choose_cv(y, groups)[2] == "Leave-One-Group-Out over 2 recordings"
    # A group with a single class can't be a LOGO test fold -> stratified fallback
    y1 = np.array(["A"] * 10 + list("AB" * 5), dtype=object)
    assert choose_cv(y1, groups)[2].startswith("StratifiedKFold")
    with pytest.raises(ValueError, match=">= 2 classes"):
        choose_cv(np.array(["A"] * 10))


def test_channel_sets_use_real_names():
    from nautilus_analysis.montage import GPYPE_32
    sets = channel_sets(GPYPE_32)
    assert sets["fp1_fp2_only"] == ["Fp1", "Fp2"]
    assert not {"Fp1", "AF3", "F7", "Fz"} & set(sets["no_frontal"]) and {"FC1", "C3", "Oz"} <= set(sets["no_frontal"])


def test_decode_finds_planted_effect_and_sham_passes(prepared):
    es = _epochs(prepared, "sub-01_ses-01_task-imagery")
    res = decode_epochset(es, n_perm=20, n_sham_draws=5, pipelines=FAST, n_jobs=1)
    cells = res["cells"]
    assert res["multiple_comparisons"]["n_cells"] == 3 and res["chance"] == 0.25
    assert cells["no_frontal"]["Riemann TS+LR"]["acc"] > 0.5
    assert cells["no_frontal"]["Riemann TS+LR"]["p_maxstat"] < 0.05
    assert cells["fp1_fp2_only"]["Riemann TS+LR"]["acc"] < 0.45  # no effect frontally
    assert res["sham"]["Riemann TS+LR"]["random"]["status"] == "pass"
    q = res["quantification"]
    assert q["n_trials"] == 32 and q["active_task_s"] == 96.0 and "fit_wall_s" in q
    json.dumps(res)  # serialisable (no null arrays left behind)


def test_decode_refuses_single_class(prepared):
    es = _epochs(prepared, "sub-01_ses-01_task-imagery").pick_labels(["Left"])
    with pytest.raises(ValueError, match=">= 2 classes"):
        decode_epochset(es, n_perm=0, pipelines=FAST)


def test_erd_recovers_planted_channels(prepared):
    tables, spec = prepared
    raw, table = tables["sub-01_ses-01_task-imagery"]
    res = erd(raw, table, spec, (8.0, 13.0))
    st = erd_stats(res)
    for label, ch in ERD_CHANNEL.items():
        m = st[label]["mean_erd"]
        assert res.ch_names[int(np.argmin(m))] == ch, (label, dict(zip(res.ch_names, m.round())))
        assert m[res.ch_names.index(ch)] < -50


def test_parse_pipelines():
    assert parse_pipelines("decode,topomap,qc") == ["qc", "erd_topomap", "decode"]
    assert parse_pipelines("all") == ["qc", "psd", "erd_topomap", "decode", "transfer"]
    assert parse_pipelines("bandpower") == ["psd"]
    with pytest.raises(ValueError, match="unknown pipeline"):
        parse_pipelines("qc,nope")


def test_run_dataset_end_to_end(synthetic_bids, tmp_path):
    opts = RunOptions(results_root=str(tmp_path), n_perm=5, n_sham_draws=3, models=["Riemann TS+LR"], n_jobs=1)
    summary = run_dataset(synthetic_bids, parse_pipelines("all"), opts, log=lambda *a: None)
    ds = os.path.basename(synthetic_bids)
    recs = summary["recordings"]
    assert not any("error" in r for r in recs.values())

    # The dead cap is refused by every pipeline, with the reason written next to the outputs.
    dead = recs["sub-03_ses-01_task-imagery"]
    assert dead["refused"] and dead["qc_status"] == "refused"
    for name in ("psd", "erd_topomap", "decode"):
        d = tmp_path / name / ds / "03_01_imagery"
        assert json.loads((d / f"{name}.json").read_text())["status"] == "refused"
        assert "only 3/32 EEG channels" in (d / f"{name}.md").read_text()
        assert (d / "RUN_INFO.txt").exists()

    good = tmp_path / "decode" / ds / "01_01_imagery"
    md = (good / "decode.md").read_text()
    assert "montage assumed: gpype_32" in md and "Sham Rest controls" in md and (good / "decode.png").exists()
    for name, files in {"qc": ["qc.json", "qc_channels.png"], "psd": ["psd.png", "psd.npz"],
                        "erd_topomap": ["erd_mu.png", "erd_topomap.json"]}.items():
        for f in files:
            assert (tmp_path / name / ds / "01_01_imagery" / f).exists(), (name, f)

    # Transfer holds out whole recordings within a subject and whole subjects across them.
    t = summary["transfer"]
    assert t["sub-01"]["status"] == "ok" and t["sub-01"]["cv"] == "Leave-One-Group-Out over 2 recordings"
    assert t["all_subjects"]["cv"] == "Leave-One-Group-Out over 2 subjects"
    assert "sub-03" not in t  # refused by QC, so it contributes no epochs
    assert (tmp_path / "summary" / ds / "summary.md").exists()
    assert (tmp_path / "spec" / ds / "inferred_spec.json").exists()


def test_run_cli_help():
    from analysis.run import build_parser
    assert "--bids-root" in build_parser().format_help()


def _stream(prepared, onsets, labels):
    """The fixture's epochs relabelled and re-timed, to build event-stream / blocked designs."""
    es = _epochs(prepared, "sub-01_ses-01_task-imagery").pick_channels(["C3", "Cz", "C4", "Pz"])
    return es._replace(onsets=np.asarray(onsets, float), y=np.asarray(labels, dtype=object), X_sham=es.X_sham[:0],
                       y_sham_prev=es.y_sham_prev[:0], groups_sham=es.groups_sham[:0])


def test_purged_cv_drops_training_epochs_that_overlap_test(prepared):
    from nautilus_analysis.decode import PurgedCV
    onsets = np.arange(32) * 0.4  # 3 s windows every 0.4 s, like FNF arrows
    es = _stream(prepared, onsets, np.resize(CLASSES, 32))
    assert es.overlap_fraction == 1.0
    cv = PurgedCV(choose_cv(es.y, contiguous=True)[0], es.onsets, es.recordings, es.window_s)
    for train, test in cv.split(es.X, es.y):
        gaps = np.abs(es.onsets[train][:, None] - es.onsets[test][None, :])
        assert gaps.min() >= es.window_s - 1e-9


def test_blocked_labels_are_flagged_or_held_out_by_block(prepared):
    onsets = np.arange(32) * 10.0
    one_block = _stream(prepared, onsets, np.repeat(CLASSES, 8))  # each class one time block
    res = decode_epochset(one_block, n_perm=5, n_sham_draws=2, pipelines=FAST, n_jobs=1)
    assert res["blocked_design"]["status"] == "CONFOUNDED" and res["verdict"].startswith("CONFOUNDED")

    two_blocks = _stream(prepared, onsets, np.tile(np.repeat(CLASSES, 4), 2))  # two blocks per class
    res = decode_epochset(two_blocks, n_perm=5, n_sham_draws=2, pipelines=FAST, n_jobs=1)
    assert res["blocked_design"]["status"] == "block-wise CV"
    assert res["cv"].startswith("StratifiedGroupKFold 2 over label blocks")

    shuffled = _stream(prepared, onsets, np.resize(CLASSES, 32))
    assert decode_epochset(shuffled, n_perm=0, pipelines=FAST, n_jobs=1)["blocked_design"] is None


def test_block_permutation_keeps_blocks_intact():
    from nautilus_analysis.decode import _permuted_labels
    y = np.repeat(np.array(CLASSES, dtype=object), 3)
    units = np.repeat(np.arange(4), 3)
    yp = _permuted_labels(y, None, units, np.random.default_rng(0))
    assert all(len(set(yp[units == u])) == 1 for u in range(4)) and sorted(yp) == sorted(y)
