import os

import pytest

from nautilus_analysis.discover import discover, find_datasets, is_dataset_root
from conftest import REAL_BIDS_PARENTS, RECORDINGS, TASK


def test_lists_every_recording_in_both_formats(synthetic_bids):
    recs = discover(synthetic_bids)
    assert [(r.sub, r.ses) for r in recs] == [(s, e) for s, e, _, _ in RECORDINGS]
    exts = {r.id: os.path.splitext(r.eeg_path)[1] for r in recs}
    assert exts["sub-02_ses-02_task-imagery"] == ".set"
    assert exts["sub-01_ses-01_task-imagery"] == ".vhdr"
    for r in recs:
        assert r.task == TASK and r.dataset == "synthetic_lrud"
        assert r.events_path and r.channels_path and r.sidecar_path
        assert r.size_bytes > 100_000


@pytest.mark.parametrize("kwargs, expected", [
    ({"sub": "01"}, [("01", "01"), ("01", "02")]),
    ({"ses": "02"}, [("01", "02"), ("02", "02")]),
    ({"sub": ["02", "03"], "ses": "01"}, [("02", "01"), ("03", "01")]),
    ({"task": "nope"}, []),
])
def test_filters(synthetic_bids, kwargs, expected):
    assert [(r.sub, r.ses) for r in discover(synthetic_bids, **kwargs)] == expected


def test_nested_datasets_are_not_mixed(synthetic_bids, tmp_path):
    # A container folder holding two datasets: not a root itself; find_datasets lists both.
    import shutil
    shutil.copytree(synthetic_bids, tmp_path / "container" / "a")
    shutil.copytree(synthetic_bids, tmp_path / "container" / "b(old)")
    (tmp_path / "container" / "a" / "sub-01" / "dataset_description.json").write_text("{}")  # stray copy
    with pytest.raises(FileNotFoundError, match=r"contains these datasets: a, b\(old\)"):
        discover(tmp_path / "container")
    assert [os.path.basename(p) for p in find_datasets(tmp_path / "container")] == ["a", "b(old)"]
    # A dataset with a nested dataset inside it lists only its own recordings.
    shutil.copytree(synthetic_bids, tmp_path / "container" / "a" / "nested")
    assert len(discover(tmp_path / "container" / "a")) == len(RECORDINGS)


@pytest.mark.realdata
def test_discovers_every_repo_dataset():
    parents = [p for p in REAL_BIDS_PARENTS if os.path.isdir(p)]
    if not parents:
        pytest.skip("no scripts/bids*/ datasets present")
    roots = [r for p in parents for r in find_datasets(p)]
    assert len(roots) >= 10
    empty = []
    for root in roots:
        assert is_dataset_root(root)
        recs = discover(root)
        if not recs:
            empty.append(root)
        for r in recs:
            assert os.path.exists(r.eeg_path) and r.sub
    assert not empty, f"datasets with no EEG recordings: {empty}"
