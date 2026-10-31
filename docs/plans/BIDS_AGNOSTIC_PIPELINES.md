# Plan: dataset-agnostic analysis pipelines ("if it's BIDS, it runs")

Status: proposed, 2026-10-24. Owner: a follow-up task (Claude Code or human).
Read first: `AGENTS.md`, `docs/guides/STRICT_DECODING_METHODOLOGY.md`, `docs/science-notes/DATASET_CONFOUNDS.md`.

## Goal

Any EEG dataset in BIDS (an existing `scripts/bids*/` folder or a fresh recording copied
from the bci machine) can be analysed with **one command per pipeline**. Example:

```bash
cd scripts
uv run python -m analysis.run --bids-root bids/music_lrud --pipeline qc,topomap,decode
```

No code edits needed. Anything dataset-specific lives in a small, optional **analysis spec**
(JSON). The code never has it hard-wired.

## Why (evidence from the 2026-10-24 session)

| Problem | Evidence |
|---|---|
| Dataset roots are hard-wired | ~150 literal `bids_tower_defense`, `bids_td_classical`, `bids/bids_fnf`, ... strings across `scripts/analysis/` |
| Subjects/sessions are hard-wired | 15 scripts contain `sub="01"`-style literals; e.g. `plot_fnf_lateralization.py` only runs on FNF sub-01 ses-01..04 |
| Event names are hard-wired | `'FIRE selected'`, `'Imagine'`, `'Start Listen'`, `'Arrow_Left_HitZone'`, `'Task_Top'` literals in many scripts |
| 11 private copies of the channel montage, and they **disagree** | `STANDARD_32` (analysis scripts) vs `DEFAULT_CHANNELS_32` (`bridges/gpype_to_lsl.py`). Index 4 is C3 in one and F7 in the other. Files only store `EEG001..EEG033` |
| Only 17/50 scripts have a CLI | the rest run fixed paths from `__main__` |
| Task names are unreliable | `task-leftright` is used for music recall (see DATASET_CONFOUNDS.md), so the task entity cannot define the paradigm |
| No signal QC gate | `music_lrud` sub-04 ses-02 has 29/32 channels at exactly 0 µV and sub-06 is drift-only (alpha 0.03 µV²/Hz), yet every script ran "successfully" and produced plausible-looking accuracies and scalp maps |
| Mixed-mode sessions | `music_lrud` sessions contain false-start runs in other paradigm modes (`Experiment_Start_..._Mode_X` markers) that silently get pooled |
| No tests | no `tests/` directory, no pytest in `pyproject.toml` |

## Target architecture

```
scripts/
  nautilus_analysis/            # new library package (pure functions, no __main__ paths)
    discover.py   # BIDS discovery -> list[Recording(sub, ses, task, run, acq, paths)]
    io.py         # load_raw(recording, spec) -> mne.io.Raw with real channel names + montage + units
    montage.py    # THE single channel-layout source; resolves names from channels.tsv / electrodes.tsv / spec
    qc.py         # flat/dead channels, drift ratio, line noise, alpha plausibility -> QCReport; gate
    events.py     # events.tsv -> annotated trial table (label, anchor onsets, block, mode) via spec rules
    spec.py       # AnalysisSpec schema, loading order, auto-inference from events.tsv
    epochs.py     # baseline-corrected epochs + sham-rest epochs (strict rules 1,2,5)
    decode.py     # fold-internal pipelines, CV selection (LOSO if >1 session else StratifiedKFold), permutation, sham
    spectral.py   # PSD / band power / ERD% / topomaps
    report.py     # JSON + Markdown + figures, results/<pipeline>/<dataset>/<sub>_<ses>_<task>/ + write_provenance
  analysis/
    run.py        # single CLI: --bids-root --sub --ses --task --spec --pipeline --out-dir
    specs/        # repo-side specs for existing datasets (bids_fnf.json, bids_td_classical.json, music_lrud.json, ...)
    <legacy>.py   # existing scripts, ported to thin wrappers or moved to analysis/legacy/
  tests/
    conftest.py   # synthetic BIDS fixture built with mne_bids.write_raw_bids (tiny, simulated)
```

### Analysis spec (JSON, no new dependency; `pyyaml` is not installed)

The loader takes the first spec it finds, in this order:

1. `--spec path.json`
2. `<bids_root>/code/nautilus_spec.json`. BIDS reserves `code/`, and the spec travels with the data.
3. `scripts/analysis/specs/<dataset_name>.json`
4. Auto-inference from `events.tsv`. The loader prints the inferred spec and saves a draft to the output dir so a human can confirm it.

The spec contains:

```json
{
  "montage": "gpype_32 | standard_32 | from_channels_tsv",
  "units": "uV",
  "conditions": {
    "label_regex": "^Task_(Top|Bottom|Left|Right)$",
    "anchor": "trial_type match (epoch t=0)",
    "baseline": {"event_regex": "^Cue_", "tmin": -0.5, "tmax": 0.0},
    "window": {"tmin": 0.5, "tmax": 3.5}
  },
  "sham": {"event_regex": "^Rest$", "tmin": 0.5, "tmax": 3.5, "must_end_before_regex": "^Cue_"},
  "segments": {"mode_regex": "_Mode_(\\w+)$", "block_regex": "^Block_Start_(Music|NoMusic)", "select_mode": "AudioEyesClosed", "main_block_only": true},
  "bands": {"mu": [8, 13], "beta": [13, 30]},
  "confounds": ["subject<->music in AudioEyesClosed"]
}
```

## Invariants every pipeline must enforce

1. **The QC gate runs first.** Flat channels (std < 0.5 µV) are reported and dropped. If fewer than N channels survive (`--min-live-channels`, default 16 for topomaps and 8 for decoding), the run refuses with a clear message. QC also flags implausible spectra (e.g. alpha power < 0.1 µV²/Hz, or > 95% of power below 4 Hz) as "probably disconnected".
2. **Labels come only from within-session events** and never from file or session membership (strict rule 1).
3. **All `STRICT_DECODING_METHODOLOGY.md` invariants live in `decode.py`**, so no script can skip them: pre-cue baseline correction, fold-internal transforms, sham rest, permutation null, data quantification, provenance.
4. **Multiple-comparison awareness.** When a run tests many (channel set × pipeline) cells, report how many were tested and add a max-statistic or FDR correction. The music_lrud sub-06 "p=0.01" from a disconnected cap is the cautionary example.
5. **Montage honesty.** If the montage comes from an index mapping rather than real names in `channels.tsv`, every figure title and report says "montage assumed: <name>".
6. **Output paths** go under `results/<pipeline>/<dataset>/<sub>_<ses>_<task>[_run]/` and always include `RUN_INFO.txt`. Never use cwd-relative paths.

## Phases

### Phase 0: Decisions (blocking; ask the user)
- **Which 32-channel layout is real?** `gpype_to_lsl.DEFAULT_CHANNELS_32` or `STANDARD_32`? Check the g.Nautilus cap PDF in `docs/vendor-manuals/` and the recorder config on the bci machine. Going forward, recorders should write real names into `channels.tsv` (fix this in `recorders/`) so the question never comes up again.
- Legacy scripts: port each one to a thin wrapper over the library, or move it to `analysis/legacy/` and freeze it?

### Phase 1: Foundation (library + tests, no behaviour change to existing scripts)
- `discover.py` (via `mne_bids.get_entity_vals` / `find_matching_paths`), `io.py`, `montage.py`, `qc.py`, `spec.py`, `events.py`.
- `tests/`: add `pytest` as a dev dependency with `uv add --dev pytest`. Build a synthetic BIDS fixture with 2 subjects, 2 sessions, a 4-class events table with Rest and a planted mu-ERD effect, plus one dead-channel recording. The fixture is how we check that QC refuses the dead recording and decoding finds the planted effect.
- Acceptance: `uv run pytest` passes; `discover` lists every dataset under `scripts/bids*/` without errors (`.set/.fdt` EEGLAB and BrainVision both).

### Phase 2: Generic pipelines
Build these in order of value: `qc` → `bandpower/psd` → `erd_topomap` → `decode` (strict) → `transfer` (cross-session/subject).
- Seed from the code that already does it correctly:
  - `analysis/compare_lrud_subjects.py`: strict decoding, sham, permutation, event parsing by mode/block.
  - `analysis/plot_lrud_topomaps.py`: ERD topomaps with a flat-channel guard.
  - `analysis/benchmark_strict_2class_music.py`: the reference strict implementation and GPU models.
- `analysis/run.py` is the CLI that chains pipelines and works with `--sub/--ses/--task` omitted (meaning all).
- Acceptance: one command runs every pipeline on `bids/music_lrud`, `bids_clean/bids_fnf` and `bids_clean/bids_td_classical`, plus one fresh synthetic dataset, without per-dataset code.

### Phase 3: Specs for existing datasets
- Write `scripts/analysis/specs/<dataset>.json` for each dataset under `scripts/bids*/`, deriving the event rules from the current hard-coded scripts.
- Record the known confounds from DATASET_CONFOUNDS.md in each spec so reports print them.

### Phase 4: Migrate the 50 legacy scripts
- Triage table: script → which generic pipeline replaces it → port / wrap / archive.
- Regression check: for each ported script, the generic pipeline's numbers on the original dataset match the existing `results/` JSON within tolerance (seeded CV), *or* the difference is explained (e.g. the old script leaked).
- Make this one PR per pipeline family. Don't attempt it as one big-bang change.

### Phase 5: Docs
- `docs/guides/RUNNING_ANALYSES.md`: the `run.py` CLI, the spec format, the QC gate, and how to bring in a new recording from the bci machine.
- Update `AGENTS.md` "Running things" and "Path conventions". Ask before editing it, because it currently has unstaged changes.

## Out of scope
- Moving or restructuring BIDS datasets (forbidden by AGENTS.md); edits inside submodules.
- Real-time/online decoders (`bridges/`, `tasks/`). Their shared bits (montage, QC) can adopt the library later.

## Known data issues to surface, not fix in code
- `music_lrud` sub-04 ses-02 has only 3/32 live channels; sub-04 ses-01 is empty (3 KB); sub-06 ses-01 has no EEG signal (disconnected) and its markers say `Sub_01_Ses_01`.
- `bids_listening/sub-01/ses-02` duplicates `bids_music/sub-01/ses-02`.
