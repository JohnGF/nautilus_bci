"""Analysis spec: everything dataset-specific, kept as JSON and out of the code.

Loading order (``load_spec``); the first hit wins:

1. an explicit ``--spec path.json``;
2. ``<bids_root>/code/nautilus_spec.json`` (travels with the data);
3. ``scripts/analysis/specs/<dataset_name>.json`` (repo-side);
4. auto-inference from the dataset's ``events.tsv`` files. The inferred spec
   is printed and saved as ``inferred_spec.json`` so a human can confirm it.

Schema (keys starting with ``_`` are comments and are ignored; any other
unknown key is an error, so typos fail loudly)::

    {
      "name": "music_lrud",
      "montage": "auto | from_channels_tsv | gpype_32 | gpype_16 | gpype_8 | standard_32",
      "units": "uV",
      "conditions": {
        "label_regex": "^Task_(Top|Bottom|Left|Right)$",   # group 1 = class label
        "anchor_regex": null,           # epoch t=0; null = the label event itself
        "max_label_lag_s": 5.0,         # label event may precede its anchor by this much
        "label_map": {"TOP": "Top"},    # optional renaming of captured labels
        "baseline": {"event_regex": "^Cue_", "tmin": -0.5, "tmax": 0.0},  # null regex = anchor
        "window": {"tmin": 0.5, "tmax": 3.5}
      },
      "sham": {"event_regex": "^Rest$", "tmin": 0.5, "tmax": 3.5, "must_end_before_regex": "^Cue_"},
      "segments": {"mode_regex": "_Mode_(\\\\w+)$", "block_regex": "^Block_Start_([A-Za-z]+)",
                   "select_mode": "AudioEyesClosed", "main_block_only": true},
      "bands": {"mu": [8, 13], "beta": [13, 30]},
      "confounds": ["subject<->music in AudioEyesClosed"]
    }
"""

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields

from .montage import MONTAGE_MODES

REPO_SPECS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis", "specs"))
DEFAULT_BANDS = {"theta": [4.0, 8.0], "mu": [8.0, 13.0], "beta": [13.0, 30.0]}
VALID_UNITS = ("uV", "mV", "V")


class SpecError(ValueError):
    pass


@dataclass
class BaselineSpec:
    event_regex: str | None = None
    tmin: float = -0.5
    tmax: float = 0.0


@dataclass
class WindowSpec:
    tmin: float = 0.5
    tmax: float = 3.5


@dataclass
class ConditionsSpec:
    label_regex: str
    anchor_regex: str | None = None
    max_label_lag_s: float = 5.0
    label_map: dict = field(default_factory=dict)
    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    window: WindowSpec = field(default_factory=WindowSpec)


@dataclass
class ShamSpec:
    event_regex: str
    tmin: float = 0.5
    tmax: float = 3.5
    must_end_before_regex: str | None = None


@dataclass
class SegmentsSpec:
    mode_regex: str | None = None
    block_regex: str | None = None
    select_mode: str | None = None  # None: the mode with the most trials (a warning is raised if several)
    main_block_only: bool = False


@dataclass
class AnalysisSpec:
    name: str | None = None
    montage: str = "auto"
    units: str = "uV"
    conditions: ConditionsSpec | None = None
    sham: ShamSpec | None = None
    segments: SegmentsSpec = field(default_factory=SegmentsSpec)
    bands: dict = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_BANDS.items()})
    confounds: list = field(default_factory=list)
    notes: str | None = None
    source: str = field(default="defaults", compare=False)  # where it was loaded from (not serialised)

    def to_dict(self):
        d = asdict(self)
        d.pop("source")
        return d

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


_NESTED = {"conditions": ConditionsSpec, "sham": ShamSpec, "segments": SegmentsSpec,
           "baseline": BaselineSpec, "window": WindowSpec}


def _build(cls, data, where):
    if not isinstance(data, dict):
        raise SpecError(f"{where}: expected an object, got {type(data).__name__}")
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    names = {f.name for f in fields(cls)} - {"source"}
    unknown = sorted(set(data) - names)
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(names)}")
    kwargs = {}
    for k, v in data.items():
        if k in _NESTED and v is not None:
            v = _build(_NESTED[k], v, f"{where}.{k}")
        kwargs[k] = v
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise SpecError(f"{where}: {exc}") from None


def _check_regex(value, where):
    if value is None:
        return
    try:
        re.compile(value)
    except re.error as exc:
        raise SpecError(f"{where}: invalid regex {value!r}: {exc}") from None


def validate(spec):
    if spec.montage not in MONTAGE_MODES:
        raise SpecError(f"montage: {spec.montage!r} not in {MONTAGE_MODES}")
    if spec.units not in VALID_UNITS:
        raise SpecError(f"units: {spec.units!r} not in {VALID_UNITS}")
    c = spec.conditions
    if c is not None:
        for name in ("label_regex", "anchor_regex"):
            _check_regex(getattr(c, name), f"conditions.{name}")
        _check_regex(c.baseline.event_regex, "conditions.baseline.event_regex")
        if c.baseline.tmin >= c.baseline.tmax or c.window.tmin >= c.window.tmax:
            raise SpecError("conditions: baseline/window need tmin < tmax")
        if c.baseline.event_regex is None and c.baseline.tmax > 0:
            raise SpecError("conditions.baseline: an anchor-relative baseline must end at or before 0 s")
    if spec.sham is not None:
        _check_regex(spec.sham.event_regex, "sham.event_regex")
        _check_regex(spec.sham.must_end_before_regex, "sham.must_end_before_regex")
    _check_regex(spec.segments.mode_regex, "segments.mode_regex")
    _check_regex(spec.segments.block_regex, "segments.block_regex")
    for band, rng in spec.bands.items():
        if len(rng) != 2 or not rng[0] < rng[1]:
            raise SpecError(f"bands.{band}: expected [lo, hi] with lo < hi, got {rng}")
    return spec


def spec_from_dict(data, source="dict"):
    spec = validate(_build(AnalysisSpec, data, "spec"))
    spec.source = source
    return spec


def read_spec(path):
    with open(path) as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{path}: invalid JSON: {exc}") from None
    return spec_from_dict(data, source=os.path.abspath(path))


def spec_candidates(bids_root, spec_path=None, specs_dir=REPO_SPECS_DIR):
    """The file locations ``load_spec`` checks, in priority order (steps 1-3)."""
    name = os.path.basename(os.path.normpath(bids_root))
    out = [spec_path] if spec_path else []
    return out + [os.path.join(bids_root, "code", "nautilus_spec.json"), os.path.join(specs_dir, f"{name}.json")]


def load_spec(bids_root, spec_path=None, events_frames=None, draft_dir=None, specs_dir=REPO_SPECS_DIR,
              verbose=True):
    """Return the ``AnalysisSpec`` for ``bids_root``, following the loading order.

    ``events_frames`` (events DataFrames of the dataset's recordings) are needed
    only for step 4 (inference). If ``draft_dir`` is given, the inferred spec
    is saved there as ``inferred_spec.json``.
    """
    if spec_path and not os.path.exists(spec_path):
        raise FileNotFoundError(f"--spec {spec_path} does not exist")
    for path in spec_candidates(bids_root, spec_path, specs_dir):
        if os.path.exists(path):
            spec = read_spec(path)
            if spec.name is None:
                spec.name = os.path.basename(os.path.normpath(bids_root))
            return spec
    spec = infer_spec(events_frames or [], name=os.path.basename(os.path.normpath(bids_root)))
    if draft_dir:
        spec.source = f"inferred (draft saved to {spec.save(os.path.join(draft_dir, 'inferred_spec.json'))})"
    if verbose:
        print(f"[spec] no spec file found for {bids_root}; inferred from events.tsv. "
              f"Please confirm it and save it as <bids_root>/code/nautilus_spec.json:")
        print(json.dumps(spec.to_dict(), indent=2))
    return spec


# --------------------------------------------------------------------------- inference

_NUM_RE = re.compile(r"\d+")
_SPLIT_RE = re.compile(r"([_ ]+)")
_ANCHOR_HINTS = ("task", "imag", "hitzone", "selected", "select")
_BASELINE_HINTS = ("cue",)
# Tokens that mark event lifecycles, not classes ("Start Listen" / "End Listen").
_LIFECYCLE = {"start", "started", "end", "ended", "stop", "stopped", "begin", "finish", "finished",
              "on", "off", "show", "hide", "played", "spawn", "open", "close", "pause", "resume"}


def _templ_rx(part):
    return "".join(r"\d+" if t == "N" else re.escape(t) for t in re.split(r"(N)", part) if t)


def _families(frames, min_count, max_classes):
    """Group event names that differ in one alphabetic token, e.g. ``Task_<Top|Left|...>``.

    Digits are wildcards, so ``Trial_Start_3_Dir_Top`` belongs to
    ``Trial_Start_N_Dir_{}``. Returns a list of dicts with keys name, regex,
    labels ({label: count}), total and onsets ({frame index: [onset, ...]}).
    """
    fam = defaultdict(lambda: {"labels": Counter(), "onsets": defaultdict(list)})
    for fi, df in enumerate(frames):
        for onset, tt in zip(df["onset"], df["trial_type"].astype(str)):
            tokens = _SPLIT_RE.split(tt)
            for i in range(0, len(tokens), 2):
                if not re.fullmatch(r"[A-Za-z]+", tokens[i]):
                    continue
                pre, post = "".join(tokens[:i]), "".join(tokens[i + 1:])
                if not pre and not post:
                    continue
                d = fam[(_NUM_RE.sub("N", pre), _NUM_RE.sub("N", post))]
                d["labels"][tokens[i]] += 1
                d["onsets"][fi].append(float(onset))
    out = []
    for (pre, post), d in fam.items():
        labels = {k: v for k, v in d["labels"].items() if v >= min_count}
        if not 2 <= len(labels) <= max_classes or any(l.lower() in _LIFECYCLE for l in labels):
            continue
        rx = "^" + _templ_rx(pre) + "(" + "|".join(sorted(map(re.escape, labels))) + ")" + _templ_rx(post) + "$"
        out.append({"name": f"{pre}{{}}{post}", "regex": rx, "labels": labels,
                    "total": sum(labels.values()), "onsets": d["onsets"]})
    return out


def _median_lag(f, ref, max_gap=10.0):
    """Median time from each of ``ref``'s events to the nearest ``f`` event (same file, within max_gap)."""
    lags = []
    for fi, ref_onsets in ref["onsets"].items():
        mine = f["onsets"].get(fi, [])
        for r in ref_onsets:
            near = [t - r for t in mine if abs(t - r) <= max_gap]
            if near:
                lags.append(min(near, key=abs))
    lags.sort()
    return lags[len(lags) // 2] if lags else 0.0


def infer_spec(events_frames, name=None, min_count=5, max_classes=8):
    """Infer a draft ``AnalysisSpec`` from events tables. Conditions are None if nothing fits.

    Heuristics: the class family is the group of event families with the
    same label set and the most events. Its anchor is a family whose name
    hints at the task period (Task_, Imagine, HitZone, "selected"); if none
    does, the family that comes last within a trial. The baseline is a family
    whose name contains "cue". An exact ``Rest`` event becomes the sham.
    """
    spec = AnalysisSpec(name=name, source="inferred")
    frames = [df for df in events_frames if df is not None and len(df)]
    all_types = Counter(tt for df in frames for tt in df["trial_type"].astype(str))
    if any(re.search(r"_Mode_[A-Za-z0-9]+$", t) for t in all_types):
        spec.segments.mode_regex = r"_Mode_([A-Za-z0-9]+)$"
    if any(t.startswith("Block_Start_") for t in all_types):
        spec.segments.block_regex = r"^Block_Start_([A-Za-z]+)"
        spec.segments.main_block_only = True

    fams = _families(frames, min_count, max_classes)
    if not fams:
        spec.notes = "No condition family found in events.tsv; set conditions.label_regex by hand."
        return spec
    groups = defaultdict(list)
    for f in fams:
        groups[frozenset(l.lower() for l in f["labels"])].append(f)
    group = max(groups.values(), key=lambda g: sum(f["total"] for f in g))

    hinted = lambda f, hints: any(h in f["name"].lower() for h in hints)
    ref = max(group, key=lambda f: f["total"])
    anchor = max([f for f in group if hinted(f, _ANCHOR_HINTS)] or group, key=lambda f: _median_lag(f, ref))
    others = [f for f in group if f is not anchor]
    baseline = next((f for f in others if hinted(f, _BASELINE_HINTS) and _median_lag(f, anchor) <= 0), None)

    spec.conditions = ConditionsSpec(
        label_regex=anchor["regex"],
        label_map={l: l.capitalize() for l in anchor["labels"] if l.isupper() and len(l) > 1},
        baseline=BaselineSpec(event_regex=baseline["regex"] if baseline else None, tmin=-0.5, tmax=0.0),
    )
    rest = next((t for t in all_types if t.lower() == "rest"), None)
    if rest and all_types[rest] >= min_count:
        spec.sham = ShamSpec(event_regex=f"^{re.escape(rest)}$", tmin=0.5, tmax=3.5,
                             must_end_before_regex=baseline["regex"] if baseline else anchor["regex"])
    spec.notes = (f"Inferred from {len(frames)} events.tsv file(s): label family '{anchor['name']}' "
                  f"{dict(sorted(anchor['labels'].items()))}; same-label families: {[f['name'] for f in others]}; "
                  f"baseline {'before ' + baseline['name'] if baseline else 'relative to the anchor'}. "
                  "Check anchor, baseline and window before trusting results.")
    return validate(spec)
