"""The generic pipelines, chained per dataset: qc -> psd -> erd_topomap -> decode -> transfer.

``run_dataset(bids_root, pipelines, opts)`` discovers recordings, loads the
spec (``spec.load_spec`` order), and handles each recording once:

1. load + QC (``qc.compute_qc``). Each pipeline then gates the report at its own
   ``min_live`` (topomaps 16, everything else 8 by default). A refused pipeline
   writes a short report that explains why and moves on. It never crashes the
   run or produces numbers from a dead cap;
2. flat channels dropped (``qc.apply_qc``), trials built from within-session
   events (``events.build_trial_table``);
3. per-recording pipelines write to ``<results>/<pipeline>/<dataset>/<unit>/``.

``transfer`` pools the decode epochs. Within each subject it uses
Leave-One-Recording-Out over that subject's recordings. Across subjects it
uses Leave-One-Subject-Out. Only labels that every pooled unit has with at
least ``min_per_class`` trials are used. A dataset summary goes to
``<results>/summary/<dataset>/``.
"""

import copy
import os
import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np

from . import report as rp
from .decode import decode_epochset, model_factories
from .discover import discover
from .epochs import concat, make_epochs
from .events import build_trial_table
from .io import load_raw, read_events_tsv
from .qc import QCRefusal, QCThresholds, apply_qc, compute_qc, gate
from .spec import load_spec
from .spectral import band_power, erd, erd_stats, psd

PIPELINES = ("qc", "psd", "erd_topomap", "decode", "transfer")
ALIASES = {"bandpower": "psd", "topomap": "erd_topomap", "erd": "erd_topomap", "strict": "decode"}
MOTOR = ("C3", "Cz", "C4")


@dataclass
class RunOptions:
    results_root: str
    spec_path: str | None = None
    sub: list | None = None
    ses: list | None = None
    task: list | None = None
    min_live_topomap: int = 16
    min_live_decode: int = 8
    allow_implausible: bool = False
    l_freq: float = 8.0
    h_freq: float = 30.0
    n_perm: int = 100
    folds: int = 5
    repeats: int = 1
    n_sham_draws: int = 20
    min_per_class: int = 5
    n_jobs: int = -1
    models: list | None = None  # decode.make_pipelines names; None = all
    script_file: str | None = None
    thresholds: QCThresholds = field(default_factory=QCThresholds)


def parse_pipelines(text):
    """'qc,topomap,decode' or 'all' -> the canonical names, in pipeline order."""
    want = set()
    for name in (t.strip() for t in text.split(",") if t.strip()):
        if name == "all":
            want.update(PIPELINES)
            continue
        name = ALIASES.get(name, name)
        if name not in PIPELINES:
            raise ValueError(f"unknown pipeline {name!r}; choose from {PIPELINES} (aliases {ALIASES}) or 'all'")
        want.add(name)
    return [p for p in PIPELINES if p in want]


def _gate(qc, min_live, opts):
    """(passed, gated copy of the report). Never raises."""
    rep = copy.deepcopy(qc)
    try:
        return True, gate(rep, min_live, opts.thresholds, opts.allow_implausible)
    except QCRefusal:
        return False, rep


def _params(opts, bids_root, spec, extra=None):
    d = {k: v for k, v in asdict(opts).items() if k not in ("script_file", "thresholds")}
    d.update(bids_root=bids_root, spec_source=spec.source, montage_mode=spec.montage, **(extra or {}))
    return d


def _refused(path, title, ctx, rep, min_live):
    rp.write_md(os.path.join(path, f"{title}.md"), rp.header(f"{title}: refused by the QC gate", ctx) + [
        f"**Not analysed.** QC gate (min live channels {min_live}) refused this recording:", ""]
        + [f"- {r}" for r in rep.reasons])
    rp.write_json(os.path.join(path, f"{title}.json"), dict(status="refused", reasons=rep.reasons, qc=rep.to_dict()))


def _skipped(path, title, ctx, reason):
    rp.write_md(os.path.join(path, f"{title}.md"), rp.header(f"{title}: skipped", ctx) + [f"**Skipped:** {reason}"])
    rp.write_json(os.path.join(path, f"{title}.json"), dict(status="skipped", reason=reason))


def run_dataset(bids_root, pipelines, opts, log=print):
    """Run ``pipelines`` on every matching recording of ``bids_root``. Returns the summary dict."""
    bids_root = os.path.abspath(bids_root)
    recs = discover(bids_root, sub=opts.sub, ses=opts.ses, task=opts.task)
    dataset = os.path.basename(bids_root)
    model_factories(opts.models)  # fail fast on a typo
    if not recs:
        raise FileNotFoundError(f"no EEG recordings in {bids_root} for sub={opts.sub} ses={opts.ses} task={opts.task}")
    frames = [read_events_tsv(r.events_path) for r in recs]
    spec = load_spec(bids_root, opts.spec_path, frames, draft_dir=os.path.join(opts.results_root, "spec", dataset),
                     verbose=False)
    log(f"\n=== {dataset}: {len(recs)} recording(s); spec: {spec.source}; pipelines: {','.join(pipelines)}")
    if spec.source.startswith("inferred"):
        log(f"[spec] inferred, please confirm: label_regex={spec.conditions.label_regex if spec.conditions else None}")
    summary = dict(dataset=dataset, bids_root=bids_root, spec_source=spec.source, spec=spec.to_dict(),
                   pipelines=pipelines, recordings={}, transfer={})
    epoch_sets = {}
    t0 = time.time()
    for i, rec in enumerate(recs, 1):
        row = summary["recordings"].setdefault(rec.id, {})
        log(f"  [{i}/{len(recs)}] {rec.id} (dataset elapsed {time.time() - t0:.0f} s)")
        try:
            es = _run_recording(rec, spec, pipelines, opts, row, log)
            if es is not None:
                epoch_sets[rec.id] = (rec, es)
        except Exception as exc:  # one broken file must not stop the dataset
            row["error"] = f"{type(exc).__name__}: {exc}"
            log(f"  [error] {rec.id}: {row['error']}")
            log(traceback.format_exc(limit=3))
    if "transfer" in pipelines:
        summary["transfer"] = _run_transfer(dataset, bids_root, epoch_sets, spec, opts, log)
    _write_summary(summary, opts, spec)
    return summary


def _run_recording(rec, spec, pipelines, opts, row, log):
    unit = rp.unit_name(rec.id)
    L = load_raw(rec, montage=spec.montage)
    qc = compute_qc(L.raw, rec.id, opts.thresholds, L.line_freq)
    ok_dec, rep_dec = _gate(qc, opts.min_live_decode, opts)
    ok_topo, rep_topo = _gate(qc, opts.min_live_topomap, opts)
    ctx = dict(dataset=rec.dataset, unit=unit, montage=L.channels.label, spec_source=spec.source,
               confounds=spec.confounds, qc=rep_dec.summary())
    row.update(qc_status=rep_dec.status, n_live=qc.n_live, n_eeg=qc.n_eeg, alpha_uv2_hz=qc.alpha_uv2_hz,
               montage=L.channels.label, qc_reasons=rep_dec.reasons)
    log(f"  {rep_dec.summary()}")
    params = _params(opts, rec.root, spec, dict(recording=rec.eeg_path))
    mk = lambda name: rp.out_dir(opts.results_root, name, rec.dataset, unit, params, opts.script_file)

    if "qc" in pipelines:
        path = mk("qc")
        rp.write_json(os.path.join(path, "qc.json"), dict(
            report=qc.to_dict(), gate={f"min_live_{opts.min_live_decode}": dict(status=rep_dec.status, reasons=rep_dec.reasons,
                                                                              warnings=rep_dec.warnings),
                                       f"min_live_{opts.min_live_topomap}": dict(status=rep_topo.status,
                                                                               reasons=rep_topo.reasons)},
            channels=dict(source=L.channels.source, assumed=L.channels.assumed, original=L.channels.original,
                          names=L.channels.names, extra_misc=L.channels.extra_misc),
            events_source=L.events_source, units=L.units))
        rp.plot_qc(rep_dec, L.channels.label, os.path.join(path, "qc_channels.png"), opts.thresholds.flat_std_uv)
        rp.write_md(os.path.join(path, "qc.md"), rp.header("Signal QC", ctx) + [
            "| Gate | Status | Reasons / warnings |", "| :--- | :--- | :--- |",
            f"| decode/psd (min {opts.min_live_decode} live) | {rep_dec.status} | {'; '.join(rep_dec.reasons + rep_dec.warnings)} |",
            f"| topomap (min {opts.min_live_topomap} live) | {rep_topo.status} | {'; '.join(rep_topo.reasons)} |", "",
            f"Flat channels: {qc.flat_channels or 'none'}", "", "![channels](qc_channels.png)"])

    if not ok_dec:
        for name in ("psd", "erd_topomap", "decode"):
            if name in pipelines:
                _refused(mk(name), name, ctx, rep_dec, opts.min_live_decode)
        row["refused"] = True
        return None
    ctx["warnings"] = rep_dec.warnings
    clean = apply_qc(L.raw, qc)

    table = None
    if spec.conditions is not None and len(L.events):
        table = build_trial_table(L.events, spec)
        ctx["warnings"] = ctx["warnings"] + table.warnings
        row.update(class_counts=table.class_counts, n_sham=int(len(table.sham)), trial_drops=table.dropped,
                   mode=table.selected_mode)
    no_trials = ("spec has no conditions" if spec.conditions is None else
                 "no events" if table is None else "no trials matched the spec" if table.trials.empty else None)

    if "psd" in pipelines:
        log("    psd ...")
        path = mk("psd")
        f, pxx, chans = psd(clean)
        bp = band_power(f, pxx, spec.bands)
        rp.plot_psd(f, pxx, bp, chans, spec.bands, f"{rec.id} PSD ({L.channels.label})", os.path.join(path, "psd.png"))
        np.savez(os.path.join(path, "psd.npz"), freqs=f, pxx_uv2_hz=pxx, ch_names=chans)
        rp.write_json(os.path.join(path, "psd.json"), dict(ch_names=chans, band_power_uv2_hz={k: dict(zip(chans, v)) for k, v in bp.items()},
                                                           median_band_power_uv2_hz={k: float(np.median(v)) for k, v in bp.items()}))
        rp.write_md(os.path.join(path, "psd.md"), rp.header("PSD and band power", ctx) + [
            "| Band | Hz | Median over channels (µV²/Hz) |", "| :--- | :--- | ---: |"]
            + [f"| {k} | {spec.bands[k][0]:g}-{spec.bands[k][1]:g} | {np.median(v):.3g} |" for k, v in bp.items()]
            + ["", "![psd](psd.png)"])
        row["psd"] = "ok"

    if "erd_topomap" in pipelines:
        path = mk("erd_topomap")
        if not ok_topo:
            _refused(path, "erd_topomap", ctx, rep_topo, opts.min_live_topomap)
            row["erd_topomap"] = "refused"
        elif no_trials:
            _skipped(path, "erd_topomap", ctx, no_trials)
            row["erd_topomap"] = "skipped"
        else:
            row["erd_topomap"] = _erd(clean, table, spec, path, ctx, rec, log)

    es = None
    if "decode" in pipelines or "transfer" in pipelines:
        if no_trials is None:
            log(f"    epoching {len(table.trials)} trials ({opts.l_freq:g}-{opts.h_freq:g} Hz) ...")
            es = make_epochs(clean, table, spec, opts.l_freq, opts.h_freq, group=rec.id)
    if "decode" in pipelines:
        log("    decode ...")
        path = mk("decode")
        if es is None:
            _skipped(path, "decode", ctx, no_trials)
            row["decode"] = "skipped"
        else:
            try:
                res = decode_epochset(es, opts.n_perm, opts.folds, opts.repeats, opts.n_sham_draws,
                                      n_jobs=opts.n_jobs, min_per_class=opts.min_per_class,
                                      pipelines=model_factories(opts.models), log=log)
            except ValueError as exc:
                _skipped(path, "decode", ctx, f"not decodable: {exc} (class counts {es.class_counts})")
                row["decode"] = "skipped"
            else:
                _write_decode(path, "decode", res, ctx, f"{rec.id} strict decoding ({L.channels.label})")
                row["decode"] = _decode_row(res)
                log(f"    decode: {res['verdict']}")
    return es


def _erd(clean, table, spec, path, ctx, rec, log):
    out, lines = {}, rp.header("ERD% topomaps", ctx)
    w, b = spec.conditions.window, spec.conditions.baseline
    lines += [f"Task window anchor + [{w.tmin}, {w.tmax}] s; baseline "
              f"{'`' + b.event_regex + '`' if b.event_regex else 'anchor'} + [{b.tmin}, {b.tmax}] s. "
              "Average reference, Hilbert power. Black dots: channels with ERD != 0 after FDR over all "
              "(class x channel) cells, one-sample t-test on log(task/baseline).", ""]
    for band_name, rng in spec.bands.items():
        log(f"    erd_topomap {band_name} {rng[0]:g}-{rng[1]:g} Hz ...")
        res = erd(clean, table, spec, rng)
        if len(res.labels) < 3:
            lines.append(f"- {band_name}: only {len(res.labels)} usable trial(s); skipped")
            continue
        st = erd_stats(res)
        fig = rp.plot_erd(st, res, f"{rec.id} {band_name} {rng[0]:g}-{rng[1]:g} Hz ERD ({ctx['montage']})",
                          os.path.join(path, f"erd_{band_name}.png"))
        out[band_name] = {k: dict(n=v["n"], mean_erd=dict(zip(res.ch_names, v["mean_erd"])),
                                  p_fdr=dict(zip(res.ch_names, v["p_fdr"])),
                                  significant=[c for c, s in zip(res.ch_names, v["significant"]) if s])
                          for k, v in st.items()}
        out[band_name]["_n_cells_tested"] = len(st) * len(res.ch_names)
        motor = [c for c in MOTOR if c in res.ch_names]
        lines += [f"## {band_name} ({rng[0]:g}-{rng[1]:g} Hz)", "",
                  "| Condition | n | " + " | ".join(f"{c} ERD%" for c in motor) + " | Strongest drop | # FDR-sig ch |",
                  "| :--- | ---: |" + " ---: |" * len(motor) + " :--- | ---: |"]
        for k, v in st.items():
            i = int(np.nanargmin(v["mean_erd"]))
            lines.append(f"| {k} | {v['n']} | " + " | ".join(f"{v['mean_erd'][res.ch_names.index(c)]:+.1f}" for c in motor)
                         + f" | {res.ch_names[i]} {v['mean_erd'][i]:+.1f}% | {int(np.sum(v['significant']))} |")
        lines += ["", f"![{band_name}]({os.path.basename(fig)})", ""]
    rp.write_json(os.path.join(path, "erd_topomap.json"), out)
    rp.write_md(os.path.join(path, "erd_topomap.md"), lines)
    return "ok" if out else "skipped"


def _write_decode(path, name, res, ctx, title):
    rp.write_json(os.path.join(path, f"{name}.json"), res)
    fig = rp.plot_decode(res, title, os.path.join(path, f"{name}.png"))
    rp.write_md(os.path.join(path, f"{name}.md"), rp.header(title, ctx) + rp.decode_md(res)
                + [f"![{name}]({os.path.basename(fig)})"])


def _decode_row(res):
    best = max(((s, p, r) for s, d in res["cells"].items() for p, r in d.items()), key=lambda t: t[2]["acc"])
    sham = [v["random"].get("status") for v in res["sham"].values()]
    bd = res.get("blocked_design") or {}
    flag = "CONFOUNDED (class = time block): " if bd.get("status") == "CONFOUNDED" else ""
    return dict(best=f"{flag}{best[1]} / {best[0]}", acc=best[2]["acc"], chance=res["chance"],
                p_maxstat=best[2].get("p_maxstat"), sham="FAIL" if "FAIL" in sham else
                ("n/a" if all(s == "unavailable" for s in sham) else "pass"), cv=res["cv"],
                n_trials=res["quantification"]["n_trials"])


def _common_labels(sets, min_per_class):
    """Labels that every set has >= min_per_class trials of (sets with < 2 such labels are dropped first)."""
    usable = {k: {c for c, n in es.class_counts.items() if n >= min_per_class} for k, es in sets.items()}
    keep = {k: v for k, v in usable.items() if len(v) >= 2}
    common = sorted(set.intersection(*keep.values())) if keep else []
    return list(keep), common


def _run_transfer(dataset, bids_root, epoch_sets, spec, opts, log):
    by_sub = {}
    for rid, (rec, es) in epoch_sets.items():
        by_sub.setdefault(rec.sub, {})[rid] = es
    out = {f"sub-{s}": _transfer_unit(dataset, bids_root, f"sub-{s}", sets, "recording", spec, opts, log)
           for s, sets in sorted(by_sub.items())}
    # Cross-subject: each subject's recordings pooled, grouped by subject.
    subj_sets = {}
    for s, sets in sorted(by_sub.items()):
        try:
            pooled = concat(list(sets.values()))
        except ValueError:
            continue
        sid = f"sub-{s}"
        subj_sets[sid] = pooled._replace(groups=np.array([sid] * len(pooled.y), dtype=object),
                                         groups_sham=np.array([sid] * len(pooled.y_sham_prev), dtype=object))
    out["all_subjects"] = _transfer_unit(dataset, bids_root, "all_subjects", subj_sets, "subject", spec, opts, log)
    return out


def _transfer_unit(dataset, bids_root, unit, sets, level, spec, opts, log):
    params = _params(opts, bids_root, spec, dict(transfer_level=level, units=sorted(sets)))
    path = rp.out_dir(opts.results_root, "transfer", dataset, unit, params, opts.script_file)
    ctx = dict(dataset=dataset, unit=unit, montage="montage: see each recording's qc.md (assumed if index-named)",
               spec_source=spec.source, confounds=spec.confounds)
    keep, common = _common_labels(sets, opts.min_per_class)
    what = "recordings" if level == "recording" else "subjects"
    if len(keep) < 2 or len(common) < 2:
        reason = (f"needs >= 2 {what} sharing >= 2 classes with >= {opts.min_per_class} trials each; "
                  f"have {len(sets)} {what}, usable {keep}, common classes {common}")
        _skipped(path, "transfer", ctx, reason)
        log(f"  transfer {unit}: skipped ({reason})")
        return dict(status="skipped", reason=reason)
    try:
        log(f"  transfer {unit}: pooling {keep} on classes {common} ...")
        pooled = concat([sets[k].pick_labels(common) for k in keep])
        res = decode_epochset(pooled, opts.n_perm, opts.folds, opts.repeats, opts.n_sham_draws,
                              n_jobs=opts.n_jobs, min_per_class=opts.min_per_class,
                              pipelines=model_factories(opts.models), log=log)
    except ValueError as exc:
        _skipped(path, "transfer", ctx, str(exc))
        return dict(status="skipped", reason=str(exc))
    if not res["cv"].startswith("Leave-One-Group-Out"):
        reason = f"could not hold out whole {what} (cv fell back to {res['cv']})"
        _skipped(path, "transfer", ctx, reason)
        return dict(status="skipped", reason=reason)
    res["held_out_units"] = keep
    res["cv"] = res["cv"].replace("recordings", what)
    title = f"{dataset} {unit}: leave-one-{level}-out transfer"
    ctx["warnings"] = [f"held-out units: {keep}; common classes: {common}"]
    _write_decode(path, "transfer", res, ctx, title)
    log(f"  transfer {unit}: {res['verdict']}")
    row = _decode_row(res)
    row["status"] = "ok"
    return row


def _write_summary(summary, opts, spec):
    ds = summary["dataset"]
    path = rp.out_dir(opts.results_root, "summary", ds, "", _params(opts, summary["bids_root"], spec),
                      opts.script_file)
    rp.write_json(os.path.join(path, "summary.json"), summary)
    L = [f"# {ds}: pipeline summary", "", f"- Spec: {summary['spec_source']}",
         f"- Pipelines: {', '.join(summary['pipelines'])}"]
    L += [f"- > **Confound:** {c}" for c in spec.confounds] + ["",
          "| Recording | QC | Live ch | Alpha µV²/Hz | Classes | ERD | Decode best | Acc / chance | p max-stat | Sham |",
          "| :--- | :--- | ---: | ---: | :--- | :--- | :--- | :--- | ---: | :--- |"]
    for rid, r in summary["recordings"].items():
        if "error" in r:
            L.append(f"| {rid} | ERROR | | | | | {r['error']} | | | |")
            continue
        d = r.get("decode")
        dec = (f"| {d['best']} | {d['acc']:.1%} / {d['chance']:.0%} | "
               f"{d['p_maxstat']:.3f} | {d['sham']} |" if isinstance(d, dict) else f"| {d or '-'} | | | |")
        alpha = "n/a" if r.get("alpha_uv2_hz") is None else f"{r['alpha_uv2_hz']:.3g}"
        L.append(f"| {rid} | {r.get('qc_status')} | {r.get('n_live')}/{r.get('n_eeg')} | {alpha} | "
                 f"{r.get('class_counts', '-')} | {r.get('erd_topomap', '-')} " + dec)
    if summary["transfer"]:
        L += ["", "## Transfer", "", "| Unit | Status | Best | Acc / chance | p max-stat | Sham | CV |",
              "| :--- | :--- | :--- | :--- | ---: | :--- | :--- |"]
        for unit, t in summary["transfer"].items():
            if t.get("status") != "ok":
                L.append(f"| {unit} | {t.get('status')} | {t.get('reason', '')} | | | | |")
            else:
                L.append(f"| {unit} | ok | {t['best']} | {t['acc']:.1%} / {t['chance']:.0%} | "
                         f"{t['p_maxstat']:.3f} | {t['sham']} | {t['cv']} |")
    refused = [rid for rid, r in summary["recordings"].items() if r.get("refused")]
    if refused:
        L += ["", f"**QC refused {len(refused)} recording(s):** " + "; ".join(
            f"{rid} ({'; '.join(summary['recordings'][rid]['qc_reasons'])})" for rid in refused)]
    rp.write_md(os.path.join(path, "summary.md"), L)
    return path
