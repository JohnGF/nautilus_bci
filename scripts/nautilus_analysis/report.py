"""Outputs: folders, JSON, Markdown and figures for every pipeline.

Layout (plan invariant 6): ``<results_root>/<pipeline>/<dataset>/<unit>/``,
where ``<unit>`` is a recording id without the ``sub-``/``ses-``/``task-``
keys (``sub-01_ses-02_task-x`` -> ``01_02_x``), ``sub-XX`` for per-subject
transfer, or ``all_subjects``. Each folder gets ``RUN_INFO.txt``
(``utils.provenance.write_provenance``), ``<pipeline>.json``,
``<pipeline>.md`` and PNG figures. Every figure title and report header
carries the montage label (montage honesty).

Colours are the dataviz reference palette's light-mode slots. Text uses ink
tokens, never series colours.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from utils.provenance import write_provenance  # noqa: E402

from .decode import CHANCE_MARGIN  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"


def unit_name(recording_id):
    """'sub-01_ses-02_task-x_run-1' -> '01_02_x_1' (the plan's ``<sub>_<ses>_<task>[_run]``)."""
    return "_".join(p.split("-", 1)[1] if "-" in p else p for p in recording_id.split("_"))


def out_dir(results_root, pipeline, dataset, unit, params=None, script_file=None):
    path = os.path.join(results_root, pipeline, dataset, unit)
    os.makedirs(path, exist_ok=True)
    write_provenance(path, script_file=script_file or __file__, params=params or {})
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path, data):
    with open(path, "w") as fh:
        json.dump(_jsonable(data), fh, indent=2)
    return path


def write_md(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    return path


def header(title, ctx):
    """Markdown header shared by all pipelines. ``ctx`` keys: dataset, unit, montage, spec_source, confounds, qc."""
    lines = [f"# {title}", "", f"- Dataset: `{ctx['dataset']}`, unit: `{ctx['unit']}`",
             f"- **{ctx['montage']}**", f"- Spec: {ctx['spec_source']}"]
    if ctx.get("qc"):
        lines.append(f"- QC: {ctx['qc']}")
    for c in ctx.get("confounds") or []:
        lines.append(f"- > **Confound:** {c}")
    for w in ctx.get("warnings") or []:
        lines.append(f"- Warning: {w}")
    return lines + [""]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, path):
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def _info(ch_names, sfreq=250.0):
    info = mne.create_info(list(ch_names), sfreq, "eeg")
    info.set_montage("standard_1020", match_case=False, on_missing="ignore")
    return info


def _has_positions(info):
    return all(np.isfinite(ch["loc"][:3]).all() and np.any(ch["loc"][:3]) for ch in info["chs"])


# --------------------------------------------------------------------------- figures

def plot_qc(report, montage_label, path, flat_std_uv):
    names = list(report.channel_std_uv)
    std = np.array([report.channel_std_uv[n] for n in names])
    fig, ax = plt.subplots(figsize=(max(6, 0.28 * len(names)), 3.2), facecolor=SURFACE)
    colors = [MUTED if n in report.flat_channels else SERIES[0] for n in names]
    ax.bar(range(len(names)), np.maximum(std, 1e-3), color=colors, width=0.75)
    ax.axhline(flat_std_uv, color=INK_2, lw=1, ls="--")
    ax.text(len(names) - 0.5, flat_std_uv, f" flat < {flat_std_uv:g} µV", color=INK_2, fontsize=8, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Channel std (µV)", color=INK_2, fontsize=9)
    _style(ax)
    ax.set_title(f"{report.recording_id}: {report.n_live}/{report.n_eeg} live, status={report.status}  "
                 f"({montage_label}; grey = flat)", fontsize=9.5, color=INK)
    return _save(fig, path)


def plot_psd(freqs, pxx, bp, ch_names, bands, title, path):
    has_topo = len(ch_names) >= 8 and _has_positions(_info(ch_names))
    n_top = len(bp) if has_topo else 0
    fig = plt.figure(figsize=(4.5 + 2.4 * n_top, 3.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 1 + n_top, width_ratios=[2.2] + [1] * n_top, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    logp = 10 * np.log10(np.maximum(pxx, 1e-12))
    ax.fill_between(freqs, np.percentile(logp, 25, axis=0), np.percentile(logp, 75, axis=0),
                    color=SERIES[0], alpha=0.18, lw=0)
    ax.plot(freqs, np.median(logp, axis=0), color=SERIES[0], lw=2)
    for name, (lo, hi) in bands.items():
        for edge in (lo, hi):
            ax.axvline(edge, color=MUTED, lw=0.8, ls=":")
        ax.text((lo + hi) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=INK_2)
    ax.set_xlabel("Frequency (Hz)", color=INK_2, fontsize=9)
    ax.set_ylabel("PSD (dB re 1 µV²/Hz)\nmedian, IQR over channels", color=INK_2, fontsize=9)
    _style(ax)
    if has_topo:
        info = _info(ch_names)
        for k, (name, vals) in enumerate(bp.items()):
            tax = fig.add_subplot(gs[0, 1 + k])
            im, _ = mne.viz.plot_topomap(10 * np.log10(np.maximum(vals, 1e-12)), info, axes=tax, show=False,
                                         cmap="Blues", contours=0)
            tax.set_title(f"{name} {bands[name][0]:g}-{bands[name][1]:g} Hz", fontsize=9, color=INK)
            fig.colorbar(im, ax=tax, shrink=0.7).set_label("dB re 1 µV²/Hz", fontsize=8, color=INK_2)
    fig.suptitle(title, fontsize=10, color=INK, y=1.04)
    return _save(fig, path)


def plot_erd(stats, res, title, path, tc_channels=("C3", "Cz", "C4", "Pz")):
    """Topomaps of mean ERD% for 'all' and each class (FDR-significant channels dotted) + time course."""
    info = _info(res.ch_names)
    keys = list(stats)
    vals = np.concatenate([np.nan_to_num(stats[k]["mean_erd"]) for k in keys])
    lim = float(max(5.0, np.ceil(np.percentile(np.abs(vals), 95) / 5) * 5))
    n = len(keys)
    fig = plt.figure(figsize=(2.6 * max(n, 3) + 0.8, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, max(n, 3), hspace=0.45, wspace=0.3)
    im = None
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        mask_params = dict(marker="o", markerfacecolor=INK, markeredgecolor=SURFACE, markersize=4)
        im, _ = mne.viz.plot_topomap(np.nan_to_num(stats[k]["mean_erd"]), info, axes=ax, show=False,
                                     cmap="RdBu_r", vlim=(-lim, lim), contours=6,
                                     mask=np.asarray(stats[k]["significant"], bool), mask_params=mask_params)
        ax.set_title(f"{k} (n={stats[k]['n']})\n{int(np.sum(stats[k]['significant']))} ch FDR<.05",
                     fontsize=9, color=INK)
    cax = fig.add_axes([0.92, 0.58, 0.01, 0.28])
    fig.colorbar(im, cax=cax).set_label("ERD % vs baseline\n(blue = power drop)", fontsize=8, color=INK_2)

    ax = fig.add_subplot(gs[1, :])
    chans = [c for c in tc_channels if c in res.ch_names]
    if len(chans) < 2:  # no motor channels: show the strongest ERD channels
        order = np.argsort(np.nan_to_num(stats["all"]["mean_erd"]))
        chans = [res.ch_names[i] for i in order[:4]]
    k_smooth = max(1, int(0.25 * (len(res.t_rel) / max(res.t_rel[-1] - res.t_rel[0], 1e-9))))
    tcs = res.mean_timecourse()
    for color, ch in zip(SERIES, chans):
        tc = np.convolve(tcs[res.ch_names.index(ch)], np.ones(k_smooth) / k_smooth, mode="same")
        ax.plot(res.t_rel, tc, color=color, lw=2, label=ch)
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Time from anchor event (s)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"{res.band[0]:g}-{res.band[1]:g} Hz power change (%)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=len(chans), loc="lower left")
    _style(ax)
    ax.set_title("Time course, all trials (0.25 s smoothing; dashed = anchor)", fontsize=9, color=INK)
    fig.suptitle(title, fontsize=10.5, color=INK)
    return _save(fig, path)


def plot_decode(res, title, path):
    """Grouped bars: accuracy per channel set, one bar per pipeline; chance line, null-95th ticks, sham panel."""
    sets = list(res["cells"])
    pipes = list(next(iter(res["cells"].values())))
    fig, (ax, sax) = plt.subplots(1, 2, figsize=(3 + 1.8 * len(sets) + 2.2, 3.8), facecolor=SURFACE,
                                  gridspec_kw=dict(width_ratios=[len(sets) + 0.5, 1.4]), sharey=True)
    w = 0.8 / len(pipes)
    for pi, (pname, color) in enumerate(zip(pipes, SERIES)):
        for si, s in enumerate(sets):
            r = res["cells"][s][pname]
            x = si + (pi - (len(pipes) - 1) / 2) * w
            ax.bar(x, r["acc"] * 100, w - 0.04, color=color, label=pname if si == 0 else None)
            if "null_p95" in r:
                ax.plot([x - w / 2.5, x + w / 2.5], [r["null_p95"] * 100] * 2, color=INK, lw=2)
            star = "*" if r.get("p_maxstat", 1) < 0.05 else ""
            ax.text(x, r["acc"] * 100 + 1, f"{r['acc'] * 100:.0f}{star}", ha="center", va="bottom", fontsize=8, color=INK)
    chance = res["chance"] * 100
    for a in (ax, sax):
        a.axhline(chance, color=INK_2, lw=1, ls="--")
        _style(a)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets, fontsize=8.5, color=INK_2)
    ax.set_ylabel("Balanced accuracy (%)", color=INK_2, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(f"{res['cv']}; tick = null 95th pct; * = max-stat p<.05", fontsize=8.5, color=INK_2)
    for pi, (pname, color) in enumerate(zip(pipes, SERIES)):
        sh = res["sham"].get(pname, {}).get("random", {})
        if "acc" in sh:
            sax.bar(pi, sh["acc"] * 100, 0.7, color=color)
            sax.text(pi, sh["acc"] * 100 + 1, f"{sh['acc'] * 100:.0f}\n{sh['status']}", ha="center", va="bottom",
                     fontsize=7.5, color=INK)
        else:
            sax.text(pi, chance + 3, "n/a", ha="center", fontsize=8, color=INK_2)
    sax.set_xticks(range(len(pipes)))
    sax.set_xticklabels([p.split()[0] for p in pipes], fontsize=8, color=INK_2)
    sax.set_title("Sham Rest\n(random labels)", fontsize=8.5, color=INK_2)
    fig.suptitle(title, fontsize=10, color=INK, y=1.03)
    return _save(fig, path)


# --------------------------------------------------------------------------- markdown tables

def decode_md(res):
    q = res["quantification"]
    L = ["## Data quantification", "",
         "| Wear time (s) | Trials | Active task (s) | Class counts | Sham epochs | Channels | Fit wall (s) |",
         "| ---: | ---: | ---: | :--- | ---: | ---: | ---: |",
         f"| {q['wear_time_s']:.0f} | {q['n_trials']} | {q['active_task_s']:.0f} | {res['class_counts']} | "
         f"{q['n_sham_epochs']} | {q['n_channels']} | {q.get('fit_wall_s', '')} |", ""]
    if res.get("dropped_small_classes"):
        L += [f"Classes dropped for having too few trials: {res['dropped_small_classes']}", ""]
    mc = res["multiple_comparisons"]
    L += [f"## Decoding ({res['cv']}, {res['scoring']}, chance {res['chance']:.1%})", "",
          f"{mc['n_cells']} cells tested; correction: {mc['method']}.", "",
          "| Channels | Pipeline | Acc | Fold SD | Null mean [95% CI] | p | p max-stat | p FDR |",
          "| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: |"]
    for s, d in res["cells"].items():
        for p, r in d.items():
            null = (f"{r['null_mean']:.1%} [{r['null_ci95'][0]:.1%}, {r['null_ci95'][1]:.1%}]"
                    if "null_mean" in r else "n/a")
            L.append(f"| {s} | {p} | {r['acc']:.1%} | {r['fold_sd']:.1%} | {null} | {r.get('p_value', float('nan')):.3f} | "
                     f"{r.get('p_maxstat', float('nan')):.3f} | {r.get('p_fdr', float('nan')):.3f} |")
    L += ["", f"## Sham Rest controls (all channels; must sit within chance + {CHANCE_MARGIN:.0%})", "",
          "| Pipeline | Random pseudo-labels | Status | Carry-over (prev. trial label) | Carry-over p | Status |",
          "| :--- | ---: | :--- | ---: | ---: | :--- |"]
    for p, sh in res["sham"].items():
        r, c = sh["random"], sh["carry_over"]
        ra = f"{r['acc']:.1%} ± {r['sd']:.1%}" if "acc" in r else r.get("reason", "")
        ca = f"{c['acc']:.1%}" if "acc" in c else c.get("reason", "")
        cp = f"{c['p_value']:.3f}" if "p_value" in c else ""
        L.append(f"| {p} | {ra} | {r['status']} | {ca} | {cp} | {c['status']} |")
    L += ["", f"**Verdict:** {res['verdict']}", ""]
    return L
