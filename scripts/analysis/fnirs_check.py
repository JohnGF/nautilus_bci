#!/usr/bin/env python3
"""
Record fNIRS checks now, analyse them all together later.

    uv run python analysis/fnirs_check.py record --label after-reseat
    uv run python analysis/fnirs_check.py record --label after-reseat-2 --seconds 60
    uv run python analysis/fnirs_check.py report

record only saves data, from the running fNIRS bridge and the watch if it is streaming.
report analyses today's recordings using the viewer's own pulse and contact code, prints
the result and saves it beside the recordings. Use --all or --last N for other recordings.

Recordings are saved in fnirs_checks/ beside the nautilus_bci repo, because the repo is public.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visualizers"))

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

import fnirs_visualizer as fv  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "fnirs_checks"
WINDOW_SEC = 20.0  # The viewer's window
STEP_SEC = 5.0
FIND_TIMEOUT_SEC = 40.0
WATCH_LAGS_SEC = (0, 5, 10, 15)
AGREE_BPM = 6.0  # Resolution of the pulse peak
# The smallest change in OD tracks absolute light, because the device's counts are whole numbers.
# Fitted on one recording, then predicted startup OD within 0.08 on 8 channels from 1.0 to 1.8.
LIGHT_SLOPE, LIGHT_OFFSET = 0.97, -5.14


def find_stream(resolve, wanted, timeout):
    """A single search can miss a stream on another PC, so keep trying."""
    deadline = time.monotonic() + timeout
    while True:
        match = next((s for s in resolve(2.0) if wanted(s)), None)
        if match is not None or time.monotonic() >= deadline:
            return match


def is_watch(stream):
    return fv.is_watch_heart_rate_stream(stream.name(), stream.type())


def is_fnirs(stream):
    return fv.is_fnirs_stream(stream.name(), stream.type())


def stream_description(info):
    labels, units, startup = [], [], []
    channel = info.desc().child("channels").child("channel")
    while not channel.empty():
        labels.append(channel.child_value("label"))
        units.append(channel.child_value("unit"))
        startup.append(channel.child_value("startup_od"))
        channel = channel.next_sibling("channel")
    return labels, units, startup


def record(label, seconds, out):
    from pylsl import StreamInlet, resolve_streams

    print("Looking for the fNIRS bridge...", flush=True)
    fnirs = find_stream(resolve_streams, is_fnirs, FIND_TIMEOUT_SEC)
    if fnirs is None:
        print("No fNIRS stream found. Start 'ssh artinis (od)' in the pipeline dashboard, then try again.")
        return 1
    watch = find_stream(resolve_streams, is_watch, 6.0)
    print("Watch: " + (watch.name() if watch else "not streaming, recording without it"), flush=True)

    inlet = StreamInlet(fnirs, max_buflen=int(seconds) + 30)
    info = inlet.info(timeout=5.0)
    labels, units, startup = stream_description(info)
    srate = info.nominal_srate()
    watch_inlet = StreamInlet(watch, max_buflen=int(seconds) + 30) if watch else None

    frames, frame_times, heart, heart_times = [], [], [], []
    started = datetime.now()
    begin = time.monotonic()
    print(f"Recording {seconds:.0f} s. Keep still. Ctrl+C stops early and keeps what was recorded.", flush=True)
    try:
        while time.monotonic() - begin < seconds:
            chunk, stamps = inlet.pull_chunk(timeout=0.2)
            frames += chunk
            frame_times += stamps
            if watch_inlet:
                chunk, stamps = watch_inlet.pull_chunk(timeout=0.0)
                heart += [sample[0] for sample in chunk]
                heart_times += stamps
    except KeyboardInterrupt:
        print("\nStopped early.")
    elapsed = time.monotonic() - begin
    if not frames:
        print("Nothing arrived from the bridge, so nothing was saved. Is the cap on the head?")
        return 1

    # Both clocks on this PC's time, so the watch lines up with the fNIRS
    fnirs_offset = inlet.time_correction(timeout=3.0)
    watch_offset = watch_inlet.time_correction(timeout=3.0) if watch_inlet else 0.0

    out.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^\w-]+", "-", label).strip("-") or "run"
    path = out / f"{started:%Y%m%d_%H%M%S}_{safe_label}.npz"
    meta = {"label": label, "started": started.isoformat(timespec="seconds"), "srate": srate,
            "labels": labels, "units": units, "startup_od": startup, "watch": watch.name() if watch else None}
    np.savez_compressed(path, frames=np.asarray(frames, dtype=np.float32),
                        times=np.asarray(frame_times) + fnirs_offset,
                        heart=np.asarray(heart, dtype=np.float32), heart_times=np.asarray(heart_times) + watch_offset,
                        meta=json.dumps(meta))
    rate = len(frames) / max(elapsed, 1e-3)
    print(f"Saved {path}\n{len(frames)} frames at {rate:.1f} Hz, {len(heart)} watch readings.")
    if rate < 0.9 * srate:
        print(f"Note: expected {srate:.0f} Hz. The stream paused, which usually means the optodes came off the head.")
    print("Record again, or run 'report' when you're finished.")
    return 0


def estimated_light(first):
    """Absolute OD per channel from the signal's smallest step. NaN if the signal never moves."""
    light = []
    for channel in first.T:
        steps = np.unique(np.round(np.abs(np.diff(channel)), 6))
        steps = steps[steps > 0][:3]
        light.append((np.log10(steps.mean()) - LIGHT_OFFSET) / LIGHT_SLOPE if len(steps) else np.nan)
    return np.array(light)


def compare_with_watch(centres, peaks_bpm, heart_times, heart):
    """Share of pulse peaks within AGREE_BPM of the watch, and their correlation, with the watch read later by each lag."""
    results = {}
    usable = heart > 0  # The watch sends 0 while it has no reading
    heart_times, heart = heart_times[usable], heart[usable]
    half = WINDOW_SEC / 2.0
    for lag in WATCH_LAGS_SEC:
        watched = []
        for centre in centres:
            inside = (heart_times >= centre + lag - half) & (heart_times <= centre + lag + half)
            watched.append(heart[inside].mean() if inside.sum() >= 0.75 * WINDOW_SEC else np.nan)
        watched = np.array(watched)
        both = np.isfinite(watched) & np.isfinite(peaks_bpm)
        if both.sum() < 3:
            results[lag] = None
            continue
        spread = peaks_bpm[both].std() > 0 and watched[both].std() > 0
        results[lag] = {
            "agree": float(np.mean(np.abs(peaks_bpm[both] - watched[both]) <= AGREE_BPM)),
            "r": float(np.corrcoef(peaks_bpm[both], watched[both])[0, 1]) if spread else np.nan,
            "n": int(both.sum()),
        }
    return results


def analyse(path):
    with np.load(path, allow_pickle=False) as saved:
        frames, times = saved["frames"].astype(np.float64), saved["times"]
        heart, heart_times = saved["heart"].astype(np.float64), saved["heart_times"]
        meta = json.loads(str(saved["meta"]))
    srate = meta["srate"]
    layout = fv.parse_stream_layout(meta["labels"], meta["units"], frames.shape[1])
    first, second = frames[:, layout.first], frames[:, layout.second]
    window, step = int(srate * WINDOW_SEC), int(srate * STEP_SEC)
    sos = signal.butter(2, [0.8 / (srate / 2), 2.0 / (srate / 2)], btype="bandpass", output="sos")

    sci_all, strength_all, contact_all, peaks, centres = [], [], [], [], []
    for start in range(0, len(frames) - window + 1, step):
        part_first, part_second = first[start:start + window], second[start:start + window]
        sci = fv.scalp_coupling_index(part_first, part_second, sos, int(srate * 5.0))
        pulse = fv.pulse_metrics(part_first, part_second, srate, sos)
        sci_all.append(sci)
        strength_all.append(pulse.strength)
        contact_all.append(fv.contact_score(sci, pulse))
        peaks.append(pulse.heart_hz * 60.0)
        centres.append(times[start + window // 2])

    result = {"path": path, "meta": meta, "seconds": len(frames) / srate, "windows": len(centres),
              "rate": len(frames) / max(times[-1] - times[0], 1e-3), "biggest_gap": float(np.diff(times).max()),
              "biggest_jump": float(np.abs(np.diff(frames, axis=0)).max()), "light": estimated_light(first)}
    if not centres:
        return result
    contact_all = np.array(contact_all)
    peaks = np.array(peaks)
    usable = heart[heart > 0]
    result.update({
        "sci": np.median(sci_all, axis=0), "strength": np.median(strength_all, axis=0),
        "contact": np.median(contact_all, axis=0), "good_share": np.mean(contact_all >= fv.GOOD_SCI, axis=0),
        "peaks": peaks, "watch_range": (usable.min(), usable.max()) if len(usable) else None,
        "watch": compare_with_watch(np.array(centres), peaks, heart_times, heart),
    })
    return result


def format_run(run):
    meta = run["meta"]
    lines = [f"{meta['started'].replace('T', ' ')}  {meta['label']}",
             f"  {run['seconds']:.0f} s at {run['rate']:.1f} Hz, longest gap {run['biggest_gap']:.2f} s, "
             f"biggest one-frame jump {run['biggest_jump']:.3f} OD"]
    if not run["windows"]:
        lines.append(f"  Too short to analyse: needs at least {WINDOW_SEC:.0f} s.")
        return lines
    sites = fv.MONTAGES["FRONTAL"]["sites"]
    lines.append("  Channel  Site   Light(OD)  Pulse   SCI  Contact  Time good")
    for ch in range(len(run["contact"])):
        lines.append(f"  T{ch + 1:<7} {sites[ch]:<6} {run['light'][ch]:>8.2f} {run['strength'][ch]:>6.1f} "
                     f"{run['sci'][ch]:>5.2f} {run['contact'][ch]:>8.2f} {100 * run['good_share'][ch]:>9.0f}%")
    found = np.isfinite(run["peaks"])
    if found.any():
        low, high = np.nanmin(run["peaks"]), np.nanmax(run["peaks"])
        span = f"{low:.0f}" if low == high else f"{low:.0f}-{high:.0f}"
        lines.append(f"  Pulse peak: {span} bpm, found in {found.sum()} of {len(found)} windows")
    else:
        lines.append(f"  Pulse peak: none found in {len(found)} windows")
    if run["watch_range"] is None:
        lines.append("  Watch: no readings")
    else:
        low, high = run["watch_range"]
        lines.append(f"  Watch: {low:.0f}-{high:.0f} bpm. Pulse peak within {AGREE_BPM:.0f} bpm of it, by how much later the watch is read:")
        for lag, score in run["watch"].items():
            text = "too little data" if score is None else f"{100 * score['agree']:3.0f}% of {score['n']} windows, r={score['r']:+.2f}"
            lines.append(f"    {lag:>2} s later: {text}")
    return lines


def format_summary(runs):
    usable = [run for run in runs if run["windows"]]
    if not usable:
        return []
    lines = ["ACROSS ALL RECORDINGS", "  Channel  Site   Contact by recording" + " " * max(0, 6 * len(usable) - 20) + "  Recordings good"]
    sites = fv.MONTAGES["FRONTAL"]["sites"]
    for ch in range(len(usable[0]["contact"])):
        values = [run["contact"][ch] for run in usable]
        good = sum(value >= fv.GOOD_SCI for value in values)
        lines.append(f"  T{ch + 1:<7} {sites[ch]:<6} " + " ".join(f"{value:5.2f}" for value in values) +
                     f"   {good} of {len(values)}")
    scored = [run for run in usable if run["watch_range"] is not None]
    if scored:
        lines.append("  Pulse peak against the watch, all recordings together:")
        for lag in WATCH_LAGS_SEC:
            scores = [run["watch"][lag] for run in scored if run["watch"][lag]]
            count = sum(score["n"] for score in scores)
            if count:
                agree = sum(score["agree"] * score["n"] for score in scores) / count
                lines.append(f"    {lag:>2} s later: {100 * agree:3.0f}% of {count} windows")
        lines.append("    Windows overlap, so these look more certain than they are.")
    lines.append(f"  Contact is the median across each recording. Good means {fv.GOOD_SCI} or more.")
    lines.append("  Sites assume the frontal layout. Light is an estimate from the signal's step size.")
    return lines


def report(out, last, show_all):
    paths = sorted(out.glob("*.npz"))
    if not show_all and last is None:
        today = f"{datetime.now():%Y%m%d}_"
        paths = [path for path in paths if path.name.startswith(today)]
    elif last is not None:
        paths = paths[-last:]
    if not paths:
        print(f"No recordings to report in {out}. Run 'record' first, or use --all.")
        return 1

    runs = [analyse(path) for path in paths]
    lines = [f"fNIRS check report, {len(runs)} recording(s)", ""]
    for run in runs:
        lines += format_run(run) + [""]
    lines += format_summary(runs)
    text = "\n".join(lines)
    print(text)
    saved = out / f"report_{datetime.now():%Y%m%d_%H%M%S}.txt"
    saved.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved {saved}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Folder for recordings and reports")
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", help="Save one recording")
    rec.add_argument("--label", default="run", help="A name for this recording, such as after-reseat")
    rec.add_argument("--seconds", type=float, default=120.0, help="Length, default 120")
    rep = commands.add_parser("report", help="Analyse today's recordings together")
    rep.add_argument("--last", type=int, help="The latest N recordings, from any day")
    rep.add_argument("--all", action="store_true", help="Every recording")
    args = parser.parse_args(argv)
    if args.command == "record":
        return record(args.label, args.seconds, args.out)
    return report(args.out, args.last, args.all)


if __name__ == "__main__":
    raise SystemExit(main())
