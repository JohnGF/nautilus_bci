"""
Offline wearable clock refit
============================

The wearable bridge maps each device's clock onto LSL time live, using only the
ping/echo exchanges seen so far. After a recording, the whole session's exchanges
are known, so each sample can be placed using exchanges from both before and after
it. That is more accurate, especially early in a session and when drift changes
with temperature.

Reads what multimodal_bids_recorder.py puts in sourcedata/:
  <category>/*_devicetime.tsv   live timestamp_sec + raw device clock per sample
  clocksync/*_clocksync.tsv     ping/echo exchanges per device

Usage:
  uv run python recorders/refit_wearable_clocks.py bids_dataset_multimodal          # report only
  uv run python recorders/refit_wearable_clocks.py bids_dataset_multimodal --apply  # rewrite timestamp_sec

--apply rewrites timestamp_sec in each wearable TSV and records the refit in its JSON.
The live timestamps stay in the devicetime file, so applying again gives the same result.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

BURST_GAP_S = 2.0      # exchanges further apart than this belong to different bursts
WINDOW_S = 300.0       # centred window for the local offset + drift fit


def burst_points(exchanges):
    """Fastest exchange of each ping burst -> arrays (device time, offset, delay)."""
    t1, t2, t3, t4 = (exchanges[c].to_numpy() for c in
                      ('t1_ping_sent', 't2_ping_received', 't3_echo_sent', 't4_echo_received'))
    order = np.argsort(t1)
    t1, t2, t3, t4 = t1[order], t2[order], t3[order], t4[order]
    offset = ((t1 - t2) + (t4 - t3)) / 2
    delay = (t4 - t1) - (t3 - t2)
    starts = np.flatnonzero(np.diff(t1, prepend=-np.inf) > BURST_GAP_S)
    best = [s + int(np.argmin(delay[s:e])) for s, e in zip(starts, np.append(starts[1:], len(t1)))]
    return t2[best], offset[best], delay[best]


def fit_offsets(x, y, device_times, window_s=WINDOW_S):
    """Offset (recording time - device time) at each device time, and the device clock's
    rate error (positive = device runs fast).

    A local line is fitted through the burst points within +-window_s/2 of each point
    (so each one uses exchanges before and after it), then evaluated between points by
    linear interpolation and beyond the ends along the end lines."""
    if len(x) == 1:
        return np.full(len(device_times), y[0]), 0.0
    smooth = np.empty(len(x))
    slopes = np.empty(len(x))
    for i, xi in enumerate(x):
        near = np.abs(x - xi) <= window_s / 2
        if near.sum() >= 2 and np.ptp(x[near]) > 0:
            slopes[i], intercept = np.polyfit(x[near] - xi, y[near], 1)
            smooth[i] = intercept
        else:
            slopes[i], smooth[i] = 0.0, y[i]
    out = np.interp(device_times, x, smooth)
    before, after = device_times < x[0], device_times > x[-1]
    out[before] = smooth[0] + slopes[0] * (device_times[before] - x[0])
    out[after] = smooth[-1] + slopes[-1] * (device_times[after] - x[-1])
    slope = np.polyfit(x - x[0], y, 1)[0] if np.ptp(x) > 0 else 0.0
    return out, -slope  # offset shrinks when the device clock runs fast; report fast as positive


def find_clocksync(session_source_dir, device_id):
    for path in glob.glob(os.path.join(session_source_dir, "clocksync", "*_clocksync.json")):
        with open(path) as f:
            if json.load(f).get("DeviceID") == device_id:
                return path.replace('.json', '.tsv')
    return None


def refit_session_stream(bids_root, devicetime_json, apply=False):
    with open(devicetime_json) as f:
        dev_meta = json.load(f)
    session_source_dir = os.path.dirname(os.path.dirname(devicetime_json))
    sync_tsv = find_clocksync(session_source_dir, dev_meta.get("DeviceID"))
    name = dev_meta.get("StreamName", os.path.basename(devicetime_json))
    if sync_tsv is None:
        return {"stream": name, "status": "no clock-sync exchanges for this device"}

    dev = pd.read_csv(devicetime_json.replace('.json', '.tsv'), sep='\t')
    x, y, delay = burst_points(pd.read_csv(sync_tsv, sep='\t'))
    offsets, drift = fit_offsets(x, y, dev['device_time_s'].to_numpy())
    refit = dev['device_time_s'].to_numpy() + offsets
    change = refit - dev['timestamp_sec'].to_numpy()
    report = {
        "stream": name,
        "status": "applied" if apply else "report only",
        "bursts": int(len(x)),
        "device_drift_ppm": round(float(drift) * 1e6, 2),  # positive = device clock runs fast
        "median_bound_ms": round(float(np.median(delay) / 2 * 1e3), 3),
        "change_vs_live_ms": {"median": round(float(np.median(np.abs(change)) * 1e3), 3),
                              "max": round(float(np.max(np.abs(change)) * 1e3), 3)},
    }

    if apply:
        data_tsv = os.path.join(bids_root, dev_meta["DataFile"])
        data = pd.read_csv(data_tsv, sep='\t')
        if len(data) != len(refit):
            return {**report, "status": f"skipped: {len(data)} rows in data, {len(refit)} in devicetime"}
        data['timestamp_sec'] = refit
        data.to_csv(data_tsv, sep='\t', index=False)
        sidecar_path = data_tsv.replace('.tsv', '.json')
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        sidecar["ClockRefit"] = {
            "Method": "offline ping/echo: fastest exchange per burst, local linear fit over "
                      f"+-{WINDOW_S / 2:.0f} s, linear interpolation between bursts",
            "ClockSyncFile": os.path.relpath(sync_tsv, bids_root),
            "Bursts": report["bursts"],
            "DeviceDriftPPM": report["device_drift_ppm"],
            "MedianErrorBoundMs": report["median_bound_ms"],
            "LiveTimestampsIn": os.path.relpath(devicetime_json.replace('.json', '.tsv'), bids_root),
        }
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=2)
    return report


def refit_dataset(bids_root, apply=False):
    pattern = os.path.join(bids_root, "sourcedata", "sub-*", "ses-*", "*", "*_devicetime.json")
    return [refit_session_stream(bids_root, path, apply) for path in sorted(glob.glob(pattern))]


def main():
    parser = argparse.ArgumentParser(description="Refit wearable timestamps offline from ping/echo exchanges")
    parser.add_argument("bids_root", help="BIDS root written by multimodal_bids_recorder.py")
    parser.add_argument("--apply", action="store_true", help="Rewrite timestamp_sec in the wearable TSVs")
    args = parser.parse_args()

    reports = refit_dataset(os.path.abspath(args.bids_root), apply=args.apply)
    if not reports:
        print("[-] No *_devicetime.json files under sourcedata/. Record with the W2 wearable bridge first.")
    for r in reports:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
