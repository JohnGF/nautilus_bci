#!/usr/bin/env python3
"""
Dataset-agnostic analysis CLI ("if it's BIDS, it runs")
=======================================================
Runs the generic pipelines of ``nautilus_analysis`` on any BIDS dataset(s):

    cd scripts
    uv run python -m analysis.run --bids-root bids/music_lrud --pipeline qc,topomap,decode
    uv run python analysis/run.py --bids-root bids/music_lrud bids_clean/bids_fnf --pipeline all

Pipelines (in run order): qc, psd (alias bandpower), erd_topomap (aliases topomap, erd),
decode (strict; alias strict), transfer (leave-one-recording-out per subject and
leave-one-subject-out), or ``all``. ``--sub/--ses/--task`` omitted = every recording.

Dataset specifics come only from the analysis spec (``--spec``, then
``<bids_root>/code/nautilus_spec.json``, then ``analysis/specs/<dataset>.json``,
else inferred from events.tsv with a draft saved to ``results/spec/<dataset>/``).
Outputs: ``results/<pipeline>/<dataset>/<sub>_<ses>_<task>[_run]/`` (+ RUN_INFO.txt),
and ``results/summary/<dataset>/summary.md``. Plan: docs/plans/BIDS_AGNOSTIC_PIPELINES.md.
"""

import argparse
import os
import sys

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")  # also silences joblib workers (CSP covariance logs)
_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from nautilus_analysis.pipelines import PIPELINES, RunOptions, parse_pipelines, run_dataset  # noqa: E402
from nautilus_analysis.qc import QCThresholds  # noqa: E402
from utils.paths import RESULTS_DIR  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="Run dataset-agnostic EEG pipelines on BIDS datasets.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bids-root", nargs="+", required=True, help="one or more BIDS dataset roots")
    p.add_argument("--pipeline", default="all", help=f"comma list of {PIPELINES} or 'all'")
    p.add_argument("--sub", nargs="+", help="subject label(s); default all")
    p.add_argument("--ses", nargs="+", help="session label(s); default all")
    p.add_argument("--task", nargs="+", help="task label(s); default all")
    p.add_argument("--spec", help="analysis spec JSON (overrides the dataset/repo spec); only with one --bids-root")
    p.add_argument("--out-dir", default=RESULTS_DIR, help="results root")
    p.add_argument("--min-live-channels", type=int, default=8, help="QC gate for psd/decode/transfer")
    p.add_argument("--min-live-topomap", type=int, default=16, help="QC gate for erd_topomap")
    p.add_argument("--allow-implausible", action="store_true",
                   help="analyse 'probably disconnected' recordings anyway (QC warning instead of refusal)")
    p.add_argument("--flat-std-uv", type=float, default=QCThresholds.flat_std_uv)
    p.add_argument("--l-freq", type=float, default=8.0, help="decode band-pass low edge (Hz)")
    p.add_argument("--h-freq", type=float, default=30.0, help="decode band-pass high edge (Hz)")
    p.add_argument("--n-perm", type=int, default=100, help="label permutations per decoding cell (0 = none)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1, help=">1 = RepeatedStratifiedKFold (single-recording CV)")
    p.add_argument("--n-sham-draws", type=int, default=20)
    p.add_argument("--min-per-class", type=int, default=5, help="classes with fewer trials are dropped")
    p.add_argument("--models", nargs="+", help="decoders to run (default all): 'CSP+LDA' 'Riemann TS+LR'")
    p.add_argument("--jobs", type=int, default=-1, help="parallel jobs for CV / permutations")
    return p


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live even when piped to a file
    args = build_parser().parse_args(argv)
    if args.spec and len(args.bids_root) > 1:
        sys.exit("--spec applies to a single dataset; pass one --bids-root or put a spec in each <root>/code/")
    try:
        pipelines = parse_pipelines(args.pipeline)
    except ValueError as exc:
        sys.exit(str(exc))
    opts = RunOptions(
        results_root=os.path.abspath(args.out_dir), spec_path=args.spec, sub=args.sub, ses=args.ses,
        task=args.task, min_live_topomap=args.min_live_topomap, min_live_decode=args.min_live_channels,
        allow_implausible=args.allow_implausible, l_freq=args.l_freq, h_freq=args.h_freq, n_perm=args.n_perm,
        folds=args.folds, repeats=args.repeats, n_sham_draws=args.n_sham_draws, min_per_class=args.min_per_class,
        n_jobs=args.jobs, models=args.models, script_file=__file__, thresholds=QCThresholds(flat_std_uv=args.flat_std_uv),
    )
    errors = 0
    for root in args.bids_root:
        try:
            summary = run_dataset(root, pipelines, opts)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[error] {root}: {exc}", file=sys.stderr)
            errors += 1
            continue
        errors += sum("error" in r for r in summary["recordings"].values())
        print(f"[+] {summary['dataset']}: summary at "
              f"{os.path.join(opts.results_root, 'summary', summary['dataset'], 'summary.md')}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
