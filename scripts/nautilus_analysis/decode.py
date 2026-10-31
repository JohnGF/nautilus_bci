"""Strict decoding. All ``STRICT_DECODING_METHODOLOGY.md`` invariants live here.

Callers pass an ``epochs.EpochSet``, which is already baseline-corrected
(rule 2) and labelled from within-session events (rule 1). ``decode_epochset``
then adds the rest:

- rule 3: every transform (CSP, covariance/tangent space, scaling) sits inside
  an sklearn Pipeline, so it is fit inside each training fold;
- rule 4: ``choose_cv`` uses Leave-One-Group-Out when the epochs come from more
  than one recording (each with >= 2 classes), and otherwise
  (Repeated)StratifiedKFold. When epoch windows overlap (fast event streams
  such as FNF arrows), folds are time-contiguous (unshuffled) and
  ``PurgedCV`` drops every training epoch that overlaps a test epoch.
  Otherwise shuffled k-fold leaks shared samples between train and test;
  FNF scored 94% that way;
- rule 5: sham Rest epochs with balanced random pseudo-labels must score at
  chance (within ``CHANCE_MARGIN``), and a carry-over control (each Rest
  labelled with the preceding trial's class) is reported;
- rule 6: label-permutation null per cell, with an empirical p-value;
- rule 7: wear time, active task seconds, fit wall-clock;
- invariant 4 of the plan: every (channel set x pipeline) cell is counted,
  and p-values are corrected with the max-statistic over cells (the
  permutations are identical across cells) plus Benjamini-Hochberg FDR.

Scoring is balanced accuracy, so chance is 1/n_classes even when classes are
unbalanced. BLAS is limited to one thread while decoding. Parallelism comes
from joblib workers, and multi-threaded BLAS on these small covariance
matrices made each CSP fit about 4x slower.
"""

import re
import time
import warnings

import numpy as np
from mne.decoding import CSP
from mne.stats import fdr_correction
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_score)
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

SCORING = "balanced_accuracy"
CHANCE_MARGIN = 0.05  # sham must score within chance + this (methodology rule 5)
FRONTAL_RE = re.compile(r"^(Fp|AF|F)(\d+|z)$", re.IGNORECASE)  # FC/FT are fronto-central and kept
EYE_PROXY = ("Fp1", "Fp2")


def make_pipelines(n_channels):
    """The default decoders: {name: unfitted Pipeline}."""
    return {
        "CSP+LDA": make_pipeline(
            CSP(n_components=min(6, n_channels), reg="ledoit_wolf", log=True, rank="full"),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
        "Riemann TS+LR": make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann"),
            LogisticRegression(C=1.0, max_iter=2000),
        ),
    }


def model_factories(names=None):
    """{name: f(n_channels) -> Pipeline} for ``names`` (default: all of ``make_pipelines``)."""
    available = list(make_pipelines(2))
    names = names or available
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; available: {available}")
    return {n: (lambda k, n=n: make_pipelines(k)[n]) for n in names}


def channel_sets(ch_names):
    """Montage-aware channel subsets: all, no frontal (EOG check), Fp1/Fp2 only (eye proxy)."""
    sets = {"all": list(ch_names)}
    no_frontal = [c for c in ch_names if not FRONTAL_RE.match(c)]
    if 2 <= len(no_frontal) < len(ch_names):
        sets["no_frontal"] = no_frontal
    eye = [c for c in EYE_PROXY if c in ch_names]
    if len(eye) == 2:
        sets["fp1_fp2_only"] = eye
    return sets


class PurgedCV:
    """Wrap a CV splitter; remove training epochs whose window overlaps any test epoch's window."""

    def __init__(self, cv, onsets, groups, window_s):
        self.cv, self.onsets, self.groups, self.window_s = cv, np.asarray(onsets), np.asarray(groups), window_s

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cv.get_n_splits(X, y, groups)

    def split(self, X, y=None, groups=None):
        for train, test in self.cv.split(X, y, groups):
            keep = np.ones(len(train), bool)
            for g in np.unique(self.groups[test]):
                t_on = np.sort(self.onsets[test][self.groups[test] == g])
                in_g = self.groups[train] == g
                tr_on = self.onsets[train][in_g]
                # nearest test onset on either side of each training onset
                i = np.searchsorted(t_on, tr_on)
                gap = np.minimum(np.abs(tr_on - t_on[np.clip(i - 1, 0, len(t_on) - 1)]),
                                 np.abs(tr_on - t_on[np.clip(i, 0, len(t_on) - 1)]))
                keep[np.flatnonzero(in_g)[gap < self.window_s - 1e-9]] = False
            yield train[keep], test

    def purged_fraction(self, X, y, groups=None):
        """Mean share of each fold's training epochs removed by purging."""
        sizes = [(len(tr_p), len(tr)) for (tr_p, _), (tr, _) in zip(self.split(X, y, groups), self.cv.split(X, y, groups))]
        return float(np.mean([1 - a / b for a, b in sizes])) if sizes else 0.0


def choose_cv(y, groups=None, n_splits=5, n_repeats=1, seed=42, contiguous=False):
    """Return (cv, groups_for_cv, description). Raises ValueError if y can't be cross-validated.

    ``contiguous`` (for overlapping epochs) uses unshuffled stratified folds, which are blocks in time.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"need >= 2 classes, got {dict(zip(classes, counts))}")
    if groups is not None:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        if len(ug) > 1 and all(len(np.unique(y[groups == g])) >= 2 for g in ug):
            return LeaveOneGroupOut(), groups, f"Leave-One-Group-Out over {len(ug)} recordings"
    k = int(min(n_splits, counts.min()))
    if k < 2:
        raise ValueError(f"smallest class has {counts.min()} trial(s); need >= 2 per class")
    if contiguous:
        return StratifiedKFold(n_splits=k, shuffle=False), None, f"StratifiedKFold {k} (contiguous, unshuffled)"
    if n_repeats > 1:
        return (RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed), None,
                f"RepeatedStratifiedKFold {k}x{n_repeats}")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), None, f"StratifiedKFold {k} (shuffled)"


def _permuted_labels(y, groups, units, rng):
    """One null labelling: shuffle whole label runs (``units``), within ``groups``, or freely."""
    y = np.asarray(y, dtype=object)
    if units is not None:
        uniq, first = np.unique(units, return_index=True)
        new = dict(zip(uniq, rng.permutation(y[first])))
        return np.array([new[u] for u in units], dtype=object)
    if groups is None:
        return rng.permutation(y)
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[rng.permutation(idx)]
    return out


def _cv_mean(pipe, X, y, cv, groups):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(cross_val_score(clone(pipe), X, y, groups=groups, cv=cv, scoring=SCORING)))


def evaluate(pipe, X, y, cv, groups=None, n_perm=100, seed=0, n_jobs=-1, perm_groups=None, perm_units=None):
    """Cross-validated score + label-permutation null. ``null`` is kept for max-stat correction.

    ``groups`` go to the CV splitter. Permutations shuffle labels within
    ``perm_groups`` (default ``groups``; e.g. within recordings for LOGO), or,
    with ``perm_units``, shuffle the labels of whole units (label blocks), so a
    blocked design is compared with a null that keeps its block structure. A
    fixed ``seed`` gives the same permutations for every cell, which the
    max-statistic correction requires.
    """
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    null = np.array([])
    if n_perm:
        rng = np.random.default_rng(seed)
        pg = groups if perm_groups is None else perm_groups
        perms = [_permuted_labels(y, pg, perm_units, rng) for _ in range(n_perm)]
        null = np.array(Parallel(n_jobs=n_jobs)(delayed(_cv_mean)(pipe, X, yp, cv, groups) for yp in perms))
    acc = float(np.mean(folds))
    out = dict(acc=acc, fold_sd=float(np.std(folds)), n_folds=int(len(folds)), fit_wall_s=round(time.time() - t0, 2))
    if len(null):
        out.update(p_value=float((np.sum(null >= acc) + 1) / (len(null) + 1)), null_mean=float(np.mean(null)),
                   null_sd=float(np.std(null)), null_ci95=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                   null_p95=float(np.percentile(null, 95)), null=null)
    return out


def label_blocks(es):
    """Label runs in time within each recording -> (run id per epoch, repeat fraction, chance repeat fraction)."""
    runs = np.empty(len(es.y), dtype=object)
    same, n_pairs = 0, 0
    for g in np.unique(es.recordings):
        idx = np.flatnonzero(es.recordings == g)
        order = idx[np.argsort(es.onsets[idx], kind="stable")]
        k = 0
        for j, i in enumerate(order):
            if j and es.y[i] != es.y[order[j - 1]]:
                k += 1
            runs[i] = f"{g}#{k}"
        same += int(np.sum(es.y[order][1:] == es.y[order][:-1]))
        n_pairs += max(len(order) - 1, 0)
    p = np.array(list(es.class_counts.values()), float)
    p /= p.sum()
    return runs, (same / n_pairs if n_pairs else 0.0), float(np.sum(p ** 2))


def sham_random(pipe, X_sham, labels, n_trials, n_splits=5, n_draws=20, seed=1, n_jobs=-1):
    """Sham Rest epochs with balanced random pseudo-labels; the mean must sit at chance."""
    rng = np.random.default_rng(seed)
    n = int(min(n_trials, len(X_sham)))
    k = int(min(n_splits, n // len(labels)))
    if k < 2:
        return dict(status="unavailable", reason=f"only {len(X_sham)} sham epoch(s) for {len(labels)} classes")
    accs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_draws):
            idx = rng.choice(len(X_sham), size=n, replace=False)
            y = np.resize(np.asarray(labels, dtype=object), n)
            rng.shuffle(y)
            cv = StratifiedKFold(k, shuffle=True, random_state=int(rng.integers(1 << 31)))
            accs.append(np.mean(cross_val_score(pipe, X_sham[idx], y, cv=cv, scoring=SCORING, n_jobs=n_jobs)))
    chance = 1.0 / len(labels)
    acc = float(np.mean(accs))
    return dict(status="pass" if acc <= chance + CHANCE_MARGIN else "FAIL", acc=acc, sd=float(np.std(accs)),
                n_epochs=n, n_draws=n_draws, chance=chance, margin=CHANCE_MARGIN)


def sham_carry_over(pipe, X_sham, y_prev, groups=None, n_splits=5, n_perm=100, seed=2, n_jobs=-1):
    """Rest epochs labelled with the preceding trial's class. Significant = residual activity or drift."""
    keep = np.array([isinstance(p, str) for p in y_prev], bool)
    try:
        cv, g, desc = choose_cv(y_prev[keep], None if groups is None else groups[keep], n_splits)
    except ValueError as exc:
        return dict(status="unavailable", reason=str(exc))
    res = evaluate(pipe, X_sham[keep], y_prev[keep], cv, g, n_perm, seed, n_jobs)
    res.pop("null", None)
    res.update(cv=desc, n_epochs=int(keep.sum()),
               status="flag" if res.get("p_value", 1.0) < 0.05 else "pass")
    return res


def decode_epochset(es, *args, **kwargs):
    """Run the strict decoding battery on an ``EpochSet``. Returns a JSON-serialisable dict.

    Arguments: see ``_decode_epochset``. Raises ValueError when ``es`` can't be cross-validated.
    """
    with threadpool_limits(1):
        return _decode_epochset(es, *args, **kwargs)


def _decode_epochset(es, n_perm=100, n_splits=5, n_repeats=1, n_sham_draws=20, seed=0, n_jobs=-1,
                     pipelines=None, sets=None, min_per_class=5, log=None):
    """``log`` (e.g. ``print``) gets a progress line before and after every cell and sham control.

    ``pipelines`` maps names to factories ``f(n_channels) -> Pipeline`` (default:
    ``make_pipelines``). ``sets`` maps channel-set names to channel lists
    (default: ``channel_sets(es.ch_names)``).
    """
    counts = es.class_counts
    small = {c: n for c, n in counts.items() if n < min_per_class}
    if small:
        es = es.pick_labels([c for c in counts if c not in small])
    labels = es.labels
    out = dict(classes=labels, class_counts=es.class_counts, dropped_small_classes=small,
               chance=1.0 / len(labels) if labels else None, scoring=SCORING, n_perm=n_perm,
               quantification=dict(wear_time_s=round(es.wear_time_s, 1), n_trials=int(len(es.y)),
                                   active_task_s=round(es.active_task_s, 1), window_s=round(es.window_s, 3),
                                   n_sham_epochs=int(len(es.X_sham)), sfreq=es.sfreq,
                                   n_channels=len(es.ch_names), band_hz=list(es.band)))
    overlap = es.overlap_fraction
    out["quantification"]["overlap_fraction"] = round(overlap, 3)
    cv, groups, desc = choose_cv(es.y, es.groups, n_splits, n_repeats, contiguous=overlap > 0)  # raises if not decodable
    perm_groups = perm_units = None
    runs, repeat, repeat_chance = label_blocks(es)
    q = out["quantification"]
    q.update(label_repeat_fraction=round(repeat, 3), label_repeat_chance=round(repeat_chance, 3),
             n_label_blocks=int(len(set(runs))))
    blocked = repeat > 0.5 and repeat > repeat_chance + 0.25
    out["blocked_design"] = None
    if blocked and not desc.startswith("Leave-One-Group-Out"):
        blocks_per_class = {c: len({r for r, lab in zip(runs, es.y) if lab == c}) for c in labels}
        k = min(n_splits, min(blocks_per_class.values()))
        if k >= 2:
            cv, groups = StratifiedGroupKFold(n_splits=k, shuffle=False), runs
            perm_units = runs
            desc = f"StratifiedGroupKFold {k} over label blocks (whole blocks held out)"
            out["blocked_design"] = dict(status="block-wise CV", blocks_per_class=blocks_per_class)
        else:
            out["blocked_design"] = dict(status="CONFOUNDED", blocks_per_class=blocks_per_class)
        desc += f"; labels blocked in time ({repeat:.0%} of trials repeat the previous label, chance {repeat_chance:.0%})"
    elif desc.startswith("Leave-One-Group-Out"):
        perm_groups = es.groups
    if overlap > 0:
        cv = PurgedCV(cv, es.onsets, es.recordings, es.window_s)
        purged = cv.purged_fraction(es.X[:, :1, :1], es.y, groups)
        out["quantification"]["purged_train_fraction"] = round(purged, 3)
        desc += f", purged ({overlap:.0%} of epochs overlap; {purged:.1%} of training epochs removed per fold)"
    out["cv"] = desc
    factories = pipelines or model_factories()
    sets = sets or channel_sets(es.ch_names)
    t0 = time.time()
    log = log or (lambda *a, **k: None)
    n_steps = len(sets) * len(factories) + 2 * len(factories)
    step = 0

    def progress(what):
        nonlocal step
        step += 1
        log(f"      [{step}/{n_steps}] {what} ... (decode elapsed {time.time() - t0:.0f} s)")

    log(f"      decoding {len(es.y)} trials x {len(es.ch_names)} ch, {desc}, {n_perm} permutations per cell")
    X = es.X.astype(np.float64)
    cells = {}
    for set_name, chans in sets.items():
        idx = [es.ch_names.index(c) for c in chans]
        cells[set_name] = {}
        for pname, factory in factories.items():
            progress(f"{pname} on '{set_name}' ({len(idx)} ch)")
            r = evaluate(factory(len(idx)), X[:, idx], es.y, cv, groups, n_perm, seed, n_jobs, perm_groups, perm_units)
            cells[set_name][pname] = r
            log(f"            -> {r['acc']:.1%} (p={r.get('p_value', float('nan')):.3f}, {r['fit_wall_s']:.0f} s)")
    out["cells"] = cells
    out["multiple_comparisons"] = _correct(cells, n_perm)

    Xs = es.X_sham.astype(np.float64)
    out["sham"] = {}
    for pname, factory in factories.items():
        n_ch = len(es.ch_names)
        if not len(Xs):
            out["sham"][pname] = dict(random=dict(status="unavailable", reason="no sham Rest epochs (spec.sham)"),
                                      carry_over=dict(status="unavailable", reason="no sham Rest epochs"))
            continue
        progress(f"sham random-label control, {pname} ({n_sham_draws} draws)")
        rnd = sham_random(factory(n_ch), Xs, labels, len(es.y), n_splits, n_sham_draws, seed + 1, n_jobs)
        log(f"            -> {rnd.get('acc', float('nan')):.1%} {rnd['status']}")
        progress(f"sham carry-over control, {pname}")
        carry = sham_carry_over(factory(n_ch), Xs, es.y_sham_prev, es.groups_sham, n_splits, n_perm,
                                seed + 2, n_jobs)
        log(f"            -> {carry.get('acc', float('nan')):.1%} {carry['status']}")
        out["sham"][pname] = dict(random=rnd, carry_over=carry)
    out["quantification"]["fit_wall_s"] = round(time.time() - t0, 1)
    out["verdict"] = _verdict(out)
    for d in cells.values():
        for r in d.values():
            r.pop("null", None)
    return out


def _correct(cells, n_perm):
    flat = [(s, p, r) for s, d in cells.items() for p, r in d.items()]
    out = dict(n_cells=len(flat), method="max-statistic over cells (shared permutations) + Benjamini-Hochberg FDR")
    if not n_perm or not flat:
        out["method"] = "none (no permutations)"
        return out
    max_null = np.max(np.vstack([r["null"] for _, _, r in flat]), axis=0)
    p_raw = np.array([r["p_value"] for _, _, r in flat])
    _, p_fdr = fdr_correction(p_raw)
    for (s, p, r), pf in zip(flat, p_fdr):
        r["p_maxstat"] = float((np.sum(max_null >= r["acc"]) + 1) / (len(max_null) + 1))
        r["p_fdr"] = float(pf)
    out["n_significant_maxstat"] = int(sum(r["p_maxstat"] < 0.05 for _, _, r in flat))
    return out


def _verdict(out):
    """One-line interpretation: the best cell, whether it survives correction, whether shams pass."""
    best_set, best_pipe, best = max(((s, p, r) for s, d in out["cells"].items() for p, r in d.items()),
                                    key=lambda t: t[2]["acc"])
    sham_status = [v["random"].get("status") for v in out["sham"].values()]
    lines = [f"best {best_pipe} on '{best_set}': {best['acc']:.1%} (chance {out['chance']:.1%})"]
    bd = out.get("blocked_design")
    if bd and bd["status"] == "CONFOUNDED":
        lines.insert(0, f"CONFOUNDED: some classes occur in a single time block ({bd['blocks_per_class']}), so class "
                        "is inseparable from time-on-task/drift; this accuracy is NOT evidence of decoding")
    elif bd:
        lines.append(f"labels come in time blocks {bd['blocks_per_class']}: whole blocks held out")
    ov = out["quantification"].get("overlap_fraction", 0)
    if ov > 0:
        lines.append(f"{ov:.0%} of epoch windows overlap neighbouring trials: folds purged; each epoch still "
                     "contains responses to neighbouring events, so read accuracy as event-stream decoding")
    if "p_maxstat" in best:
        lines.append(f"corrected p={best['p_maxstat']:.3f} over {out['multiple_comparisons']['n_cells']} cells"
                     + (" (significant)" if best["p_maxstat"] < 0.05 else " (not significant)"))
    if "FAIL" in sham_status:
        lines.append("SHAM CONTROL FAILED: the pipeline decodes un-cued Rest; reject these results")
    elif all(s == "unavailable" for s in sham_status):
        lines.append("sham control unavailable (no Rest epochs): rule 5 not satisfied")
    eye = out["cells"].get("fp1_fp2_only", {})
    if eye and max(r["acc"] for r in eye.values()) >= best["acc"] - 0.02:
        lines.append("Fp1/Fp2 alone decode about as well as the best set: likely eye-movement driven")
    return "; ".join(lines)
