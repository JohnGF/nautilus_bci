"""The fixture's planted mu-ERD can be found with the library's building blocks.

This checks the fixture itself (Phase 2's decode.py will be tested against it),
and that io -> qc -> events give correct epochs: a fold-internal log-variance
+ LDA decoder finds the effect, and the same decoder on sham Rest epochs with
random labels stays at chance.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from nautilus_analysis import apply_qc, build_trial_table, compute_qc, discover, gate, infer_spec, load_raw

PICKS = ["C3", "Cz", "C4", "Pz"]


def _epochs(raw, onsets, base_onsets, tmin, tmax, btmin=-0.5, btmax=0.0):
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=PICKS)
    out = []
    for o, b in zip(onsets, base_onsets):
        base = data[:, int((b + btmin) * sf):int((b + btmax) * sf)].mean(axis=1, keepdims=True)
        out.append(data[:, int((o + tmin) * sf):int((o + tmax) * sf)] - base)
    return np.array(out)


def _decoder():
    logvar = FunctionTransformer(lambda X: np.log(np.var(X, axis=-1)))
    return make_pipeline(logvar, StandardScaler(), LinearDiscriminantAnalysis())


def test_planted_mu_erd_is_decodable_and_sham_is_not(synthetic_bids):
    recs = discover(synthetic_bids, sub="01")
    loaded = [load_raw(r) for r in recs]
    spec = infer_spec([L.events for L in loaded])
    X, y, Xs = [], [], []
    for L in loaded:
        gate(compute_qc(L.raw, L.recording.id), min_live_channels=16)
        raw = apply_qc(L.raw, compute_qc(L.raw)).filter(8.0, 13.0, verbose="error")
        table = build_trial_table(L.events, spec)
        w = spec.conditions.window
        X.append(_epochs(raw, table.trials["anchor_onset"], table.trials["baseline_onset"], w.tmin, w.tmax))
        y += list(table.trials["label"])
        Xs.append(_epochs(raw, table.sham["onset"], table.sham["onset"], spec.sham.tmin, spec.sham.tmax,
                          btmin=0.0, btmax=0.5))
    X, y, Xs = np.concatenate(X), np.array(y), np.concatenate(Xs)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    acc = cross_val_score(_decoder(), X, y, cv=cv).mean()
    assert acc > 0.8, acc  # chance 0.25

    rng = np.random.default_rng(0)
    sham_acc = np.mean([cross_val_score(_decoder(), Xs, rng.permutation(y[:len(Xs)]), cv=cv).mean()
                        for _ in range(5)])
    assert sham_acc < 0.4, sham_acc
