# -*- coding: utf-8 -*-
"""Smoke tests: verify the clean dp_proton package against paper numbers.

Paper (RESULTS_MULTISEED.md, 5 seeds):
  - digits 30-D PCA-whitened: ACC = 0.9303 +- 0.0090 (DP-ProtOn, n_sub=1)
  - DELETE class 9: class-9 acc -> 0, novelty score rises
  - RELABEL 7->99: predictions on other classes unchanged
  - INJECT class 9 from 20 examples: class 9 recovered, others unchanged

Run:  python -m pytest tests/ -v   (or: python tests/smoke_test.py)
"""
import os
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dp_proton import DPProtOn

PAPER_ACC = 0.9303
PAPER_STD = 0.0090
SEEDS = [0, 1, 2, 3, 4]


def build_split_digits(n_comp=30, seed=0):
    """Same protocol as experiments/exp_highdim.py: 30% pretrain pool for
    PCA, then 5 tasks x 2 classes, 70/30 split, shuffled per task."""
    d = load_digits()
    X, y = d.data, d.target
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_pre = int(0.3 * len(X))
    scaler = StandardScaler().fit(X[:n_pre])
    pca = PCA(n_components=n_comp, whiten=True, random_state=seed).fit(
        scaler.transform(X[:n_pre]))
    Xr = pca.transform(scaler.transform(X[n_pre:]))
    yr = y[n_pre:]
    tasks = []
    for k in range(5):
        cls = (2 * k, 2 * k + 1)
        m = np.isin(yr, cls)
        Xk, yk = Xr[m], yr[m]
        r = np.random.default_rng(100 + k)
        p = r.permutation(len(Xk))
        Xk, yk = Xk[p], yk[p]
        s = int(0.7 * len(Xk))
        tasks.append((Xk[:s], yk[:s], Xk[s:], yk[s:]))
    return tasks


def run_continual(tasks, **kw):
    """Learn task-by-task, evaluate on all classes at the end (paper §5)."""
    model = DPProtOn(**kw)
    for Xtr, ytr, _, _ in tasks:
        model.fit_stream(Xtr, ytr)
    Xte = np.vstack([t[2] for t in tasks])
    yte = np.concatenate([t[3] for t in tasks])
    preds = model.classify(Xte)
    return model, float((preds == yte).mean())


def test_accuracy_matches_paper_level():
    """5-seed digits accuracy within 2 sigma of the paper's 0.9303+-0.0090."""
    accs = [run_continual(build_split_digits(seed=s))[1] for s in SEEDS]
    mean, std = float(np.mean(accs)), float(np.std(accs))
    print(f'\nDP-ProtOn digits 30-D, 5 seeds: {mean:.4f} +- {std:.4f} '
          f'(paper: {PAPER_ACC} +- {PAPER_STD})')
    assert mean > PAPER_ACC - 2 * PAPER_STD - 0.02, f'mean ACC {mean:.4f} too low'
    assert mean > 0.92, f'mean ACC {mean:.4f} below paper level (0.93)'


def test_delete_class():
    """DELETE: class gone (acc 0), other classes untouched, novelty rises."""
    tasks = build_split_digits(seed=0)
    X = np.vstack([t[2] for t in tasks])
    y = np.concatenate([t[3] for t in tasks])
    model, _ = run_continual(tasks)
    nov_before = float(model.novelty(X[y == 9]).mean())
    n_deleted = model.delete_class(9)
    assert n_deleted > 0
    preds = model.classify(X)
    assert not np.any(preds[y == 9] == 9), 'deleted class still predicted'
    others = y != 9
    acc_others = float((preds[others] == y[others]).mean())
    assert acc_others > 0.92, f'other-class ACC dropped to {acc_others:.4f}'
    nov_after = float(model.novelty(X[y == 9]).mean())
    print(f'DELETE: novelty {nov_before:.2f} -> {nov_after:.2f} '
          f'(should rise); others ACC {acc_others:.4f}')
    assert nov_after > nov_before, 'novelty of deleted class did not rise'


def test_relabel_class():
    """RELABEL 7->99: predictions elsewhere identical."""
    tasks = build_split_digits(seed=0)
    X = np.vstack([t[2] for t in tasks])
    y = np.concatenate([t[3] for t in tasks])
    model, _ = run_continual(tasks)
    p0 = model.classify(X)
    n = model.relabel_class(7, 99)
    assert n > 0
    p1 = model.classify(X)
    # paper's criterion (exp_edit.py): remap 99->7, predictions must be identical
    remapped = np.where(p1 == 99, 7, p1)
    assert float((remapped == p0).mean()) == 1.0, 'other classes changed'


def test_inject_class():
    """INJECT: class 9 resurrected from 20 examples, others unchanged."""
    tasks = build_split_digits(seed=0)
    X = np.vstack([t[2] for t in tasks])
    y = np.concatenate([t[3] for t in tasks])
    Xtr = np.vstack([t[0] for t in tasks])
    ytr = np.concatenate([t[1] for t in tasks])
    model, _ = run_continual(tasks)
    model.delete_class(9)
    p0 = model.classify(X)
    Xk = Xtr[ytr == 9][:20]
    model.inject_class(Xk, np.full(20, 9))
    p1 = model.classify(X)
    acc9 = float((p1[y == 9] == 9).mean())
    mo = y != 9
    unchanged = float((p1[mo] == p0[mo]).mean())
    print(f'INJECT: class-9 ACC {acc9:.4f} after 20 examples; '
          f'others unchanged {unchanged:.4f}')
    assert acc9 > 0.5, f'class 9 not recovered (ACC {acc9:.4f})'
    assert unchanged > 0.95, f'other classes disturbed ({unchanged:.4f})'


def test_single_pass_no_revisit():
    """Basins only ever updated on their own label's data (local updates)."""
    tasks = build_split_digits(seed=0)
    model = DPProtOn(alpha=1.0, n_sub=1)
    for Xtr, ytr, _, _ in tasks:
        model.fit_stream(Xtr, ytr)
    assert all(b.n >= 1 for b in model.basins)
    # stream processed exactly once: basins never exceed training size
    total = sum(t[0].shape[0] for t in tasks)
    assert sum(b.n for b in model.basins) == total


if __name__ == '__main__':
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'[PASS] {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {e}')
    sys.exit(1 if failed else 0)
