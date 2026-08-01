# -*- coding: utf-8 -*-
"""Reproduction runner: run every paper experiment and verify the numbers
against the golden reference (paper/RESULTS_MULTISEED.md).

Usage:
    python run_all.py             # full reproduction (needs torch + MNIST data)
    python run_all.py --skip-mnist  # digits-only experiments (fast, no torch)

Exit code 0 = all numbers within tolerance of the paper.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, got, expected, tol=0.01):
    ok = abs(got - expected) <= tol
    print(f'  [{"PASS" if ok else "FAIL"}] {name}: '
          f'got {got:.4f}, golden {expected:.4f} (tol {tol})')
    if not ok:
        FAILURES.append(name)


def ms(a):
    import numpy as np
    a = np.asarray(a, dtype=float)
    return float(a.mean())


def run_digits():
    """Sections A + B: build-rule and prototype-source ablations (5 seeds)."""
    from exp_multiseed import SEEDS_DIGITS, run_A, run_B
    print('=' * 72)
    print('Section A. Build-rule ablation: CRP vs threshold k-means (digits 30D, 5 seeds)')
    print('=' * 72)
    rA = run_A(SEEDS_DIGITS)
    check('A. DP-ProtOn ACC', ms(rA['dp_acc']), 0.9303, tol=0.015)
    check('A. Threshold k-means ACC', ms(rA['km_acc']), 0.2342, tol=0.015)
    d = ms([a - b for a, b in zip(rA['dp_acc'], rA['km_acc'])])
    check('A. Delta (DP - kmeans)', d, 0.6960, tol=0.02)

    print('=' * 72)
    print('Section B. Prototype-source ablation (digits 30D, 5 seeds)')
    print('=' * 72)
    rB = run_B(SEEDS_DIGITS)
    check('B. Online denoised sub-centroid k=1', ms(rB['sub']), 0.9303, tol=0.015)
    check('B. Raw reservoir memory', ms(rB['raw']), 0.9143, tol=0.015)
    check('B. Offline kmeans-30', ms(rB['cent']), 0.8963, tol=0.015)
    check('B. 1-NN upper bound', ms(rB['knn']), 0.9516, tol=0.015)
    d = ms([a - b for a, b in zip(rB['sub'], rB['cent'])])
    check('B. Delta (sub-centroid - kmeans30)', d, 0.0340, tol=0.015)


def run_editing():
    """Section D: knowledge editing (digits 30D, 5 seeds).

    exp_edit.py executes at import time (module-level script, unchanged);
    its results dict `R` is the exact source of the paper's numbers.
    """
    from exp_edit import R
    print('=' * 72)
    print('Section D. Knowledge editing: delete / relabel / inject (digits 30D, 5 seeds)')
    print('=' * 72)
    check('D. DELETE class-9 ACC after', ms(R['dt1']), 0.0000, tol=0.01)
    check('D. DELETE other-class max shift', ms(R['dsh']), 0.0467, tol=0.02)
    check('D. DELETE novelty before', ms(R['n0']), 9.52, tol=0.5)
    check('D. DELETE novelty after', ms(R['n1']), 11.98, tol=0.5)
    check('D. RELABEL remap-identical', ms(R['rl']), 1.0000, tol=0.01)
    check('D. RELABEL non-target identical', ms(R['ro']), 0.9941, tol=0.01)
    check('D. INJECT class-9 ACC after', ms(R['it1']), 0.8564, tol=0.02)
    check('D. INJECT old-class identical', ms(R['io']), 0.9883, tol=0.01)
    check('D. INJECT other-class max shift', ms(R['ish']), 0.0443, tol=0.02)


def run_mnist():
    """Section C: Split-MNIST vs baselines (3 seeds; needs torch + MNIST data)."""
    from exp_multiseed import SEEDS_MNIST, run_C
    print('=' * 72)
    print('Section C. Split-MNIST continual learning (50D, 5 tasks, 3 seeds)')
    print('=' * 72)
    rC = run_C(SEEDS_MNIST)
    golden = {
        'Joint': (0.9693, None), 'DP-ProtOn': (0.8747, -0.0471),
        'Replay': (0.6787, -0.3921), 'EWC': (0.1949, -0.9927),
        'Fine-tune': (0.1978, -0.9947),
    }
    for k, (g_acc, g_bwt) in golden.items():
        check(f'C. {k} ACC', ms(rC['acc'][k]), g_acc, tol=0.015)
        if g_bwt is not None:
            check(f'C. {k} BWT', ms(rC['bwt'][k]), g_bwt, tol=0.02)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-mnist', action='store_true',
                    help='skip Split-MNIST section (no torch / data)')
    args = ap.parse_args()

    print('#' * 72)
    print('#  DP-ProtOn paper reproduction')
    print('#  golden reference: paper/RESULTS_MULTISEED.md')
    print('#' * 72)

    run_digits()
    run_editing()
    if not args.skip_mnist:
        run_mnist()
    else:
        print('\n[SKIP] Section C (--skip-mnist)')

    print('\n' + '=' * 72)
    if FAILURES:
        print(f'RESULT: FAIL ({len(FAILURES)} check(s) outside tolerance)')
        for f in FAILURES:
            print(f'  - {f}')
        sys.exit(1)
    print('RESULT: PASS - all paper numbers reproduced')
    sys.exit(0)


if __name__ == '__main__':
    main()
