# DP-ProtOn

**Dirichlet-Process-guided Prototype Online learner** —
https://github.com/Fengrru/dp-proton

A minimal, gradient-free, single-pass prototype classifier for continual
learning under frozen features — with explicit, zero-retraining knowledge
editing (delete / relabel / inject a class).

> ## ⚠️ Correction (2026-10-11): the §5.2 creation-rule comparison does not hold
>
> The headline ablation of this paper — that the CRP criterion beats a
> distance-threshold online *k*-means by **ΔACC = 0.6960 ± 0.0528** at matched
> prototype budget, and that therefore "the DP criterion is necessary, not
> excess complexity" — **is an artifact of a confounded comparison.**
>
> The two models did not differ only in their creation rule:
>
> - The threshold *k*-means baseline allocated basins **label-blind** — it
>   searched all centres regardless of class — while DP/CRP's rejection baseline
>   carried a **per-class** base measure. Per-class allocation is itself
>   load-bearing: on the same features, label-blind basins reach 0.17–0.68
>   accuracy where per-class basins reach 0.92–0.99.
> - The baseline's distance threshold was not calibrated to a structural target.
>
> Re-measured on the same dataset (sklearn *digits*, 20 basins, matched budget),
> with each criterion's single scalar hyperparameter fixed by a **structural
> target** rather than tuned for accuracy, and one no-knob readout for all four:
>
> | Criterion | ACC @ 20 basins |
> |---|---|
> | hard threshold | 0.916 |
> | threshold online *k*-means | **0.935** |
> | DP / CRP (estimated base) | 0.917 |
> | DP / CRP (fixed prior base) | 0.916 |
>
> The DP figure reproduces (0.9303 → 0.917); the **baseline** moved from 0.2342
> to 0.935. Across every real dataset tested the spread between all four criteria
> is ≤ 1.9 accuracy points, with no consistent winner.
>
> **What still stands:** the per-class architecture, the denoised in-basin
> centroid, the rank/density readout, and exact local deletion. **What does not:**
> the necessity claim for DP/CRP basin creation, the ΔACC = 0.6960 figure in the
> abstract, and the "strongest single ablation result" framing in §5.2.
>
> DP/CRP does retain two properties, both narrower than claimed: its single
> hyperparameter has a more tractable shape at d ≤ 64, and it has a basin-count
> floor above d ≈ 128 that no α can lower.
>
> The re-measurement — seven paired experiments on synthetic and real data, with
> the confounded-vs-fair comparison and the mechanism of the failure — is in the
> companion repository `Fengrru/energy-basin-cl`.

> **What this is.** A fully-ablated, empirically grounded baseline for the
> *classifier head* of a continual learning system. Every component earns
> its place through a head-to-head ablation reported in the paper
> (see [paper/PAPER.md](paper/PAPER.md) and
> [paper/RESULTS_MULTISEED.md](paper/RESULTS_MULTISEED.md)).
>
> **What this is not.** Not a state-of-the-art CL system. Not an
> end-to-end learner (the feature extractor is frozen). Not a
> rehearsal-based method (single pass, no revisit). If you need SOTA on
> Split-CIFAR, this repo is not for you — if you need a *clean, honest,
> verifiable control* for your own CL experiments, this repo is for you.

---

## TL;DR — Why it exists

Continual learning (CL) research is dominated by increasingly complex
methods (distillation, generative replay, dynamic architectures) compared
against increasingly complex SOTA. What the community lacks is a
**minimal, understandable, gradient-free lower-bound baseline**: one where
every component can be turned on and off, and where the numbers in the
paper are reproducible with one command. DP-ProtOn is that baseline.

Three components, nothing else:

| # | Component | Role | Paper ref |
|:-:|-----------|------|-----------|
| 1 | **CRP basin creation** | Dirichlet-process posterior-predictive scores decide when to open a new prototype (`s_i = log n_i + log N(x|c_i,Σ_i)` vs rejection baseline `log α + log p_0(x)`) | §3.2 |
| 2 | **Online denoised centroids** | Competitive running-mean sub-centroids per basin (single pass, local update, no gradient) | §3.3 |
| 3 | **Parzen density readout** | Soft-kNN density over per-basin prototypes; label = argmax over aggregated densities | §3.4 |

Everything else that was tried — energy landscapes, precision gating,
barrier terms, reversible decay — was **empirically falsified** and
removed. The research history is documented in
[paper/TECH_REPORT.md](paper/TECH_REPORT.md) (a rare record of what
failed and why).

---

## Quick start

```bash
pip install -r requirements-core.txt
python -c "
from dp_proton import DPProtOn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d = load_digits()
X = PCA(n_components=30, whiten=True).fit_transform(
    StandardScaler().fit_transform(d.data))
y = d.target
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
n = int(0.7 * len(X))

model = DPProtOn(alpha=1.0, n_sub=1)      # 5 lines, no training loop
model.fit_stream(X[:n], y[:n])
acc = (model.classify(X[n:]) == y[n:]).mean()
print(f'test ACC = {acc:.4f}  (paper: 0.9303 +- 0.0090 on 5 seeds)')

model.delete_class(9)                      # zero-retraining knowledge editing
model.relabel_class(7, 99)
model.inject_class(X[n:][:20], np.full(20, 9))
"
```

Interactive demo (no install, runs in browser):
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) — open in Colab.

---

## Reproducing the paper

All numbers and figures in the paper are generated by the scripts in
[`experiments/`](experiments/), which are the **authoritative experiment
code** (unchanged from the research that produced the paper). One command:

```bash
cd experiments
python run_all.py            # full reproduction incl. Split-MNIST (needs torch)
python run_all.py --skip-mnist   # digits-only experiments (fast)
```

`run_all.py` runs every experiment and **checks the resulting numbers
against `paper/RESULTS_MULTISEED.md`** (golden reference), printing
`PASS`/`FAIL` per section.

| Script | Experiment | Paper section |
|--------|-----------|---------------|
| `exp_highdim.py` | High-dim verification (64-D vs PCA-whitened 30-D) | §5.1 |
| `exp_control_kmeans.py` | Build-rule ablation: CRP vs threshold k-means | §5.2 |
| `exp_multiseed.py` | Multi-seed robustness (5 seeds) | §5.3 |
| `exp_mnist_bench.py` | Split-MNIST vs Joint/Replay/EWC/Fine-tune | §5.4 |
| `exp_edit.py` | Knowledge editing: delete/relabel/inject | §5.5 |
| `exp_synth.py` | Synthetic multi-modal stress test | §5.1 |
| `gen_paper_figs.py` | Publication figures (fig_multiseed.png, fig_edit.png) | §5 |

### Two codebases, one truth

| Path | What it is | Status |
|------|------------|--------|
| [`dp_proton/`](dp_proton/core.py) | Clean re-implementation of the paper's Algorithm 1 (exactly 3 components + editing API) | Active, `pip install`-able |
| [`experiments/pw_afm.py`](experiments/pw_afm.py) | Original research implementation incl. *all* ablated variants (energy, gating, barriers...) — the exact code that produced the paper's numbers | Frozen, provenance only |

Why both? The paper's numbers must be traceable to the exact code that
produced them (frozen `experiments/`), while users need a clean API
(`dp_proton/`) that matches the paper's *description* of the algorithm.
The smoke test guarantees the two agree to paper-level accuracy:
`pytest tests/` asserts digits accuracy > 0.92 (paper: 0.9303 ± 0.0090).

### Environment

- Core: Python ≥ 3.9, NumPy, scikit-learn. **No deep-learning framework
  required for the model itself.**
- Experiments: `requirements-experiments.txt` adds matplotlib + PyTorch
  (only the Split-MNIST baselines EWC/Replay/Fine-tune/Joint need torch).
- MNIST: place the torchvision MNIST dataset under `experiments/data/`
  (or set `download=True` in `exp_mnist_bench.py:build_mnist_tasks`).

---

## Knowledge editing — the third contribution

Because knowledge lives in *explicit, locatable basins* (not distributed
weights), the model supports three surgeries with **zero retraining**:

```python
model.delete_class(9)                       # forget a class entirely
model.relabel_class(7, 99)                  # change semantics in place
model.inject_class(X_new, y_new)            # add a class from 20 examples
```

A softmax MLP cannot do any of these without retraining the whole network.
See paper §5.5 and the demo notebook.

---

## Results snapshot (5-seed, mean ± std)

| Benchmark | Method | ACC | BWT |
|-----------|--------|-----|-----|
| digits 30-D | DP-ProtOn | **0.9303 ± 0.0090** | — |
| digits 30-D | Threshold k-means (matched prototypes) | 0.2342 ± 0.0473 | — |
| Split-MNIST | Joint (oracle) | 0.9693 ± 0.0017 | — |
| Split-MNIST | **DP-ProtOn** | **0.8747 ± 0.0019** | **-0.0471 ± 0.0049** |
| Split-MNIST | Replay | 0.6787 ± 0.0174 | -0.3921 ± 0.0226 |
| Split-MNIST | EWC | 0.1949 ± 0.0004 | -0.9927 ± 0.0005 |
| Split-MNIST | Fine-tune | 0.1978 ± 0.0002 | -0.9947 ± 0.0003 |

Full tables and editing results: [`paper/RESULTS_MULTISEED.md`](paper/RESULTS_MULTISEED.md).

---

## Honest limitations (read before citing)

1. **Frozen features.** This is a classifier-head baseline. It does not
   learn features; performance is bounded by the quality of the frozen
   encoder. Integrating a learnable encoder is open future work (§7.4).
2. **No rehearsal.** Single pass; old data is never revisited. This is a
   feature of the design (rehearsal-free CL) and a limitation on raw
   accuracy vs methods that replay.
3. **α is not data-adaptive.** The CRP concentration α is fixed by the
   user; adapting it online is open (Escobar & West, 1995).
4. **Evidence scope.** Validated on digits (30-D) and Split-MNIST (50-D),
   plus synthetic multi-modal data. Not validated on ImageNet-scale
   features.
5. **No theoretical guarantees.** Empirical evidence only; the CRP rule
   is principled (exact DP posterior predictive) but there is no
   convergence/regret analysis.

See the paper's Limitations section for the full discussion.

---

## Repository layout

```
dp-proton/
├── dp_proton/            # clean package: Basin + DPProtOn (3 components + editing)
│   └── core.py
├── experiments/          # authoritative experiment code (frozen for paper numbers)
│   ├── pw_afm.py         # original full implementation incl. falsified variants (provenance)
│   ├── exp_*.py          # one script per paper section
│   ├── gen_paper_figs.py # publication figures
│   └── run_all.py        # reproduction runner with golden-number checks
├── tests/
│   └── smoke_test.py     # asserts paper-level accuracy + editing behavior
├── paper/                # PAPER.tex/.md, TECH_REPORT.md, RESULTS_MULTISEED.md
├── notebooks/
│   └── demo.ipynb        # interactive demo (Colab-ready)
├── results/              # generated figures (gitignored)
├── pyproject.toml        # pip install dp-proton
└── requirements-*.txt
```

---

## Citing

If you use this code or the paper's numbers, please cite the paper
(bibtex will be added at publication):

```
Feng, Y. (2026). DP-ProtOn: Dirichlet-Process-guided Prototype Online
Learning with Zero-Retraining Knowledge Editing.
```

## License

MIT — this is a public-good baseline; use it freely, cite it honestly.
