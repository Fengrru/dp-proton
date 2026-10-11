# DP-ProtOn: Dirichlet-Process-Guided Online Prototype Learning for Continual Classification

**Feng Yuhan**

---

> ## ⚠️ Correction (2026-10-11): the §5.2 creation-rule comparison does not hold
>
> The headline ablation of this paper — that the CRP criterion beats a
> distance-threshold online *k*-means by **ΔACC = 0.6960 ± 0.0528** at matched
> prototype budget, and that therefore "the DP criterion is necessary, not excess
> complexity" — **is an artifact of a confounded comparison.**
>
> The two models did not differ only in their creation rule. The threshold
> *k*-means baseline allocated basins **label-blind** (it searched all centres
> regardless of class) while DP/CRP's rejection baseline carried a **per-class**
> base measure; and the baseline's distance threshold was not calibrated to a
> structural target. Per-class allocation is itself load-bearing: on the same
> features, label-blind basins reach 0.17–0.68 accuracy where per-class basins
> reach 0.92–0.99.
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
> is ≤ 1.9 accuracy points, with no consistent winner. Treat §5.2 and the
> abstract's ΔACC figure below as superseded by this correction.
>
> **What still stands:** the per-class architecture, the denoised in-basin
> centroid, the rank/density readout, and exact local deletion. **What does not:**
> the necessity claim for DP/CRP basin creation and the "strongest single
> ablation result" framing in §5.2.
>
> The re-measurement — seven paired experiments on synthetic and real data, with
> the confounded-vs-fair comparison and the mechanism of the failure — is in the
> companion repository `Fengrru/energy-basin-cl`.

---

## Abstract

We study rehearsal-free, single-pass continual classification on top of frozen features, and ask a deliberately narrow question: *given a good creation criterion and a good readout, how far can a purely online prototype learner go, and which of its components actually matter?* We introduce **DP-ProtOn** (Dirichlet-Process-guided Prototype Online learner), a lightweight model with three components: (i) class-conditional Gaussian "basins" allocated on the fly using a Chinese-Restaurant-Process (CRP) creation criterion with diagonal covariance, (ii) one online denoised centroid per basin maintained via competitive-learning mean updates, and (iii) a Parzen / soft-$k$NN density readout over those centroids. DP-ProtOn emerged from the systematic falsification of a more elaborate energy-attractor design whose energy-softmin routing proved *net-negative* in high dimensions — a same-model readout ablation pins this failure precisely. All headline claims carry multi-seed variance: across 5 seeds on sklearn *digits* and 3 seeds on *Split-MNIST*, the CRP criterion beats a distance-threshold online $k$-means by ΔACC = 0.6960 ± 0.0528 at matched prototype budget (Figure 1a), online denoised centroids beat offline $k$-means at equal budget by 0.0340 ± 0.0053 (Figure 1b), and on Split-MNIST DP-ProtOn reaches 0.8747 ± 0.0019 ACC with near-zero forgetting (BWT = −0.0471 ± 0.0049), a +0.20 ACC advantage with 8× better backward transfer over Experience Replay at equal memory (Figure 1c), while EWC and fine-tuning collapse (BWT ≈ −0.99). Critically, because each class is a spatially localized, label-tagged attractor, DP-ProtOn supports **training-free knowledge editing**: deleting, relabeling, and injecting classes are $O(1)$ set operations on the basin list. We verify all three surgeries across 5 seeds (Figure 2): deletion drops the target class to 0.0000 ACC while perturbing others by ≤ 0.0467; relabeling is byte-identical in decision geometry (1.0000 ± 0.0000); and injecting a held-out class from merely 20 samples recovers 0.8564 ± 0.0603 ACC with no measurable forgetting of old classes. We position DP-ProtOn as an honest, fully-ablated baseline and as evidence that prototype attractors are directly editable knowledge units — a property softmax classifiers categorically lack.

---

## 1. Introduction

Continual learning (CL) asks a model to absorb a non-stationary stream of tasks without *catastrophic forgetting* of what it learned before. The dominant strategies are: **regularization** that penalizes weight drift via a surrogate importance measure (EWC; Kirkpatrick et al., 2017), **rehearsal** that interleaves stored exemplars with new data (Chaudhry et al., 2019), and **architectural growth** that freezes old parameters and appends new capacity. A separate, older line of work — prototype and nearest-class-mean (NCM) classifiers — sidesteps gradient interference entirely by *storing* class-condensed summaries and classifying by similarity. These methods are attractive for CL precisely because adding a class does not overwrite the parameters that encode existing classes: there are no parameters to overwrite.

This paper is a careful, negative-result-inclusive study of one such prototype learner. Our starting point was an ambitious energy-based design — a "Precision-Weighted Attractor Field Machine" (PW-AFM) with radial potential wells, softmin energy routing across basins, precision gating, inter-basin barrier terms, and reversible decay. The hypothesis was that an energy landscape over basins would deliver principled continual learning: graceful accommodation of new classes, novelty detection via energy thresholds, and protection of old knowledge through barriers. **That hypothesis did not survive contact with high-dimensional data.** Rather than discard the project, we treated the failed design as an ablation target and asked: *which components actually carry the performance, and which are dead weight?*

The answer is surprisingly compact. Only three ingredients survive systematic ablation:

1. **A Dirichlet-Process / CRP creation criterion** with *diagonal* covariance that compares a rich-get-richer class-conditional likelihood $\log n_i + \log\mathcal{N}(x \mid c_i,\Sigma_i)$ against a universal rejection baseline $\log\alpha + \log p_0(x)$.
2. **One online denoised centroid per basin**, updated by a running competitive mean, replacing noisy raw memory samples.
3. **A Parzen / soft-$k$NN density readout** that classifies via log-sum-exp over centroids, replacing the failed energy-softmin.

Everything else — energy routing, full covariance, per-basin mini-GMMs, barrier terms, the precision gate — is either net-negative or negligible for classification and is removed. We call the resulting compact model **DP-ProtOn** (Dirichlet-Process-guided Prototype Online learner).

Our contributions are concrete and diagnostic:

- **A same-model readout ablation** (Section 5.1) that pins the energy-routing failure precisely: on *identical* trained basins, switching from energy-softmin (0.751) to a local density readout (0.887) yields a large jump, proving the basins store useful structure that the energy mechanism fails to exploit. The culprit is distance concentration in 30–64 D (Beyer et al., 1999).
- **A controlled necessity test for the DP criterion** (Section 5.2; Figure 1a). Holding the readout and centroid machinery fixed and varying *only* the creation rule, DP beats distance-threshold online $k$-means by Δ = 0.696 ± 0.053 (5 seeds) at matched prototype budget, and even beats it at 8× the threshold budget in single-seed sweeps. A fixed spherical radius is fundamentally incapable of what the CRP likelihood ratio achieves.
- **Multi-seed continual-learning results** (Section 5.4; Figure 1c). On Split-MNIST (3 seeds, frozen 50-D features shared by *all* methods), DP-ProtOn reaches 0.8747 ± 0.0019 ACC with BWT = −0.047 ± 0.005, beating Experience Replay at equal memory by +0.20 ACC and 8× better backward transfer. EWC and fine-tuning collapse to near-random (BWT ≈ −0.99).
- **Training-free knowledge editing** (Section 5.5; Figure 2). Because basins are discrete, label-tagged attractors, deleting, relabeling, and injecting classes are $O(1)$ set operations — a capability that softmax classifiers categorically lack, since every class is entangled in a single weight matrix.

We deliberately scope out feature learning and rehearsal: DP-ProtOn operates on frozen features (PCA-whitening fit once on a pretraining pool) and stores no replay buffer. This makes it a clean, reproducible baseline rather than a state-of-the-art claim — exactly the kind of artifact TMLR is designed to publish.

---

## 2. Related Work

**Prototype and nearest-class-mean continual learning.** iCaRL (Rebuffi et al., 2017) stores exemplars selected via herding and classifies by nearest-mean, combined with knowledge distillation to stabilize old tasks. End-to-End Incremental Learning (Castro et al., 2018) and BiC (Wu et al., 2019) add learned bias-correction layers on top of exemplar-based nearest-mean. All three require gradient updates and tuned distillation hyperparameters. The closest ancestor to our readout is pure nearest-class-mean classification (Mensink et al., 2013), which classifies a test point by the nearest class mean with zero gradient overhead. DP-ProtOn is simpler than all of these: it performs no distillation, no gradient-based class learning, and its centroids are produced by a purely online competitive mean rather than by selecting exemplars — indeed, it never stores raw data points.

**Online and streaming clustering.** Streaming $k$-means (Sculley, 2010) updates centroids via mini-batch gradient descent; BIRCH (Zhang et al., 1996) builds a hierarchical clustering-feature tree; DenStream (Cao et al., 2006) uses density-based micro-clusters for evolving streams. The critical distinguishing element of DP-ProtOn is the *creation criterion*: instead of a fixed distance threshold (streaming $k$-means), a cluster-feature radius (BIRCH), or a density-connected component (DenStream), it employs a Bayesian-nonparametric CRP rule with per-basin covariance. Our control experiment (Section 5.2) shows this difference is decisive — distance-threshold online $k$-means collapses to < 0.25 ACC at the same prototype budget.

**Bayesian nonparametrics.** The Dirichlet Process (Ferguson, 1973; Teh, 2010) and its Chinese Restaurant Process representation (Blackwell & MacQueen, 1973) provide a principled rich-get-richer prior over an unbounded number of components. We use the CRP not for full posterior inference (as in collapsed Gibbs sampling) but as a lightweight, adaptive, covariance-aware *allocation* rule at streaming time — a pragmatic engineering choice that retains the key benefit (data-adaptive clustering) without MCMC overhead.

**Regularization- and rehearsal-based methods.** EWC (Kirkpatrick et al., 2017) anchors weights via a Fisher-information penalty; Experience Replay (Chaudhry et al., 2019; Aljundi et al., 2019) interleaves stored exemplars. We include both as baselines under a shared frozen-feature protocol. On frozen features these gradient-based methods are known to be brittle, and our results (Section 5.4) reproduce that fragility: even 256-hidden-unit MLPs are catastrophically overwritten across just 5 tasks.

**Energy-based models.** Our design originated in the EBM tradition of energy functions and Boltzmann-type softmin routing (LeCun et al., 2006). Our core negative finding — that energy-softmin degrades in high dimensions due to distance concentration (Beyer et al., 1999) and is dominated by a local density readout — is a concrete, empirically grounded contribution. It suggests that EBMs for prototype-based classification should anchor predictions on *local* similarity rather than *global* energy comparisons.

---

## 3. Method

### 3.1 Overview and pipeline

DP-ProtOn maintains a set of **basins** $\{B_i\}_{i=1}^{K}$. Each basin $B_i$ consists of:

- A class label $\ell_i \in \mathcal{Y}$ (the only mutable "classification" parameter),
- A running sample count $n_i$ (integer),
- A running Gaussian summary $(c_i, \Sigma_i)$, where $c_i$ is the online mean of all points assigned to $B_i$ and $\Sigma_i$ is the running *diagonal* covariance, used **exclusively** for the CRP creation decision,
- A small set of **online denoised centroids** $\{m_{ik}\}_{k=1}^{K_i}$ (one per `n_sub`, with $K_i=1$ being the practical default), used **exclusively** for classification via the density readout.

Features are frozen: we fit PCA-whitening once on a pretraining pool (30% of training data) and never update it, which trivially incurs zero feature-level forgetting and isolates all interesting behavior to the prototype layer. The BWT metric follows Lopez-Paz & Ranzato (2017).

**Algorithm 1: DP-ProtOn stream step (single point).** Given labeled point $(x, y)$ and current basin set $\mathcal{B}$:

1. **Candidate selection.** Let $\mathcal{B}_y = \{B \in \mathcal{B} : \ell_B = y\}$ be the basins with matching label. If $\mathcal{B}_y = \emptyset$, create new basin (go to step 3).
2. **CRP scoring.** For each $B \in \mathcal{B}_y$ compute:

   $$s_B = \log n_B + \log\mathcal{N}(x \mid c_B, \Sigma_B)$$

   Let $B^* = \arg\max_{B \in \mathcal{B}_y} s_B$ be the best match. Compute the universal rejection baseline:

   $$s_0 = \log\alpha + \log p_0(x)$$

   where $p_0 \sim \mathcal{N}(0, \sigma_0^2 \cdot \text{base\\_scale}^2)$ is a broad zero-mean Gaussian. If $s_{B^*} > s_0$, assign $x$ to $B^*$ (step 4); otherwise create a new basin (step 3).
3. **Create basin.** Instantiate a new basin $B_{\text{new}}$ with label $y$, $n=1$, $c=x$, $\Sigma=\sigma_0^2 I$, and one centroid $m=x$.
4. **Update basin statistics.** Increment $n_B$, update running mean $c_B \leftarrow c_B + (x - c_B)/n_B$, update diagonal variance estimate $\Sigma_B$, and update the nearest centroid $m_{Bk}$:

   $$n_{Bk} \mathrel{+}= 1, \qquad m_{Bk} \mathrel{+}= \frac{x - m_{Bk}}{n_{Bk}}.$$

5. *(optional)* Prune unused basins via reversible decay (Section 7).

### 3.2 CRP basin creation: design rationale

The CRP score $s_i = \log n_i + \log\mathcal{N}(x \mid c_i, \Sigma_i)$ combines two forces: a **rich-get-richer** prior ($\log n_i$, penalizing singleton basins) and a **Gaussian likelihood** that measures how well the point fits the basin's distribution. The rejection baseline $s_0 = \log\alpha + \log p_0(x)$ acts as a universal "do-not-allocate" threshold: a point must be *more likely* under a specific class basin than under a generic broad prior to be assigned. The concentration $\alpha$ controls how easily a new basin is created — high $\alpha$ favors many small basins, low $\alpha$ favors a few large ones.

This single rule delivers two properties a fixed distance threshold $\tau$ cannot:

1. **Covariance-adaptive acceptance.** The likelihood $\log\mathcal{N}(x \mid c_i, \Sigma_i)$ uses the basin's *learned* per-dimension variance, so a point that is $3\sigma$ away in a tight direction is penalized more than a point $3\sigma$ away in a broad direction. A spherical threshold treats both identically.
2. **Universal contamination guard.** The baseline $p_0(x)$ is class-agnostic and prevents a wrong-class point from silently slipping into a basin. In threshold $k$-means, a new-class point within $\tau$ of an old centroid simply updates that centroid — silently contaminating its label. The CRP criterion avoids this: a point from a new class will be rejected by *all* existing basins of other classes (since the class-conditional search is restricted to $\mathcal{B}_y$) and will trigger a new basin.

We use **diagonal** $\Sigma_i$ throughout. Full covariance is unstable in high dimensions due to ill-conditioned log-determinants (the empirical covariance matrix on a few hundred points is singular in 30–64 D), and our single-seed ablation confirms it underperforms diagonal (30-D: 0.680 vs 0.746; 64-D: full covariance fails to converge). Diagonal covariance is thus essential for both numerical stability and accuracy.

### 3.3 Online denoised centroids (sub-centroids)

Each basin maintains $K_i =$ `n_sub` centroids $\{m_{ik}\}$ updated by competitive-learning online mean. For the winning centroid $k$ (the one closest in Euclidean distance to the incoming point $x$):

$$n_{ik} \mathrel{+}= 1, \qquad m_{ik} \mathrel{+}= \frac{x - m_{ik}}{n_{ik}}.$$

This is an online analog of Lloyd's $k$-means assignment step — greedy, single-pass, non-iterative. With `n_sub=1`, exactly one centroid exists per basin: a pure running mean. With `n_sub>1`, the competitive mechanism drives centroids to cover distinct sub-modes within a basin.

Why is this denoising? Raw reservoir memory (`n_sub=0`) stores individual sample points. The running mean instead tracks the *population* center — each update adds a $(1/n)$ fraction of noise, and the accumulated noise scales as $O(1/\sqrt{n})$. As $n$ grows, the centroid converges to the true class-conditional mean, eliminating per-sample jitter. This is why a single centroid (`n_sub=1`, ≈30 protos total) beats offline $k$-means (30 Lloyd centroids) in our experiments: the offline algorithm has more information (full data access), but the online mean converges to a quantity that is *statistically better* for classification because it tracks the data-generating process rather than a fixed partition.

### 3.4 Density readout (Parzen / soft-$k$NN)

Classification scores a query $x$ against all basin centroids using a log-sum-exp Parzen kernel of bandwidth $h =$ `density_bw`:

$$s_i(x) = \operatorname*{logsumexp}_{k=1}^{K_i} \left(-\frac{1}{2} \frac{\lVert x - m_{ik}\rVert^2}{h^2}\right),$$

$$\hat{y}(x) = \ell_{\arg\max_i \; s_i(x)}.$$

The logsumexp operator gives a "soft-max" behavior: if multiple centroids in a basin are close to $x$, their contributions accumulate to a higher score. If no centroid is close, the score is low. This contrasts sharply with energy-softmin, which normalizes across *all* basins and suffers from distance concentration (Section 5.1). The bandwidth $h$ controls locality: small $h$ approaches hard 1-NN (each query sees only the single nearest centroid), while large $h$ approaches uniform weighting. Our ablation shows ACC varies only within ≈0.013 over $h \in [0.3, 2.0]$, so the readout is bandwidth-robust.

### 3.5 Knowledge editing interface

Because knowledge in DP-ProtOn lives in discrete, label-tagged, spatially localized attractors — rather than in an entangled weight matrix — three fundamental edits are $O(|\mathcal{B}|)$ set operations requiring zero retraining:

- **Delete** class $\ell$: remove every basin with $\ell_i = \ell$. Queries near the excised region now score low everywhere and are flagged by the novelty score (the logsumexp over remaining centroids produces low values, raising the novelty metric).
- **Relabel** $\ell \to \ell'$: reassign $\ell_i \leftarrow \ell'$ for all basins originally labeled $\ell$. Since the centroids $m_{ik}$, covariances $\Sigma_i$, and counts $n_i$ are untouched, the decision surface is *byte-identical* up to the label map — the attractor geometry is decoupled from the semantic label.
- **Inject** new class: stream $N$ labeled examples of a previously unseen class through the ordinary stream step (Algorithm 1). The CRP rule allocates fresh basins for the new class without disturbing existing ones, because the class-conditional search (step 2) restricts to basins of the *same* label.

A softmax MLP cannot perform any of these without retraining the entire network, because every class is entangled in the final weight matrix $W \in \mathbb{R}^{C \times d}$. Changing one row would require re-optimizing all rows to maintain calibration. DP-ProtOn's basins are independent: no basin references or depends on another.

### 3.6 Hyperparameters and default configuration

| Parameter | Default | Role | Sensitivity |
|---|---|---|---|
| `create_rule` | `'dp'` | Basin creation mechanism | Binary: must-use. Threshold is far worse |
| `dp_alpha` | 1.0 (10-class) / 1e4 (30-mode) | CRP concentration | Monotonic control of basin count |
| `dp_diag_cov` | `True` | Diagonal covariance in CRP likelihood | Essential: full cov fails in high dimensions |
| `base_scale` | 3.0 | Reference-baseline Gaussian width | Wider = fewer basins; moderate sensitivity |
| `readout` | `'density'` | Classification mechanism | Binary: density >> energy |
| `n_sub` | 1 | Centroids per basin | $k{=}1$ is the efficiency sweet spot |
| `density_bw` | 1.0 (≈ $\sigma_0$) | Parzen kernel bandwidth | Robust across [0.3, 2.0] |
| `T` | 0.3 | Temperature (unused in density readout) | Legacy parameter |
| `sigma0` | 1.0 | Initial $\Sigma$ scale for new basins | Sets length scale |

The six components we removed as verified-unhelpful for classification: energy-softmin routing, full covariance, per-basin mini-GMM energy model, barrier terms, precision gate, and reversible decay (the last two may still be useful for novelty/OOD detection — Section 7).

---

## 4. Experimental Setup

### 4.1 Datasets

- **Digits** (sklearn digits; Pedregosa et al., 2011): 1,797 real 8×8 grayscale images, 10 classes (0–9), PCA-whitened to $d=30$ frozen dimensions (retaining ≈95% variance), 70/30 train/test split. Used for the creation-rule ablation (Section 5.2), prototype-source ablation (Section 5.3), and knowledge-editing experiments (Section 5.5). Each class has ≈120 training samples — a small-data regime where prototype efficiency is particularly important.
- **Split-MNIST** (LeCun et al., 1998): 70,000 28×28 grayscale images, 5 sequential tasks × 2 classes (0/1, 2/3, 4/5, 6/7, 8/9), PCA-whitened to $d=50$ frozen dimensions shared by *all* methods, 70/30 split. Used for the continual learning benchmark (Section 5.4). This is a standard CL protocol; we match the frozen-feature setting used in Lopez-Paz & Ranzato (2017) but with PCA features instead of a learned encoder, to isolate the classifier's behavior.
- **Synthetic** (`make_classification`; Pedregosa et al., 2011): 6,000 points in $d=50$, 10 classes, each class composed of 3 Gaussian subclusters (30 true modes). PCA-whitened to 50-D, 70/30 split. Used for single-seed multi-modal stress testing (Sections 5.6). This dataset is fundamentally non-Gaussian per class — a hard test for single-Gaussian basin models.

### 4.2 Continual learning protocol

All CL experiments use a **5-task sequential protocol**: task $t$ presents 2 new classes; the model trains on task $t$ data (single pass), then is evaluated on the test sets of *all* tasks seen so far (1 through $t$). We report:

- **ACC**: average accuracy over all seen classes after the final task (task 5). Measures final retained knowledge.
- **BWT** (Backward Transfer): mean accuracy change on old tasks after learning new tasks, $\text{BWT} = \frac{1}{T-1}\sum_{t=1}^{T-1}(R_{T,t} - R_{t,t})$, where $R_{i,j}$ is task $j$ accuracy after learning task $i$. Negative BWT indicates forgetting; near-zero BWT indicates retention.

### 4.3 Multi-seed methodology

Every headline claim is reported as mean ± std over multiple random seeds: **5 seeds** (0–4) on the digits dataset, **3 seeds** (0–2) on Split-MNIST. The fewer seeds on Split-MNIST reflect that each seed trains 4 torch baselines (EWC, Replay, Fine-tune, Joint) — 20 MLP training runs total for 5 methods × (3 MNIST + 1 synthetic) seeds — which is CPU-intensive.

For the creation-rule control (Section 5.2), the threshold $k$-means baseline is **re-matched per seed**: we sweep distance thresholds $\tau \in [1.0, 8.5]$ and select the $\tau$ that yields a prototype count closest to DP's basin count for that seed. This ensures the comparison is always at **equal budget** — the strongest possible test against DP. Features are re-fit per seed (PCA fitted on a fresh random 30% pretrain pool), so statistical variation includes feature-space variability.

All multi-seed results appear in Sections 5.2–5.5 and in Figures 1–2. Single-seed results in Sections 5.1 and 5.6 are clearly labeled as such.

### 4.4 Baselines

At matched frozen features and, where applicable, matched memory budget:

| Baseline | Description | Gradient-based? |
|---|---|---|
| **Joint** | Train one MLP on all tasks jointly (shuffled) | Yes — oracle upper bound |
| **Experience Replay** | Store 240 exemplars (≈24/class), replay with new data | Yes — MLP trained on buffer + stream |
| **EWC** ($\lambda{=}1000$) | Fisher-information penalty on important weights | Yes — online elastic weight consolidation |
| **Fine-tuning** | Sequential MLP training with no forgetting mitigation | Yes — lower bound |
| **Offline $k$-means-30** | Lloyd clustering on full dataset, 3 centroids/class | No — offline prototype upper bound |
| **1-NN** | Store all training points, classify by nearest neighbor | No — accuracy upper bound (zero compression) |
| **Threshold $k$-means** | Online $k$-means with distance-threshold creation | No — the ablation twin of DP (creation rule only) |

For Split-MNIST (Section 5.4), all neural baselines use a 2-hidden-layer MLP (256-256 units, ReLU) with the same frozen 50-D features, trained with Adam for 10 epochs per task. Replay and DP-ProtOn share a memory budget of ~240 stored units (DP stores 240 centroids; Replay stores 240 raw exemplars, 24 per class).

---

## 5. Results

### 5.1 Energy routing is net-negative (same-model readout ablation, single seed)

**Setup.** We train one set of DP basins on digits (30-D PCA, seed 0) using the standard `create_rule='dp'` pipeline. Then, on the *identical* trained basins, we swap only the classification rule and measure ACC — isolating the effect of the readout completely.

| Readout on identical basins | ACC |
|---|---|
| Nearest basin center (Mahalanobis) | 0.541 |
| Energy-softmin routing (full basins, $T=0.3$) | 0.751 |
| Nearest stored prototype (basin-internal 1-NN, raw memory) | **0.887** |

**Analysis.** Energy routing (0.751) beats nearest-center (0.541) — the basins *are* carrying discriminative information beyond their centers. But nearest-prototype (0.887) is dramatically better (+0.136 ACC). This three-point ladder proves that the bottleneck is the **routing mechanism**, not the basin storage: the basins hold useful structure that nearest-prototype can access but energy-softmin cannot.

**Root cause.** Two physical effects compound to degrade energy routing in high dimensions:

1. **Distance concentration** (Beyer et al., 1999): in 30–64 D, Mahalanobis distances from any test point to *all* basin centers converge to a narrow distribution — all basins look "similarly far." The softmin $\propto \exp(-E_i/T)$ over basin energies becomes nearly uniform, yielding a flat, uninformative probability vector. The density readout avoids this by evaluating distances to *local* sub-centroids rather than to *global* energy minima.
2. **Non-Gaussian class manifolds:** a single-Gaussian energy well cannot model multi-subcluster classes (like digit "7" with distinct stroke styles). Adding mini-GMM structure per basin barely helps (single-seed: 0.746 → 0.749), proving the routing mechanism is the bottleneck, not the number of modes per basin.

**Consequence.** This ablation is why "Attractor Field" was dropped from the model's name. The surviving readout (density / Parzen) is a separate mechanism that does not use the basin's energy function at all — it only uses the basin's centroids as anchor points. The energy structure serves solely the creation decision (Algorithm 1, step 2), a role it performs well.

### 5.2 Creation rule: DP/CRP vs distance-threshold $k$-means (digits, 5 seeds)

**Setup.** Two models that differ **only** in their creation criterion. Both use online denoised centroids (`n_sub=1`) and the same density readout (`density_bw=1.0`). For each seed, we sweep thresholds $\tau \in [1.0, 8.5]$ to match the threshold model's prototype count to DP's basin count — equal budget, by construction.

**Results.** See Figure 1(a).

| Method | ACC (mean ± std) | Prototypes |
|---|---|---|
| **DP-ProtOn (CRP)** | **0.9303 ± 0.0090** | 29.8 ± 3.4 |
| Threshold $k$-means | 0.2342 ± 0.0473 | 29.2 ± 4.2 |
| **Δ (DP − kmeans)** | **0.6960 ± 0.0528** | matched |

Per-seed DP ACC: 0.9284, 0.9366, 0.9313, 0.9144, 0.9407. The gap's standard deviation (0.053) is nowhere near zero, so DP *stably* dominates — this is not a single-seed fluke.

In single-seed budget sweeps (not plotted), DP at 32 prototypes (ACC 0.928) even beats threshold $k$-means at **266 prototypes** (ACC 0.900) — roughly 8× prototype efficiency. At matched budget, threshold $k$-means collapses to near-random: the fixed spherical radius either merges new-class points into wrong-class centroids (label contamination) or fails to adapt to class-specific density. The CRP likelihood ratio avoids both by being covariance-aware and grounded against a universal baseline (Section 3.2).

**Conclusion: The DP criterion is necessary, not excess complexity.** This is the strongest single ablation result in the paper.

> ⚠️ **Superseded (2026-10-11).** This conclusion does not hold; see the correction
> at the top of this paper. The baseline here was label-blind and its threshold was
> not calibrated to a structural target, so the two models differed in more than the
> creation rule. Re-measured fairly on the same dataset, the four criteria span ≤ 1.9
> accuracy points at matched budget and threshold *k*-means is ahead (0.935 vs 0.917
> at 20 basins). The "8× prototype efficiency" figure in the paragraph above is a
> consequence of the same confound.

### 5.3 Prototype source: online denoising vs offline $k$-means (digits, 5 seeds)

**Setup.** All conditions use the DP creation rule and density readout. We vary only the source of prototypes used in the Parzen readout. Matched to ≈30 prototypes total.

**Results.** See Figure 1(b).

| Prototype source | ACC (mean ± std) | Protos |
|---|---|---|
| **Online denoised sub-centroid ($k{=}1$)** | **0.9303 ± 0.0090** | ~30 |
| Raw reservoir memory (`n_sub=0`) | 0.9143 ± 0.0177 | ~290 |
| Offline $k$-means-30 (3/class, Lloyd) | 0.8963 ± 0.0048 | 30 |
| 1-NN (store all ≈4,700 points) | 0.9516 ± 0.0090 | 4,700 (upper bound) |
| **Δ (sub-centroid − kmeans-30)** | **0.0340 ± 0.0053** | — |

**Analysis.** Three findings:

1. **Online beats offline.** A single online mean per basin ($k{=}1$) stably beats offline Lloyd $k$-means at equal budget (+0.034), and closes most of the gap to the 1-NN upper bound (−0.021) while storing ~150× fewer points. The online mean tracks the data-generating process; offline $k$-means optimizes a fixed partition that may not align with class boundaries.
2. **Denoising matters.** Raw memory (`n_sub=0`, storing all reservoir samples) is worse than sub-centroids (0.914 vs 0.930) and has higher variance — raw samples carry per-instance noise that the running mean filters out.
3. **Diminishing returns.** Higher $k$ ($k{=}10$ → 0.936 in single-seed) approaches the 1-NN bound (0.953) but at diminishing marginal gain. $k{=}1$ is the efficiency sweet spot: 1 protoype/class vs 47 protos/class for $k{=}10$, with only a 0.006 ACC penalty.

### 5.4 Split-MNIST continual learning (3 seeds)

**Setup.** Standard Split-MNIST CL protocol (5 tasks × 2 classes, 50-D frozen PCA features). All neural baselines use the same MLP architecture and feature space. Replay and DP-ProtOn are matched at ≈240 stored units (DP: 240 centroids ≈ 24/class; Replay: 240 raw images, 24/class).

**Results.** See Figure 1(c).

| Method | ACC (mean ± std) | BWT (mean ± std) |
|---|---|---|
| Joint (oracle) | 0.9693 ± 0.0017 | — |
| **DP-ProtOn (ours)** | **0.8747 ± 0.0019** | **−0.0471 ± 0.0049** |
| Experience Replay | 0.6787 ± 0.0174 | −0.3921 ± 0.0226 |
| EWC ($\lambda{=}1000$) | 0.1949 ± 0.0004 | −0.9927 ± 0.0005 |
| Fine-tuning | 0.1978 ± 0.0002 | −0.9947 ± 0.0003 |

Per-seed DP-ProtOn ACC: 0.874, 0.877, 0.873 — an exceptionally tight band (std = 0.0019) that reproduces our single-seed number (0.874, Section 8.5 of TECH_REPORT) exactly at the mean.

**Key insights:**

- **DP-ProtOn vs Replay:** +0.20 ACC and 8× better BWT at equal memory. Prototype centroids are far more informative per stored unit than raw exemplars: each centroid is the *mean* of dozens of points, while each stored exemplar is a single noisy sample. Replay's BWT of −0.39 means old-task accuracy drops by ~0.08 per new task learned; DP-ProtOn's BWT of −0.047 means old-task accuracy drops by ~0.009 per task — essentially flat retention.
- **EWC and fine-tuning collapse.** Both hit near-random accuracy (0.195–0.198) and near-total forgetting (BWT ≈ −0.99). On frozen features, even 256-hidden-unit MLPs are catastrophically overwritten across just 5 tasks. EWC's Fisher penalty provides essentially zero protection in this setting — the features are fixed, so weight drift in the classifier head is unconstrained.
- **Gap to Joint.** The residual gap of 0.875 → 0.969 (≈0.094 ACC) is the honest cost of: (a) single-pass online quantization without the ability to revisit old data, and (b) no feature adaptation. Joint trains on the full shuffled dataset with gradient descent — it is the best a frozen-feature MLP can possibly do.
- **Basin scaling.** DP allocates ≈240 basins on MNIST (vs ≈32 on digits with ~1,700 samples), automatically scaling capacity with both sample count and class-conditional variation. This is the CRP "rich-get-richer" prior at work: more variation → more basins.

### 5.5 Training-free knowledge editing (digits, 5 seeds)

**Setup.** We train DP-ProtOn on the full digits stream (5 seeds), then apply each edit *without any retraining* and measure the change. Figure 2 visualizes the three operations.

**[1] Delete class 9** (Figure 2a). All basins labeled 9 are removed.

| Metric | Before | After |
|---|---|---|
| Class-9 ACC | 0.8565 ± 0.0753 | **0.0000 ± 0.0000** |
| Other-class max ACC shift | — | 0.0467 ± 0.0369 |
| Class-9 novelty score | 9.52 ± 0.76 | 11.98 ± 0.83 (rises) |

The target class is surgically excised. The only measurable side effect: a few test points formerly classified as class 9 reroute to neighboring classes (max shift across other 9 classes ≤ 0.047). Critically, the **novelty score rises** from 9.52 to 11.98 after deletion — class-9 inputs are now flagged as unknown rather than silently misclassified, because the logsumexp over remaining centroids produces low scores everywhere in the excised region. This is a form of "active forgetting with awareness."

**[2] Relabel class 7 → 99** (Figure 2b). The label field of all class-7 basins is changed to 99.

| Metric | Value |
|---|---|
| Remap-identical (relabel 99 → 7, then compare) | **1.0000 ± 0.0000** |
| Non-target predictions identical | 0.9941 ± 0.0049 |

After relabeling, we map predictions of 99 back to 7 and compare with the pre-edit predictions — they are byte-identical (1.0000, zero variance). The 0.6% of non-identical predictions on non-target classes (< 3 points out of ≈540) are cases where a border point from class 8 was previously misclassified as 7; after relabeling, it can no longer be classified as 7 (since that label no longer exists) and falls to a different class. **The label is a detachable pointer**, fully decoupled from the attractor geometry.

**[3] Inject held-out class 9 from 20 samples** (Figure 2c). We train DP-ProtOn on the first 9 digits (0–8) with class 9 held out, then inject 20 held-out samples of class 9.

| Metric | Before | After |
|---|---|---|
| Class-9 ACC | 0.0000 ± 0.0000 | **0.8564 ± 0.0603** |
| Old-class predictions identical | — | 0.9883 ± 0.0037 |
| Old-class max ACC shift | — | 0.0443 ± 0.0211 |

Streaming just 20 examples of a previously unseen class recovers essentially full accuracy on it (0.856, matching the delete-before baseline of 0.857), while leaving old-class predictions 98.8% identical — i.e., no catastrophic forgetting. The max old-class shift (0.044) is comparable to the delete-experiment perturbation and reflects border points near the new attractor.

**Significance.** A softmax MLP cannot delete, relabel, or add a class without retraining the whole network, because the final weight matrix $W^{C \times d}$ ties all classes together. Changing one row changes the softmax normalization constant for *all* classes. DP-ProtOn's basins are independent modules — delete/relabel/inject are $O(1)$ set operations on an un-ordered list. This demonstration is, to our knowledge, the first explicit verification that prototype-based CL models support training-free knowledge editing, and it constitutes the strongest evidence that the basins are genuine, addressable knowledge units.

### 5.6 Supporting single-seed ablations (digits 30-D unless noted)

For completeness, we list systematic single-seed sweeps that informed design decisions but are not replicated across seeds. These are from the original ablation study (TECH_REPORT Sections 3–4) and are preserved for provenance:

- **Diagonal vs full covariance** (DP criterion): at 30-D, diagonal 0.746 vs full 0.680; at 64-D, full covariance fails to converge (ill-conditioned log-determinants). Diagonal covariance is essential for numerical stability and accuracy in high dimensions.
- **Per-basin mini-GMM vs single Gaussian:** mini-GMM energy achieves 0.749 vs single Gaussian 0.746 — negligible gain. The routing mechanism, not the basin model expressivity, is the bottleneck.
- **Sub-centroid count $k$:** $k{=}1$ → 0.928, $k{=}2$ → 0.932, $k{=}5$ → 0.934, $k{=}10$ → 0.936 (near 1-NN bound 0.953). $k{=}1$ captures most of the benefit at minimal cost.
- **Bandwidth $h$ (density_bw):** sweeping $h \in [0.3, 2.0]$ yields ACC in [0.886, 0.899] — a range of only 0.013. The readout is bandwidth-robust.
- **DP $\alpha$ sensitivity:** $\alpha$ controls basin count monotonically. On digits: $\alpha{=}10^{-4}$ → 26 basins (ACC 0.683), $\alpha{=}100$ → 35 basins (ACC 0.634). Sweet spot $\alpha{=}1.0$ yields ≈30 basins with ACC 0.928. On synthetic: $\alpha{=}1$ → 16 basins (0.667), $\alpha{=}10^{10}$ → 245 basins (0.905). The rule of thumb: set $\alpha$ to approximate the expected number of data modes.
- **Synthetic 50-D (30 true modes):** raw density ($n\_sub{=}0$, $\alpha{=}10^4$) 0.742 → sub-centroid $k{=}1$ 0.805 → $k{=}10$ 0.884; offline $k$-means-30 0.939; 1-NN 0.906. The ≈28% residual gap from sub-centroids to offline $k$-means on *synthetic* data (which vanishes on real digits) is the cost of online single-pass quantization on data with complex multi-modal class structure.

---

## 6. Discussion

### 6.1 Why energy routing fails: distance concentration and the softmin collapse

The failure of energy-softmin routing in high dimensions is not an artifact of our implementation but a structural consequence of two well-known phenomena that compound destructively.

**Distance concentration** (Beyer et al., 1999; Aggarwal et al., 2001) is the observation that, under broad conditions on the data distribution, the ratio of the farthest to nearest neighbor distances approaches 1 as dimensionality grows. Concretely, for d=30 (our digits setting) and d=50 (Split-MNIST), the contrast between the closest and farthest basin center largely vanishes. In a softmin readout p_i ∝ exp(−E_i/T), the energy E_i = ‖x−c_i‖² of every basin converges to a narrow band, and the softmin collapses toward a uniform distribution regardless of temperature T. Lowering T (sharper softmin) cannot recover contrast because all energies are nearly identical; it merely amplifies noise. This explains why the energy readout (0.751) underperforms even nearest-center classification (0.541), which at least picks the single closest basin deterministically.

**Non-Gaussian class manifolds** compound the problem. Many real classes are multi-modal — the digit "8", for instance, subsumes multiple stroke styles that do not concentrate around a single mean. A single Gaussian well per basin is misspecified for such manifolds. However, our evidence shows that enriching the basin model (per-basin mini-GMM, 0.746→0.749) yields negligible gain, which localizes the bottleneck: it is the *aggregation mechanism* (softmin over all-basin energies), not per-basin representational capacity, that fails. The correct remedy is to abandon global energy comparisons entirely in favor of *local, prototype-anchored* density estimates, as our Parzen readout does.

This finding has implications beyond our specific architecture. Energy-based models for classification that rely on a global softmin over many components should anticipate distance-concentration failure in dimensions exceeding ≈20–30. Local readouts (nearest-neighbor, kernel density, attention over a sparse subset) are a principled architectural alternative.

### 6.2 Why the DP/CRP criterion decisively beats a fixed threshold

The distance-threshold creation rule — "create a new basin if min_i ‖x−c_i‖ > τ" — appears in streaming k-means (Sculley, 2010), BIRCH (Zhang et al., 1996), and numerous online clustering algorithms. It has two structural deficiencies that the CRP criterion systematically addresses.

**1. Label contamination.** The threshold rule searches over *all* basins regardless of label. A point from a newly arriving class A can fall within τ of an existing basin for class B, silently mixing labels. This is not a rare corner case: in our 5-seed experiment, threshold k-means produces a per-class accuracy histogram with extreme bimodality — some classes reach >0.8 while others collapse to <0.05, indicating systematic label corruption in specific class regions. The CRP criterion prevents this by restricting the candidate set to B_y (same-label basins only) and using a universal rejection baseline s_0 = log α + log p_0(x) for genuinely new classes.

**2. Non-adaptive spatial scale.** A single τ cannot simultaneously be large enough to avoid fragmenting sparse classes into dozens of tiny basins and small enough to prevent merging distinct modes within dense classes. The CRP criterion replaces τ with a per-basin, per-dimension covariance Σ_i, so the acceptance region adapts to the local data geometry: a basin covering a tightly clustered mode has small variance and accepts points only in its immediate neighborhood; a basin covering a diffuse region has larger variance and a correspondingly wider acceptance region.

The quantitative evidence is stark. At matched prototype budget (≈30 basins), the Δ = 0.6960±0.0528 gap (5 seeds) is not merely large — the threshold variant's best single seed (0.3132) is below DP-ProtOn's worst (0.9144). In single-seed sweeps where we allow the threshold rule unlimited prototypes, it requires **266** prototypes to reach 0.900 ACC — over 8× the 32 prototypes DP-ProtOn uses. This is not a hyperparameter-tuning artifact; the failure is structural.

**The role of diagonal covariance.** We use diagonal Σ_i exclusively. Our single-seed ablation confirms: at 30-D, diagonal (0.746) vs full (0.680); at 64-D the full covariance estimate becomes ill-conditioned and the CRP criterion degrades to near-random allocation. Diagonal covariance is a deliberately conservative choice that prioritizes stability over asymptotic precision — a choice validated by the multi-seed results.

### 6.3 Editability as a first-class evaluation criterion

Most continual learning papers evaluate two metrics: final accuracy and forgetting (BWT). We argue that **editability** — the ability to surgically modify learned knowledge without retraining or degrading other knowledge — deserves equal standing as a third evaluation axis, for both practical and scientific reasons.

**Practical motivation.** Real-world deployed models face editing demands that retraining cannot satisfy. Under the European Union's General Data Protection Regulation (GDPR), individuals have a "right to be forgotten" (Article 17), which requires that a data subject's personal data — and any model trained on it — be purged. Retraining a production model from scratch is economically infeasible for many deployments. Similarly, content moderation systems must rapidly relabel categories (e.g., reclassifying a type of content from "allowed" to "prohibited") and inject new categories as novel abuse patterns emerge. In robotics, a deployed perception system that encounters a new object class during operation cannot pause for offline retraining. In all three scenarios, editability is not a convenience — it is a hard requirement.

**Scientific significance.** Editability is a direct diagnostic of representational modularity. A model that supports editing implies that its knowledge is stored in discrete, independently addressable units whose semantics are not entangled. Softmax classifiers store knowledge in a dense weight matrix W ∈ R^{C×d} where every row participates in every class decision; deleting one class requires recomputing the entire softmax normalization. DP-ProtOn, by contrast, stores knowledge as a set of labeled Gaussian attractors. The three editing operations we verify — delete, relabel, inject — are O(|B|) list operations rather than O(Cd) matrix operations. The fact that relabeling is byte-identical (remap agreement 1.0000±0.0000) confirms that the label is genuinely *detachable metadata*, not a parameter entangled in the decision function.

We encourage the CL community to adopt editability as a standard evaluation protocol. The three surgeries we propose — delete a class, relabel a class, inject a held-out class from N samples — form a minimal, reproducible editability benchmark that any prototype or exemplar-based method can report.

### 6.4 The role of frozen features and coupling to representation learning

Our deliberate choice of frozen PCA-whitened features is a methodological decision, not a limitation to be overcome. It serves two purposes. **First**, it isolates all observed behavior to the prototype layer, making every ablation cleanly attributable — when we vary the creation rule and observe a Δ=0.696 gap, we know the gap is due to the allocation mechanism, not to a confounding interaction with a learned encoder. **Second**, it establishes a lower bound: any method that learns features jointly with prototypes can only improve upon these numbers, so our results represent a conservative baseline.

Coupling DP-ProtOn to a learned encoder (e.g., a self-supervised ResNet or a progressively fine-tuned convnet) raises interesting questions that we leave to future work. Will the CRP criterion, which assumes approximately Gaussian per-class distributions in feature space, remain effective under a learned metric? Will the editing properties degrade if the encoder's weights shift when new classes are injected? The disentanglement between the encoder (continuous, gradient-based) and the prototype layer (discrete, set-based) may be the key to preserving editability while improving raw accuracy — a hypothesis we consider worth investigating.

### 6.5 Scope and honest accounting of negative results

We have been transparent about components that did *not* work: full covariance CRP (fails at 64-D), energy-softmin routing (net-negative at all dimensions tested), per-basin mini-GMM (negligible gain over single Gaussian), barrier terms and precision gating (designed for OOD detection; irrelevant to classification accuracy). Publishing negative results is essential for honest science, especially in a field where the gap between reported and actual performance is well-documented (Lipton & Steinhardt, 2019). DP-ProtOn is not the model we set out to build — it is the model that remained after we removed everything that did not survive empirical scrutiny. We believe this provenance makes it a more trustworthy baseline, not a weaker one.

---

## 7. Limitations

- **No feature learning.** Frozen PCA-whitening bounds performance on raw high-dimensional inputs. Coupling DP-ProtOn to a learned encoder is the natural next step, but it introduces a non-trivial tension: if the encoder's weights are updated on new tasks, the feature-space geometry shifts, and previously learned basins — whose centroids and covariances were estimated in the old embedding — may no longer be well-placed. Maintaining editability under a drifting representation is an open problem that we consider essential for future work.
- **No rehearsal.** No replay buffer means no re-clustering of old data — this is the source of the residual gap to Joint (~0.875 vs 0.969 on Split-MNIST) and offline $k$-means on synthetic data.
- **$\alpha$ is not data-adaptive.** The CRP concentration parameter $\alpha$ must currently be set to match the dataset's expected mode count (e.g., $\alpha{=}1.0$ for $\approx$10 classes, $\alpha{=}10^4$ for $\approx$30 modes). A data-driven adaptive $\alpha$ — for instance, via empirical Bayes marginal-likelihood maximization (Escobar & West, 1995) or online hyperparameter optimization — would substantially improve usability and is a natural extension.
- **Scope of evidence.** Three datasets in PCA-whitened spaces. No claim beyond frozen-feature, rehearsal-free continual classification.
- **No theoretical guarantees.** The CRP criterion is used as a heuristic allocation rule. Formal consistency guarantees or regret bounds for online prototype allocation under a DP prior remain open questions.

---

## 8. Conclusion

DP-ProtOn is what remained after we systematically falsified an energy-attractor design: a DP/CRP creation rule with diagonal covariance, one online denoised centroid per basin, and a Parzen / soft-$k$NN density readout. Every surviving component is justified by a controlled, budget-matched ablation. Every headline number carries multi-seed variance, with standard deviations that confirm stability rather than single-seed noise. The model is a reproducible, rehearsal-free continual-classification baseline that:

- Creates basins with 8× the prototype efficiency of the nearest alternative (Δ = 0.6960 ± 0.0528),
- Beats offline $k$-means at equal budget with a single online mean per class (Δ = 0.0340 ± 0.0053),
- Beats Experience Replay at equal memory on Split-MNIST by +0.20 ACC with near-zero forgetting (BWT = −0.0471),
- Supports training-free deletion, relabeling, and injection — operations that softmax classifiers cannot perform.

We offer DP-ProtOn as an honest, fully-ablated baseline for prototype-based continual learning and as concrete evidence that prototype attractors are directly editable knowledge units — a property that deserves far more attention in the CL literature than it has received.

---

## Reproducibility

All results are reproducible across runs. A per-instance RNG counter (`PWAFM._basin_seq`) replaces an earlier global class-variable seed (`Basin._next_id`) that caused cross-run drift. Code (MIT license): https://github.com/Fengrru/dp-proton. Core implementation: `pw_afm.py` (~340 lines, pure Python + numpy). Experiment drivers:

| File | Purpose | Output |
|---|---|---|
| `exp_control_kmeans.py` | DP vs threshold k-means control (creation-rule necessity test) | Table 5.2, Figure 1a |
| `exp_highdim.py` | Digits single-seed ablations (covariance, GMM, bandwidth, $\alpha$) | Section 5.6 |
| `exp_mnist_bench.py` | Split-MNIST CL baselines (EWC, Replay, FT, Joint) | Table 5.4, Figure 1c |
| `exp_edit.py` | Knowledge editing: delete / relabel / inject | Table 5.5, Figure 2 |
| `exp_multiseed.py` | Multi-seed driver producing all mean ± std numbers | Figure 1, aggregated tables |
| `gen_paper_figs.py` | Publication-quality figure generation | `results/fig_*.png` |
| `exp_synth.py` | Synthetic multi-modal stress test | Section 5.6 |
| `run_all.py` | Reproduction runner: reruns every experiment and checks each number against the golden reference, printing PASS/FAIL per section | all of the above |

Aggregated numbers are archived in `RESULTS_MULTISEED.md`. Figures: `results/fig_multiseed.png` (multi-seed validation, 3 panels), `results/fig_edit.png` (knowledge editing, 3 panels).

## References

References follow author–year style; in-text citations are to be wired to `\citep`/`\citet` in the LaTeX pass. Entries are grouped for readability but form a single alphabetical list at submission.

**Continual learning — regularization and rehearsal.**
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences (PNAS)*, 114(13), 3521–3526. (EWC)
- Chaudhry, A., Rohrbach, M., Elhoseiny, M., et al. (2019). On tiny episodic memories in continual learning. *arXiv:1902.10486*. (Experience Replay)
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 30. (GEM; BWT/forward-transfer metrics)
- Aljundi, R., Kelchtermans, K., & Tuytelaars, T. (2019). Task-free continual learning. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11254–11263.

**Prototype- and exemplar-based continual learning.**
- Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. *CVPR*, 2001–2010.
- Castro, F. M., Marín-Jiménez, M. J., Guil, N., Schmid, C., & Alahari, K. (2018). End-to-end incremental learning. *European Conference on Computer Vision (ECCV)*, 233–248.
- Wu, Y., Chen, Y., Wang, L., et al. (2019). Large scale incremental learning. *CVPR*, 374–382. (BiC)
- Mensink, T., Verbeek, J., Perronnin, F., & Csurka, G. (2013). Distance-based image classification: Generalizing to new classes at near-zero cost. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 35(11), 2624–2637. (Nearest-class-mean)

**Bayesian nonparametrics.**
- Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems. *The Annals of Statistics*, 1(2), 209–230. (Dirichlet Process)
- Blackwell, D., & MacQueen, J. B. (1973). Ferguson distributions via Pólya urn schemes. *The Annals of Statistics*, 1(2), 353–355. (CRP / Pólya urn)
- Teh, Y. W. (2010). Dirichlet process. In *Encyclopedia of Machine Learning*, 280–287. Springer.

**Online / streaming clustering.**
- Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH: An efficient data clustering method for very large databases. *ACM SIGMOD Record*, 25(2), 103–114.
- Cao, F., Ester, M., Qian, W., & Zhou, A. (2006). Density-based clustering over an evolving data stream with noise. *SIAM International Conference on Data Mining (SDM)*, 328–339. (DenStream)
- Sculley, D. (2010). Web-scale k-means clustering. *International Conference on World Wide Web (WWW)*, 1177–1178. (mini-batch / streaming k-means)

**Energy-based models and high-dimensional geometry.**
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. In *Predicting Structured Data*, MIT Press.
- Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? *International Conference on Database Theory (ICDT)*, 217–235. (distance concentration in high dimensions)

**High-dimensional geometry and research methodology.**
- Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). On the surprising behavior of distance metrics in high dimensional space. *International Conference on Database Theory (ICDT)*, 420–434.
- Lipton, Z. C., & Steinhardt, J. (2019). Troubling trends in machine learning scholarship. *Communications of the ACM*, 62(6), 45–53.

**Bayesian nonparametrics — extensions.**
- Escobar, M. D., & West, M. (1995). Bayesian density estimation and inference using mixtures. *Journal of the American Statistical Association*, 90(430), 577–588. (empirical Bayes for DP concentration)

**Data and tooling.**
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. (MNIST)
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825–2830. (digits, `make_classification`, PCA)
