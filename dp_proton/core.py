# -*- coding: utf-8 -*-
"""DP-ProtOn: Dirichlet-Process-guided Prototype Online learner.

Clean implementation of the algorithm described in the paper (Algorithm 1).
Exactly three components, nothing else:

  1. CRP basin creation (Dirichlet-process posterior predictive scores,
     diagonal covariance, data-driven base measure as rejection baseline);
  2. Online denoised centroids (competitive running mean, single pass);
  3. Parzen (soft-kNN) density readout over per-basin prototypes.

Plus the knowledge-editing interface (delete / relabel / inject a class
without any retraining), which is the paper's third contribution.

Design constraints honored here:
  - no gradients, no backprop, no global parameters;
  - every update is strictly local to one basin;
  - single pass over the stream (no rehearsal, no revisit);
  - frozen feature extractor assumed (this is a classifier head baseline).

NOTE: the exact numbers in the paper (RESULTS_MULTISEED.md) were produced
by experiments/pw_afm.py, the fully-ablated research implementation that
also contains the falsified legacy components kept for provenance. This
package is a clean re-implementation of the surviving recipe; the smoke
test (tests/smoke_test.py) verifies it reaches paper-level accuracy on the
same digits benchmark.
"""
import numpy as np


class Basin:
    """One class-conditional Gaussian basin = one explicitly locatable
    knowledge unit (can be deleted / relabeled / injected by reference)."""

    __slots__ = ('label', 'n', 'center', 'M2', 'subc', 'subc_n', 'n_sub', 'sigma0')

    def __init__(self, x, label, sigma0=1.0, n_sub=1):
        self.label = label
        self.n = 1
        self.center = np.array(x, dtype=float)
        self.M2 = np.zeros_like(x)                      # Welford M2 for diagonal variance
        self.sigma0 = float(sigma0)
        self.n_sub = int(n_sub)
        self.subc = [np.array(x, dtype=float)]          # online denoised centroids
        self.subc_n = [1]

    # ------------------------------------------------------------------
    # CRP scoring:  s_i(x) = log n_i + log N(x | c_i, Sigma_i)
    # ------------------------------------------------------------------
    def gauss_loglik(self, x):
        """log N(x | c_i, diag(Sigma_i)) with Welford-estimated variance."""
        d = len(self.center)
        diff = x - self.center
        var = np.clip(self.M2 / max(self.n - 1, 1), 1e-6, None)
        m2 = float(np.sum(diff * diff / var))
        logdet = float(np.sum(np.log(var)))
        return -0.5 * (m2 + d * np.log(2 * np.pi) + logdet)

    # ------------------------------------------------------------------
    # Online update (single-pass, local, no gradient)
    # ------------------------------------------------------------------
    def update(self, x):
        """Attach x to this basin: running-mean centroid (Welford),
        running diagonal variance, and competitive sub-centroid update."""
        self.n += 1
        delta = x - self.center
        self.center += delta / self.n
        self.M2 += delta * (x - self.center)
        self._update_subcentroids(x)

    def _update_subcentroids(self, x):
        """Online competitive learning: first n_sub points seed centroids,
        afterwards nearest-centroid online mean (denoising)."""
        if len(self.subc) < self.n_sub:
            self.subc.append(np.array(x, dtype=float))
            self.subc_n.append(1)
            return
        S = np.asarray(self.subc)
        j = int(((x - S) ** 2).sum(1).argmin())
        self.subc_n[j] += 1
        self.subc[j] += (x - self.subc[j]) / self.subc_n[j]

    def prototypes(self):
        """Denoised sub-centroids used by the density readout."""
        return np.asarray(self.subc)


class DPProtOn:
    """Dirichlet-Process-guided Prototype Online learner.

    Parameters
    ----------
    alpha : float, default=1.0
        DP concentration: smaller alpha = more reluctant to open a new basin.
    base_scale : float, default=3.0
        Width of the data-driven base measure G_0 relative to data variance.
    sigma0 : float, default=1.0
        Initial per-basin scale (prior variance and readout bandwidth anchor).
    density_bw : float, default=1.0
        Parzen kernel bandwidth h of the readout (h -> 0 approaches 1-NN).
    n_sub : int, default=1
        Number of online denoised sub-centroids per basin.
    """

    def __init__(self, alpha=1.0, base_scale=3.0, sigma0=1.0,
                 density_bw=1.0, n_sub=1):
        self.alpha = float(alpha)
        self.base_scale = float(base_scale)
        self.sigma0 = float(sigma0)
        self.density_bw = float(density_bw)
        self.n_sub = int(n_sub)
        self.basins = []
        # Online data statistics (Welford) for the adaptive base measure G_0
        self._n_seen = 0
        self._data_mean = None
        self._data_M2 = None

    # ------------------------------------------------------------------
    # Base measure: p_0(x) = N(x | data_mean, base_scale^2 * data_var)
    # ------------------------------------------------------------------
    def _update_data_stats(self, x):
        self._n_seen += 1
        if self._data_mean is None:
            self._data_mean = x.copy()
            self._data_M2 = np.zeros_like(x)
            return
        delta = x - self._data_mean
        self._data_mean += delta / self._n_seen
        self._data_M2 += delta * (x - self._data_mean)

    def _base_loglik(self, x):
        d = len(x)
        if self._n_seen < 2:
            var = (self.base_scale * self.sigma0) ** 2
        else:
            data_var = float(self._data_M2.mean() / (self._n_seen - 1))
            var = self.base_scale ** 2 * data_var + 1e-6
        m2 = float(np.sum((x - self._data_mean) ** 2) / var)
        return -0.5 * (m2 + d * np.log(2 * np.pi * var))

    # ------------------------------------------------------------------
    # Learning: the single-pass CRP loop
    # ------------------------------------------------------------------
    def _create(self, x, label):
        b = Basin(x, label, sigma0=self.sigma0, n_sub=self.n_sub)
        self.basins.append(b)
        return b

    def observe(self, x, y):
        """Process one sample (x, y).

        Decision rule (paper, Eq. 4):
          join basin  : s_i(x) = log n_i + log N(x | c_i, Sigma_i)
          reject      : s_0(x) = log alpha + log p_0(x)
          if s_0(x) > max_i s_i(x) over same-label basins -> open a new basin.
        """
        x = np.asarray(x, dtype=float)
        self._update_data_stats(x)
        if not self.basins:
            self._create(x, y)
            return 'create'

        same = [b for b in self.basins if b.label == y]
        if not same:
            self._create(x, y)                      # first example of a new class
            return 'create'

        s_best, b_best = max(
            (np.log(b.n) + b.gauss_loglik(x), b) for b in same)
        s_new = np.log(self.alpha) + self._base_loglik(x)
        if s_best > s_new:                          # explained -> local update
            b_best.update(x)
            return 'update'
        self._create(x, y)                          # rejected -> new basin
        return 'create'

    def fit_stream(self, X, y):
        """Single-pass learning over a stream. Returns per-sample events."""
        return [self.observe(xi, yi) for xi, yi in zip(X, y)]

    # ------------------------------------------------------------------
    # Readout: Parzen (soft-kNN) density per basin, aggregated by label
    # ------------------------------------------------------------------
    def _proto_logits(self, X):
        """s_i(x) = logsumexp_k(-||x - m_k||^2 / (2 h^2)) over basin prototypes."""
        X = np.atleast_2d(X)
        h2 = self.density_bw ** 2
        out = []
        for b in self.basins:
            M = b.prototypes()
            d2 = ((X[:, None, :] - M[None, :, :]) ** 2).sum(-1)
            a = -0.5 * d2 / h2
            amax = a.max(1)
            out.append(amax + np.log(np.exp(a - amax[:, None]).sum(1)))
        return np.stack(out) if out else np.zeros((0, len(X)))

    def label_scores(self, X):
        """Per-label density scores (log-domain), for calibration/analysis."""
        X = np.atleast_2d(X)
        logits = self._proto_logits(X)
        labels = sorted({b.label for b in self.basins})
        scores = np.full((len(labels), len(X)), -np.inf)
        for i, lab in enumerate(labels):
            rows = [j for j, b in enumerate(self.basins) if b.label == lab]
            if rows:
                amax = logits[rows].max(0)
                scores[i] = amax + np.log(
                    np.exp(logits[rows] - amax).sum(0))
        return labels, scores

    def classify(self, X):
        """Predict labels: argmax over per-label Parzen density."""
        labels, scores = self.label_scores(X)
        return np.array([labels[k] for k in scores.argmax(0)])

    def predict(self, X):
        """Alias of classify (scikit-learn convention)."""
        return self.classify(X)

    # ------------------------------------------------------------------
    # Novelty score (for detection after deletion):
    # novelty(x) = -max_i log N(x | c_i, Sigma_i), higher = more unknown.
    # ------------------------------------------------------------------
    def novelty(self, X):
        X = np.atleast_2d(X)
        if not self.basins:
            return np.full(len(X), np.inf)
        # explicit loop keeps memory flat
        out = np.empty(len(X))
        for i, x in enumerate(X):
            out[i] = -max(b.gauss_loglik(x) for b in self.basins)
        return out

    # ------------------------------------------------------------------
    # Knowledge editing: zero retraining, by design
    # ------------------------------------------------------------------
    def delete_class(self, label):
        """Remove a class entirely: drop all basins of that label."""
        n0 = len(self.basins)
        self.basins = [b for b in self.basins if b.label != label]
        return n0 - len(self.basins)

    def relabel_class(self, label_from, label_to):
        """Change a class's semantics in place (all basins relabeled)."""
        n = 0
        for b in self.basins:
            if b.label == label_from:
                b.label = label_to
                n += 1
        return n

    def inject_class(self, X, y):
        """Inject a new (or resurrected) class from a handful of examples;
        identical to fit_stream -- the learning rule is the same."""
        return self.fit_stream(X, y)

    # ------------------------------------------------------------------
    def summary(self):
        from collections import Counter
        c = Counter(b.label for b in self.basins)
        return {
            'n_basins': len(self.basins),
            'per_label': dict(c),
            'n_prototypes': sum(len(b.subc) for b in self.basins),
        }
