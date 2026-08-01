# -*- coding: utf-8 -*-
"""
对照实验：DP 建盆 vs 阈值在线 k-means。

设计目标：唯一变量 = 开新原型/盆地的条件（DP/CRP 准则 vs 距离阈值）。
- 两个方法共享相同的 density 读出引擎（_proto_logits 逻辑）和在线均值更新。
- 总原型数预算匹配（调整阈值 τ 使在线 k-means 的最终原型数与 DP 方案尽可能接近）。
- 数据：digits（30 维 PCA 白化）和合成集（50 维 PCA 白化）。

如果 DP 和阈值 k-means 精度持平 → DP 不是必要组件。
如果 DP 明显更优 → DP 是贡献，有辩护空间。
如果 DP 更差 → 放弃 DP，方法核心改为在线竞争学习。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Part 1: 阈值在线 k-means（与 DP 共享 density 读出设计）
# ============================================================
class ThresholdOnlineKMeans:
    """纯在线阈值 k-means：先分后学。
    
    唯一建盆条件：距离最近质心是否超过阈值 τ。
    与 DP 共享：在线均值子质心更新 + density 软 kNN 读出。
    """
    def __init__(self, threshold=3.0, readout='density', density_bw=1.0):
        self.threshold = threshold          # 开新质心的距离阈值
        self.readout = readout              # 和 DP 一样: 'density' 或 'energy'
        self.density_bw = density_bw
        self.centroids = []                 # list[array(d,)]
        self.counts = []                    # 每个质心的归属计数（在线均值）
        self.labels = []                    # 每个质心的标签
        self._rng = np.random.default_rng(0)
    
    def fit_stream(self, X, y):
        for xi, yi in zip(X, y):
            self._observe(xi, yi)
    
    def _observe(self, x, yi):
        if len(self.centroids) == 0:
            self.centroids.append(np.array(x, dtype=float))
            self.counts.append(1)
            self.labels.append(yi)
            return
        
        S = np.asarray(self.centroids)
        d2 = ((x - S) ** 2).sum(-1)
        j = int(d2.argmin())
        
        if d2[j] >= self.threshold ** 2:
            # 开新原型
            self.centroids.append(np.array(x, dtype=float))
            self.counts.append(1)
            self.labels.append(yi)
        else:
            # 在线均值更新（与 Basin._update_subcentroids 相同逻辑）
            self.counts[j] += 1
            self.centroids[j] += (x - self.centroids[j]) / self.counts[j]
            # 标签采用多数投票：如果新标签与原标签不同，记录但不强制翻转
            if yi != self.labels[j]:
                pass  # 保留原标签（第一个标签作为默认）
    
    def classify(self, X):
        X = np.atleast_2d(X)
        if len(self.centroids) == 0:
            return np.zeros(len(X), dtype=int)
        
        C = np.asarray(self.centroids)
        if self.readout == 'density':
            # 每个质心单独算软得分: s_j(x) = exp(-0.5 * ||x - c_j||² / h²)
            # 然后用 label 聚合: 每个预测点的最近质心标签
            h2 = self.density_bw ** 2
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # (n_test, n_centroids)
            a = -0.5 * d2 / h2
            # 每个测试点取最高得分质心的标签
            pred = np.array([self.labels[j] for j in a.argmax(1)])
            return pred
        else:
            # 最近质心（等效于 h->0 的极限）
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            return np.array([self.labels[j] for j in d2.argmin(1)])


# ============================================================
# Part 2: 数据构建（复用 exp_highdim.py 和 exp_synth.py 设计）
# ============================================================
def build_digits_tasks(n_comp=30, seed=0):
    """5 tasks × 2 classes, PCA 白化后返回"""
    d = load_digits()
    X, y = d.data, d.target
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_pre = int(0.3 * len(X))
    Xpre = X[:n_pre]
    Xrest, yrest = X[n_pre:], y[n_pre:]
    scaler = StandardScaler().fit(Xpre)
    if n_comp is None:
        proj = lambda Z: scaler.transform(Z)
    else:
        pca = PCA(n_components=n_comp, whiten=True, random_state=seed).fit(scaler.transform(Xpre))
        proj = lambda Z: pca.transform(scaler.transform(Z))
    Xr = proj(Xrest)
    tasks = []
    for k in range(5):
        cls = (2 * k, 2 * k + 1)
        m = np.isin(yrest, cls)
        Xk, yk = Xr[m], yrest[m]
        r = np.random.default_rng(100 + k)
        p = r.permutation(len(Xk)); Xk, yk = Xk[p], yk[p]
        s = int(0.7 * len(Xk))
        tasks.append((Xk[:s], yk[:s], Xk[s:], yk[s:]))
    return tasks


def build_synth_tasks(n_comp=50, seed=0):
    """5 tasks × 2 classes, 10 classes × 3 clusters, make_classification"""
    X, y = make_classification(
        n_samples=6000, n_features=50, n_informative=50, n_redundant=0, n_repeated=0,
        n_classes=10, n_clusters_per_class=3, class_sep=2.5,
        flip_y=0.01, random_state=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    n_pre = int(0.3 * len(X))
    Xpre = X[:n_pre]; Xrest, yrest = X[n_pre:], y[n_pre:]
    scaler = StandardScaler().fit(Xpre)
    pca = PCA(n_components=n_comp, whiten=True, random_state=seed).fit(scaler.transform(Xpre))
    proj = lambda Z: pca.transform(scaler.transform(Z))
    Xr = proj(Xrest)
    tasks = []
    for k in range(5):
        cls = list(range(k * 2, (k + 1) * 2))
        m = np.isin(yrest, cls)
        Xk, yk = Xr[m], yrest[m]
        r = np.random.default_rng(100 + k)
        p = r.permutation(len(Xk)); Xk, yk = Xk[p], yk[p]
        s = int(0.7 * len(Xk))
        tasks.append((Xk[:s], yk[:s], Xk[s:], yk[s:]))
    return tasks


# ============================================================
# Part 3: DP 建盆（从 pw_afm 导入，统一 n_sub=1 配置）
# ============================================================
from pw_afm import PWAFM


def eval_dp(tasks, dp_alpha=1.0, sigma0=1.0, density_bw=1.0):
    """DP + n_sub=1（去噪子质心）+ density 读出"""
    afm = PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=dp_alpha,
                base_scale=3.0, dp_diag_cov=True, readout='density',
                density_bw=density_bw, n_sub=1)
    for Xtr, ytr, _, _ in tasks:
        afm.fit_stream(Xtr, ytr)
    A = np.full((len(tasks), len(tasks)), np.nan)
    for t in range(len(tasks)):
        for k in range(t + 1):
            A[t, k] = (afm.classify(tasks[k][2]) == tasks[k][3]).mean()
    acc = np.nanmean(A[-1])
    n_prev = len(tasks) - 1
    bwt = np.mean([A[-1, k] - A[k, k] for k in range(n_prev)]) if n_prev > 0 else 0.0
    n_proto = sum(len(b.sub_centroids()) for b in afm.basins)
    return acc, bwt, n_proto, len(afm.basins)


def eval_kmeans(tasks, threshold, density_bw=1.0):
    """阈值在线 k-means + density 读出"""
    km = ThresholdOnlineKMeans(threshold=threshold, readout='density', density_bw=density_bw)
    for Xtr, ytr, _, _ in tasks:
        km.fit_stream(Xtr, ytr)
    Xte = np.vstack([t[2] for t in tasks]); yte = np.concatenate([t[3] for t in tasks])
    pred = km.classify(Xte)
    acc = (pred == yte).mean()
    return acc, km.threshold, len(km.centroids)


# ============================================================
# Part 4: 阈值扫描 + 对比
# ============================================================
def find_matching_kmeans(tasks, target_protos, density_bw=1.0, threshold_range=None):
    """扫描阈值，找与 DP 原型数最匹配的配置"""
    if threshold_range is None:
        threshold_range = np.arange(1.0, 8.5, 0.5)
    best = (0, float('inf'), 0, 0.0)
    for t in threshold_range:
        acc, th, np_ = eval_kmeans(tasks, threshold=t, density_bw=density_bw)
        gap = abs(np_ - target_protos)
        if gap < best[1]:
            best = (t, gap, np_, acc)
    return best[0], best[2], best[3]


def run_comparison(name, tasks, dp_alpha, density_bw, threshold_range):
    print(f'\n===== {name} =====')
    # DP baseline
    acc_dp, bwt_dp, npro_dp, nbasin_dp = eval_dp(tasks, dp_alpha=dp_alpha, density_bw=density_bw)
    print(f'DP(α={dp_alpha:.0e}): 盆地={nbasin_dp} 原型={npro_dp} ACC={acc_dp:.4f} BWT={bwt_dp:+.4f}')
    
    # 阈值扫描 - 热身（找匹配配置）
    acc_kms, npro_kms = [], []
    # 合成集需要更大阈值（50维距离尺度大）
    if '合成' in name:
        thresholds = list(np.arange(1.0, 8.5, 0.5)) + list(np.arange(8.5, 25.5, 0.5))
    else:
        thresholds = np.arange(1.0, 8.5, 0.5)
    print(f'\n  阈值扫描 (target ~{npro_dp} 原型):')
    print(f"  {'阈值tau':>8}{'原型数':>10}{'ACC':>10}")
    for t in thresholds:
        acc, _, np_ = eval_kmeans(tasks, threshold=t, density_bw=density_bw)
        acc_kms.append(acc); npro_kms.append(np_)
        print(f'  {t:>8.1f}{np_:>10}{acc:>10.4f}')
    
    # 找最匹配（按原型数接近度排序）
    best_gap = float('inf')
    best_idx = 0
    for i, np_ in enumerate(npro_kms):
        gap = abs(np_ - npro_dp)
        if gap < best_gap:
            best_gap = gap
            best_idx = i
    best_t = thresholds[best_idx]
    best_np = npro_kms[best_idx]
    best_acc = acc_kms[best_idx]
    print(f'\n  最匹配 DP: tau={best_t:.1f} → {best_np}原型 (DP={npro_dp}) ACC={best_acc:.4f}')
    
    return dict(name=name, dp_acc=acc_dp, dp_bwt=bwt_dp, dp_np=npro_dp, dp_nb=nbasin_dp,
                km_thresh=best_t, km_np=best_np, km_acc=best_acc,
                all_t=list(thresholds), all_acc=acc_kms, all_np=npro_kms)


if __name__ == '__main__':
    print('=' * 65)
    print('  对照实验: DP 建盆 vs 阈值在线 k-means')
    print('  (等原型数情况下, DP比简单阈值好多少?)')
    print('=' * 65)
    
    # ---------- Digits (30维, α=1.0) ----------
    tasks_d = build_digits_tasks(n_comp=30)
    res_d = run_comparison('Digits 30维 PCA白化', tasks_d, dp_alpha=1.0,
                           density_bw=1.0, threshold_range=np.arange(1.0, 8.5, 0.5))
    
    # ---------- 合成集 (50维, α=1e4) ----------
    tasks_s = build_synth_tasks(n_comp=50)
    res_s = run_comparison('合成集 50维 PCA白化', tasks_s, dp_alpha=1e4,
                           density_bw=1.5, threshold_range=np.arange(1.0, 8.5, 0.5))
    
    # ---------- 汇总 ----------
    print('\n' + '=' * 65)
    print('  对照实验汇总: 等原型数下的精度差异')
    print('=' * 65)
    for r in [res_d, res_s]:
        gap = r['dp_acc'] - r['km_acc']
        verdict = 'DP 领先' if gap > 0.01 else ('阈值 k-means 领先' if gap < -0.01 else '持平')
        print(f"\n{r['name']}:")
        print(f"  DP ({r['dp_np']}原型)    ACC={r['dp_acc']:.4f}  BWT={r['dp_bwt']:+.4f}")
        print(f"  阈值k-means ({r['km_np']}原型) ACC={r['km_acc']:.4f}  τ={r['km_thresh']:.1f}")
        print(f"  ΔACC(DP - kmeans) = {gap:+.4f}  → {verdict}")
    
    # ---------- 可视化 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (r, ax) in enumerate(zip([res_d, res_s], axes)):
        ts = r['all_t']
        ax.plot(ts, r['all_acc'], 'o-', color='tab:blue', lw=2, ms=6, label='阈值k-means ACC')
        ax.axhline(r['dp_acc'], color='tab:green', ls='--', lw=1.5, label=f"DP ACC={r['dp_acc']:.3f}")
        ax.axvline(r['km_thresh'], color='gray', ls=':', alpha=0.7)
        ax.set(xlabel='阈值 τ', ylabel='最终 ACC',
               title=f"{r['name']}\nDP({r['dp_np']}原型) vs 阈值k-means(等原型数≈{r['dp_np']})")
        ax.legend(fontsize=8)
        # 标注原型数
        ax2 = ax.twinx()
        ax2.plot(ts, r['all_np'], 's--', color='tab:red', ms=4, alpha=0.5, label='原型数')
        ax2.set_ylabel('原型数', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='lower right', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'control_dp_vs_kmeans.png'), dpi=150)
    plt.close(fig)
    print(f'\n图表已保存至: {OUT}')
