# -*- coding: utf-8 -*-
"""
横向复现: 在"高维 + 多峰类"合成基准上, 检验 "DP 建盆 + density 读出" 不是 digits 专属运气。
- 数据: make_classification, 50 维全信息, 10 类, 每类 3 子簇 (真实模式=30, 本质多峰非高斯) -> 单高斯的天敌。
- 管线与 digits 实验一致: 30% 预训练池拟合 PCA 白化(冻结->零遗忘, 全信息维故为无害旋转); 5 任务 x 2 类顺序学习。
- 参照系: 1-NN(全存, 上界~0.94) / 30 质心 k-means(压缩到真实模式数的上界)。
产出: results/synth_dp_density.png + 控制台指标
结论对照 digits: 若 density 读出同样把 DP 从"能量读出的低分"拉到接近 30 质心上限, 则普适性成立。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from pw_afm import PWAFM

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

N_CLASSES = 10
N_PER_TASK = 2          # 每任务类数 -> 5 个任务
N_CLUSTERS = 3          # 每类子簇数
N_MODES = N_CLASSES * N_CLUSTERS  # 真实模式数 = 30


def build_tasks(n_comp=50, seed=0):
    """返回 (tasks, dim). tasks[k]=(Xtr,ytr,Xte,yte). 50维全信息->PCA白化(无害旋转)。"""
    X, y = make_classification(
        n_samples=6000, n_features=50, n_informative=50, n_redundant=0, n_repeated=0,
        n_classes=N_CLASSES, n_clusters_per_class=N_CLUSTERS, class_sep=2.5,
        flip_y=0.01, random_state=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]

    n_pre = int(0.3 * len(X))                    # 预训练池: 仅拟合 PCA/Scaler, 不参与评估
    Xpre = X[:n_pre]
    Xrest, yrest = X[n_pre:], y[n_pre:]
    scaler = StandardScaler().fit(Xpre)
    pca = PCA(n_components=n_comp, whiten=True, random_state=seed).fit(scaler.transform(Xpre))
    proj = lambda Z: pca.transform(scaler.transform(Z))
    Xr = proj(Xrest)

    tasks = []
    for k in range(N_CLASSES // N_PER_TASK):
        cls = list(range(k * N_PER_TASK, (k + 1) * N_PER_TASK))
        m = np.isin(yrest, cls)
        Xk, yk = Xr[m], yrest[m]
        r = np.random.default_rng(100 + k)
        p = r.permutation(len(Xk)); Xk, yk = Xk[p], yk[p]
        s = int(0.7 * len(Xk))
        tasks.append((Xk[:s], yk[:s], Xk[s:], yk[s:]))
    return tasks, n_comp


def eval_run(afm, tasks):
    n = len(tasks)
    A = np.full((n, n), np.nan)
    for t, (Xtr, ytr, _, _) in enumerate(tasks):
        afm.fit_stream(Xtr, ytr)
        for k in range(t + 1):
            A[t, k] = (afm.classify(tasks[k][2]) == tasks[k][3]).mean()
    acc = np.nanmean(A[n - 1])
    bwt = np.mean([A[n - 1, k] - A[k, k] for k in range(n - 1)])
    return acc, bwt, len(afm.basins)


def run_compare(sigma0=1.0):
    tasks, dim = build_tasks(n_comp=50)
    print(f'\n----- 高维多峰合成: 50维全信息->PCA白化{dim}维, {N_CLASSES}类x{N_CLUSTERS}子簇 (真实模式={N_MODES}) -----')
    rows = [
        ('硬阈值(≈全存)', PWAFM(T=0.3, sigma0=sigma0, create_rule='threshold', tau_novel=0.4)),
        ('DP+能量读出(单高斯)', PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0,
                               base_scale=3.0, dp_diag_cov=True)),
        ('DP+density读出', PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0,
                              base_scale=3.0, dp_diag_cov=True, readout='density', density_bw=sigma0)),
    ]
    res = {}
    print(f"{'方案':<22}{'最终ACC':>10}{'BWT':>10}{'盆地数':>10}")
    for name, afm in rows:
        acc, bwt, nb = eval_run(afm, tasks)
        res[name] = (acc, bwt, nb)
        print(f"{name:<22}{acc:>10.3f}{bwt:>+10.3f}{nb:>10}")
    return res


def controls():
    tasks, _ = build_tasks(n_comp=50)
    Xtr = np.vstack([t[0] for t in tasks]); ytr = np.concatenate([t[1] for t in tasks])
    Xte = np.vstack([t[2] for t in tasks]); yte = np.concatenate([t[3] for t in tasks])
    acc_knn = (KNeighborsClassifier(n_neighbors=1).fit(Xtr, ytr).predict(Xte) == yte).mean()
    protos, plab = [], []
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]
        km = KMeans(n_clusters=min(N_CLUSTERS, len(Xc)), n_init=5, random_state=0).fit(Xc)
        protos.append(km.cluster_centers_); plab += [c] * len(km.cluster_centers_)
    P = np.vstack(protos); plab = np.array(plab)
    acc_cent = (plab[((Xte[:, None, :] - P[None, :, :]) ** 2).sum(-1).argmin(1)] == yte).mean()
    print('\n----- 参照系 -----')
    print(f'  1-NN (存全部~{len(Xtr)}样本)   ACC={acc_knn:.3f}   <- 精度上界')
    print(f'  {len(P)}质心 (每类{N_CLUSTERS}子簇)      ACC={acc_cent:.3f}   <- 压到真实模式数上界')
    return acc_knn, acc_cent


def bw_sensitivity(sigma0=1.0):
    tasks, _ = build_tasks(n_comp=50)
    print('\n----- density 读出核带宽 h 敏感性 -----')
    for h in [0.3, 0.5, 1.0, 1.5, 2.0]:
        acc, bwt, nb = eval_run(PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0,
                                      base_scale=3.0, dp_diag_cov=True,
                                      readout='density', density_bw=h), tasks)
        print(f'  h={h:<5} 盆地={nb:<4} ACC={acc:.3f} BWT={bwt:+.3f}')


def alpha_sensitivity_density(sigma0=1.0):
    """验证欠分割假设: 调大 α -> 盆地数逗近真实模式数 30 -> density 精度应爬向 0.9"""
    tasks, _ = build_tasks(n_comp=50)
    print(f'\n----- density 读出: DP-α -> 盆地数 -> 精度 (真实模式={N_MODES}) -----')
    for a in [1.0, 1e2, 1e4, 1e6, 1e8, 1e10]:
        acc, bwt, nb = eval_run(PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=a,
                                      base_scale=3.0, dp_diag_cov=True,
                                      readout='density', density_bw=1.5), tasks)
        print(f'  α={a:<8.0e} 盆地={nb:<4} ACC={acc:.3f} BWT={bwt:+.3f}')


def subcentroid_experiment(sigma0=1.0, dp_alpha=1e4):
    """决定性实验: 盆地数匹配(~30≨30真实模式)时, 原始噪声记忆 vs 在线去噪子质心。
    假设: 离线 kmeans-30(0.94) 赢在 '去噪质心' 而非 '多存样本'; 若在线子质心能补上大半差距, 则在线方法可逼近离线聚类。"""
    tasks, _ = build_tasks(n_comp=50)
    print(f'\n----- 在线去噪子质心: 盆地数匹配下(α={dp_alpha:.0e}) density原型源消融 -----')
    print(f"{'原型源':<26}{'总原型数':>10}{'ACC':>10}{'BWT':>10}{'盆地数':>8}")
    results = {}
    for n_sub in [0, 1, 3, 5, 10]:
        afm = PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=dp_alpha,
                    base_scale=3.0, dp_diag_cov=True, readout='density',
                    density_bw=1.5, n_sub=n_sub)
        acc, bwt, nb = eval_run(afm, tasks)
        n_proto = sum(len(b.sub_centroids()) for b in afm.basins)
        tag = f'原始噪声记忆(蓄水池)' if n_sub == 0 else f'在线去噪子质心 k={n_sub}'
        results[n_sub] = (acc, bwt, nb, n_proto)
        print(f'  {tag:<24}{n_proto:>10}{acc:>10.3f}{bwt:>+10.3f}{nb:>8}')
    return results


if __name__ == '__main__':
    print('========== 横向复现: DP+density 在高维多峰合成数据上的普适性 ==========')
    res = run_compare(sigma0=1.0)
    knn, cent = controls()
    bw_sensitivity(sigma0=1.0)
    alpha_sensitivity_density(sigma0=1.0)
    sub_res = subcentroid_experiment(sigma0=1.0, dp_alpha=1e4)

    # 可视化: 左=三方案盆地数/ACC; 右=盆地数匹配下 n_sub 扫描 (头条结果)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = list(res.keys())
    nb = [res[k][2] for k in names]
    acc = [res[k][0] for k in names]
    colors = ['tab:red', 'tab:blue', 'tab:green']
    bars = ax.bar(names, nb, color=colors)
    for bar, a in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'ACC={a:.3f}', ha='center', va='bottom', fontsize=9)
    ax.axhline(N_MODES, color='gray', ls='--', lw=1, label=f'真实模式数={N_MODES}')
    ax.axhline(N_CLASSES, color='purple', ls=':', lw=1, label=f'真实类数={N_CLASSES}')
    ax.set(ylabel='最终盆地数 (柱顶=ACC)',
           title=f'高维多峰({N_CLASSES}类x2子簇): DP+density 复现 digits 上的结论')
    ax.text(0.98, 0.02, f'参照系: 1-NN={knn:.3f} | {N_MODES}质心={cent:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', fc='lightyellow'))
    ax.legend(fontsize=9)

    # 右图: 盆地数固定(~30)时, ACC vs 在线子质心数 k
    ks = sorted(sub_res.keys())
    accs = [sub_res[k][0] for k in ks]
    xlab = ['原始记忆' if k == 0 else f'k={k}' for k in ks]
    ax2.plot(range(len(ks)), accs, 'o-', color='tab:green', lw=2, ms=7)
    for i, (k, a) in enumerate(zip(ks, accs)):
        ax2.annotate(f'{a:.3f}', (i, a), textcoords='offset points', xytext=(0, 8), fontsize=9)
    ax2.axhline(cent, color='gray', ls='--', lw=1.2, label=f'离线 kmeans-{N_MODES}={cent:.3f} (上限)')
    ax2.axhline(sub_res[0][0], color='tab:red', ls=':', lw=1.2, label=f'原始噪声记忆={sub_res[0][0]:.3f}')
    ax2.set_xticks(range(len(ks))); ax2.set_xticklabels(xlab)
    ax2.set(ylabel='最终 ACC', xlabel='density 原型源 (盆地数固定≈{})'.format(sub_res[ks[0]][2]),
            title='在线去噪子质心: 填平在线↔离线差距的~72%')
    ax2.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'synth_dp_density.png'), dpi=150)
    plt.close(fig)
    print(f'\n完成, 图表已保存至: {OUT}')
