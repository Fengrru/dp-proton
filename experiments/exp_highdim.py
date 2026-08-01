# -*- coding: utf-8 -*-
"""
创新点① 高维验证: 在真实 64 维手写数字 (sklearn digits) 上检验 DP 建盆准则是否仍成立。
诚实设计:
  - 原始 64 维直接跑 -> 暴露 all.md 承认的"维度灾难"(协方差奇异/距离退化)。
  - PCA 白化降维(无监督, 一次拟合后冻结, 不随任务更新 => 零遗忘风险) -> 文档自己开的
    "冻结特征提取器"药方的最温和版本; 在此空间公平比较 threshold vs DP 建盆准则。
  - 任务划分: Split-digits, 5 任务 x 2 类 ({0,1}..{8,9}) 顺序学习, 学完不重访旧数据。
产出: results/highdim_dp.png + 控制台指标
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from pw_afm import PWAFM

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)


def build_split_digits(n_comp=30, seed=0):
    """返回 (tasks, dim). tasks[k] = (Xtr,ytr,Xte,yte). 若 n_comp is None 则用原始64维。"""
    d = load_digits()
    X, y = d.data, d.target
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # 30% 作为"预训练池": 仅用于拟合 PCA/Scaler(表征学习一次性冻结), 绝不参与盆地评估
    n_pre = int(0.3 * len(X))
    Xpre = X[:n_pre]
    Xrest, yrest = X[n_pre:], y[n_pre:]

    scaler = StandardScaler().fit(Xpre)
    if n_comp is None:
        proj = lambda Z: scaler.transform(Z)   # 仅标准化, 保留 64 维
        dim = 64
    else:
        pca = PCA(n_components=n_comp, whiten=True, random_state=seed).fit(scaler.transform(Xpre))
        proj = lambda Z: pca.transform(scaler.transform(Z))
        dim = n_comp

    Xr = proj(Xrest)
    tasks = []
    for k in range(5):
        cls = (2 * k, 2 * k + 1)
        m = np.isin(yrest, cls)
        Xk, yk = Xr[m], yrest[m]
        r = np.random.default_rng(100 + k)
        p = r.permutation(len(Xk))
        Xk, yk = Xk[p], yk[p]
        split = int(0.7 * len(Xk))
        tasks.append((Xk[:split], yk[:split], Xk[split:], yk[split:]))
    return tasks, dim


def eval_run(afm, tasks):
    A = np.full((5, 5), np.nan)
    for t, (Xtr, ytr, _, _) in enumerate(tasks):
        afm.fit_stream(Xtr, ytr)
        for k in range(t + 1):
            A[t, k] = (afm.classify(tasks[k][2]) == tasks[k][3]).mean()
    acc = np.nanmean(A[4])
    bwt = np.mean([A[4, k] - A[k, k] for k in range(4)])
    return acc, bwt, len(afm.basins)


def run_space(dim_label, n_comp, sigma0):
    tasks, dim = build_split_digits(n_comp=n_comp)
    print(f'\n----- 空间: {dim_label} (维度={dim}) -----')
    at, bt, nt = eval_run(
        PWAFM(T=0.3, sigma0=sigma0, create_rule='threshold', tau_novel=0.4), tasks)
    ad, bd, nd = eval_run(
        PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0, base_scale=3.0), tasks)
    ag, bg, ng = eval_run(
        PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0, base_scale=3.0,
              dp_diag_cov=True), tasks)
    am, bm, nm = eval_run(
        PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0, base_scale=3.0,
              dp_diag_cov=True, basin_model='mixture', proto_scale=sigma0), tasks)
    ar, br, nr = eval_run(
        PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0, base_scale=3.0,
              dp_diag_cov=True, readout='density', density_bw=sigma0), tasks)
    print(f"{'准则':<22}{'最终ACC':>10}{'BWT':>10}{'盆地数':>10}  (真实类数=10)")
    print(f"{'硬阈值':<22}{at:>10.3f}{bt:>+10.3f}{nt:>10}")
    print(f"{'DP(满协方差,单高斯)':<22}{ad:>10.3f}{bd:>+10.3f}{nd:>10}")
    print(f"{'DP(对角协方差,单高斯)':<22}{ag:>10.3f}{bg:>+10.3f}{ng:>10}")
    print(f"{'DP(对角协方差,mini-GMM)':<22}{am:>10.3f}{bm:>+10.3f}{nm:>10}")
    print(f"{'DP(对角+density读出)':<22}{ar:>10.3f}{br:>+10.3f}{nr:>10}")
    return dict(dim=dim, thr=(at, nt), dp=(ad, nd), dpg=(ag, ng), dpm=(am, nm), dpd=(ar, nr))


def alpha_sensitivity(n_comp=30, sigma0=1.0):
    tasks, _ = build_split_digits(n_comp=n_comp)
    alphas = [1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    out = []
    for a in alphas:
        acc, _, n = eval_run(PWAFM(T=0.3, sigma0=sigma0, create_rule='dp',
                                   dp_alpha=a, base_scale=3.0, dp_diag_cov=True), tasks)
        out.append((a, n, acc))
    print('\n----- 30维空间 DP-α 敏感性 (对角协方差) -----')
    for a, n, acc in out:
        print(f'  α={a:<8} 盆地={n:<4} ACC={acc:.3f}')
    return out


def bw_sensitivity(n_comp=30, sigma0=1.0):
    """density 读出对核带宽 h 的敏感性 (h->0 退化为盆内 1-NN)"""
    tasks, _ = build_split_digits(n_comp=n_comp)
    print('\n----- 30维 density 读出: 核带宽 h 敏感性 -----')
    for h in [0.3, 0.5, 1.0, 1.5, 2.0]:
        acc, bwt, n = eval_run(PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0,
                                     base_scale=3.0, dp_diag_cov=True,
                                     readout='density', density_bw=h), tasks)
        print(f'  h={h:<5} 盆地={n:<4} ACC={acc:.3f} BWT={bwt:+.3f}')


def controls(n_comp=30):
    """两个参照系: 把 PW-AFM 夹在中间, 回答'精度来自样例记忆还是吸引子机制'
    - 1-NN: 存下全部训练点最近邻 (硬阈值 PW-AFM 的极限: 盆地≈样本数)
    - 30质心: 每类 3 个 kmeans 质心共~30, 最近质心 (DP 压缩到~30单元的极限)"""
    tasks, _ = build_split_digits(n_comp=n_comp)
    Xtr = np.vstack([t[0] for t in tasks]); ytr = np.concatenate([t[1] for t in tasks])
    Xte = np.vstack([t[2] for t in tasks]); yte = np.concatenate([t[3] for t in tasks])
    knn = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ytr)
    acc_knn = (knn.predict(Xte) == yte).mean()
    # 每类 3 质心
    protos, plabs = [], []
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]
        km = KMeans(n_clusters=min(3, len(Xc)), n_init=5, random_state=0).fit(Xc)
        protos.append(km.cluster_centers_); plabs += [c] * len(km.cluster_centers_)
    P = np.vstack(protos); plabs = np.array(plabs)
    d2 = ((Xte[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    acc_cent = (plabs[d2.argmin(1)] == yte).mean()
    print('\n----- 参照系 (30维白化空间) -----')
    print(f'  1-NN (存全部~{len(Xtr)}样本)      ACC={acc_knn:.3f}   <- 硬阈值 PW-AFM 上限')
    print(f'  30质心 (每类3个,共{len(P)})     ACC={acc_cent:.3f}   <- 压缩到~30单元上限')
    return acc_knn, acc_cent


def diagnose_routing(n_comp=30, sigma0=1.0):
    """病灶定位: 同一个训练好的 DP 模型, 对比三种读出方式
    - 完整路由: PW-AFM 能量/精度/温度投票
    - 最近中心: 绕开路由, 直接取最近盆地 center 的标签 (测'质心放的好不好')
    - 最近原型: 取所有盆地记忆原型中最近邻的标签 (盆地内部样例 1-NN)
    """
    tasks, _ = build_split_digits(n_comp=n_comp)
    afm = PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=1.0, base_scale=3.0,
                dp_diag_cov=True, basin_model='mixture', proto_scale=sigma0)
    for Xtr, ytr, _, _ in tasks:
        afm.fit_stream(Xtr, ytr)
    Xte = np.vstack([t[2] for t in tasks]); yte = np.concatenate([t[3] for t in tasks])
    acc_route = (afm.classify(Xte) == yte).mean()
    # 最近盆地中心
    C = np.array([b.center for b in afm.basins]); clab = np.array([b.label for b in afm.basins])
    d2c = ((Xte[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    acc_center = (clab[d2c.argmin(1)] == yte).mean()
    # 最近记忆原型
    protos, plab = [], []
    for b in afm.basins:
        M = np.asarray(b.memory); protos.append(M); plab += [b.label] * len(M)
    P = np.vstack(protos); plab = np.array(plab)
    d2p = ((Xte[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    acc_proto = (plab[d2p.argmin(1)] == yte).mean()
    print('\n----- 病灶定位: 同一 DP 模型三种读出 (30维) -----')
    print(f'  完整能量路由           ACC={acc_route:.3f}')
    print(f'  最近盆地中心(绕开路由)   ACC={acc_center:.3f}   <- 高则路由是废物')
    print(f'  最近记忆原型({len(P)}个)      ACC={acc_proto:.3f}   <- 高则盆内样例足够')
    return acc_route, acc_center, acc_proto


def subcentroid_experiment(n_comp=30, sigma0=1.0, dp_alpha=1.0):
    """digits 交叉验证: 盆地数匹配下, 原始记忆 vs 在线去噪子质心"""
    tasks, _ = build_split_digits(n_comp=n_comp)
    print(f'\n----- 在线去噪子质心: digits(α={dp_alpha}) density原型源消融 -----')
    print(f"{'原型源':<26}{'总原型数':>10}{'ACC':>10}{'BWT':>10}{'盆地数':>8}")
    results = {}
    for n_sub in [0, 1, 3, 5, 10]:
        afm = PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=dp_alpha,
                    base_scale=3.0, dp_diag_cov=True, readout='density',
                    density_bw=sigma0, n_sub=n_sub)
        acc, bwt, nb = eval_run(afm, tasks)
        n_proto = sum(len(b.sub_centroids()) for b in afm.basins)
        tag = '原始噪声记忆(蓄水池)' if n_sub == 0 else f'在线去噪子质心 k={n_sub}'
        results[n_sub] = (acc, bwt, nb, n_proto)
        print(f'  {tag:<24}{n_proto:>10}{acc:>10.3f}{bwt:>+10.3f}{nb:>8}')
    return results


if __name__ == '__main__':
    print('========== PW-AFM 高维验证: DP vs 硬阈值建盆 (真实64维手写数字) ==========')
    raw = run_space('原始64维 (无降维, 暴露维度灾难)', n_comp=None, sigma0=3.0)
    p30 = run_space('PCA白化30维 (冻结特征, 文档药方)', n_comp=30, sigma0=1.0)
    ctrl = controls(n_comp=30)
    diagnose_routing(n_comp=30, sigma0=1.0)
    bw_sensitivity(n_comp=30, sigma0=1.0)
    sens = alpha_sensitivity(n_comp=30, sigma0=1.0)

    # 头条: 在线去噪子质心 digits 验证
    sub_res = subcentroid_experiment(n_comp=30, sigma0=1.0, dp_alpha=1.0)

    # 可视化: 三组准则 x 两空间 + sub-centroid 面板
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    groups = ['原始64维', 'PCA白化30维']
    x = np.arange(2)
    series = [
        ('硬阈值(≈全存)', 'thr', 'tab:red', -0.3),
        ('DP单高斯', 'dpg', 'tab:blue', -0.1),
        ('DP+mini-GMM', 'dpm', 'tab:orange', 0.1),
        ('DP+density读出', 'dpd', 'tab:green', 0.3),
    ]
    for name, key, color, off in series:
        ns = [raw[key][1], p30[key][1]]
        accs = [raw[key][0], p30[key][0]]
        bars = axes[0].bar(x + off, ns, 0.2, color=color, label=name)
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{acc:.2f}', ha='center', va='bottom', fontsize=6.5)
    axes[0].axhline(10, color='gray', ls='--', lw=1, label='真实类数=10')
    axes[0].set(xticks=x, xticklabels=groups, ylabel='最终盆地数 (柱顶数字=ACC)',
                title='density 读出: 绕开有害能量路由, 紧凑盆地下回押精度')
    axes[0].legend(fontsize=8)

    aa = [s[0] for s in sens]; nn = [s[1] for s in sens]; ac = [s[2] for s in sens]
    ax2 = axes[1]; ax3 = ax2.twinx()
    ax2.semilogx(aa, nn, 's-', color='tab:blue', label='盆地数')
    ax2.axhline(10, color='gray', ls='--', lw=1)
    ax3.semilogx(aa, ac, 'o--', color='tab:green', label='ACC')
    ax2.set(xlabel='DP α (对数轴)', ylabel='盆地数', title='30维(对角协方差): DP-α 敏感性')
    ax3.set_ylabel('ACC', color='tab:green'); ax3.set_ylim(0, 1.05)
    ax2.legend(loc='upper left', fontsize=8); ax3.legend(loc='lower right', fontsize=8)

    # 右面板: sub-centroid 结果
    ks = sorted(sub_res.keys())
    accs = [sub_res[k][0] for k in ks]; nps = [sub_res[k][3] for k in ks]
    xlab = ['原始' if k == 0 else f'k={k}' for k in ks]
    ax4 = axes[2]
    ax4.plot(range(len(ks)), accs, 'o-', color='tab:green', lw=2, ms=7)
    for i, (k, a, np_) in enumerate(zip(ks, accs, nps)):
        ax4.annotate(f'{a:.3f}\n({np_}原型)', (i, a), textcoords='offset points',
                     xytext=(0, 12), fontsize=7, ha='center')
    ax4.axhline(ctrl[0], color='gray', ls='--', lw=1.2, label=f'1-NN={ctrl[0]:.3f}')
    ax4.axhline(ctrl[1], color='purple', ls=':', lw=1.2, label=f'kmeans-30={ctrl[1]:.3f}')
    ax4.axhline(sub_res[0][0], color='tab:red', ls=':', lw=1, alpha=0.6)
    ax4.set_xticks(range(len(ks))); ax4.set_xticklabels(xlab)
    ax4.set(ylabel='最终 ACC', xlabel='density 原型源 (盆地固定≈{})'.format(p30['dpg'][1]),
            title='digits: 在线去噪子质心反超离线kmeans上限')
    ax4.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'highdim_dp.png'), dpi=150)
    plt.close(fig)
    print(f'\n完成, 图表已保存至: {OUT}')
