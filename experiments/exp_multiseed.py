# -*- coding: utf-8 -*-
"""
多 seed 稳健性: 对三条核心结论各跑 N 个随机种子, 报告 mean±std。

为什么必须做: 单 seed 的漂亮数字在审稿人眼里等于噪声。方差(而非均值)决定
结论是否稳健 —— 这是把 TECH_REPORT 升级成可投稿论文的第一必需项。

三条被检验的结论 (与 TECH_REPORT 主张一一对应):
  A. 建盆准则消融   : 等原型数下 DP/CRP 建盆 是否稳定优于 距离阈值 k-means
  B. 原型源消融     : 在线去噪子质心(k=1) 是否稳定反超 离线 kmeans-30 与 1-NN 上限
  C. 持续学习基准   : Split-MNIST 上 DP-ProtOn 是否稳定夹在 Joint 与 Replay/EWC/Fine-tune 之间

产出: results/multiseed.png + 控制台 mean±std 汇总表 (可直接抄进论文表格)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from pw_afm import PWAFM
from exp_control_kmeans import build_digits_tasks, eval_dp, eval_kmeans
from exp_highdim import build_split_digits, eval_run
from exp_mnist_bench import (build_mnist_tasks, eval_dpproton, run_ewc,
                             run_replay, run_finetune, run_joint, N_CLASSES)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

SEEDS_DIGITS = [0, 1, 2, 3, 4]      # digits 便宜, 跑满 5 seed
SEEDS_MNIST = [0, 1, 2]             # MNIST 含 5 个 MLP 训练/method, 3 seed 已足够报方差


def ms(a):
    """mean±std 格式化"""
    a = np.asarray(a, dtype=float)
    return f'{a.mean():.4f}±{a.std():.4f}'


# ============================================================
# A. 建盆准则消融: DP vs 等原型数阈值 k-means (digits 30 维)
# ============================================================
def run_A(seeds):
    print('\n' + '=' * 68)
    print('  A. 建盆准则消融  DP/CRP vs 距离阈值 k-means  (digits 30维)')
    print('=' * 68)
    thresholds = np.arange(1.0, 8.5, 0.5)
    dp_acc, km_acc, dp_np, km_np = [], [], [], []
    for s in seeds:
        tasks = build_digits_tasks(n_comp=30, seed=s)
        a_dp, _, npro_dp, _ = eval_dp(tasks, dp_alpha=1.0, density_bw=1.0)
        # 逐 seed 重新匹配阈值 (原型数最接近 DP 者)
        best = None
        for t in thresholds:
            a_km, _, npro_km = eval_kmeans(tasks, threshold=t, density_bw=1.0)
            gap = abs(npro_km - npro_dp)
            if best is None or gap < best[0]:
                best = (gap, a_km, npro_km)
        dp_acc.append(a_dp); dp_np.append(npro_dp)
        km_acc.append(best[1]); km_np.append(best[2])
        print(f'  seed={s}: DP({npro_dp}原型) ACC={a_dp:.4f} | '
              f'kmeans({best[2]}原型) ACC={best[1]:.4f} | Δ={a_dp-best[1]:+.4f}')
    delta = np.array(dp_acc) - np.array(km_acc)
    print(f'\n  DP-ProtOn      ACC={ms(dp_acc)}  原型={ms(dp_np)}')
    print(f'  阈值 k-means   ACC={ms(km_acc)}  原型={ms(km_np)}')
    print(f'  ΔACC(DP-kmeans)= {ms(delta)}   (>0 且 std 不覆盖 0 => DP 稳定领先)')
    return dict(dp_acc=dp_acc, km_acc=km_acc, dp_np=dp_np, km_np=km_np)


# ============================================================
# B. 原型源消融: 在线去噪子质心 vs 离线 kmeans-30 vs 1-NN (digits 30 维)
# ============================================================
def _offline_baselines(tasks):
    """给定 tasks, 返回 (1-NN ACC, 离线每类3质心≈30 ACC)"""
    Xtr = np.vstack([t[0] for t in tasks]); ytr = np.concatenate([t[1] for t in tasks])
    Xte = np.vstack([t[2] for t in tasks]); yte = np.concatenate([t[3] for t in tasks])
    knn = KNeighborsClassifier(n_neighbors=1).fit(Xtr, ytr)
    acc_knn = (knn.predict(Xte) == yte).mean()
    protos, plabs = [], []
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]
        km = KMeans(n_clusters=min(3, len(Xc)), n_init=5, random_state=0).fit(Xc)
        protos.append(km.cluster_centers_); plabs += [c] * len(km.cluster_centers_)
    P = np.vstack(protos); plabs = np.array(plabs)
    d2 = ((Xte[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    acc_cent = (plabs[d2.argmin(1)] == yte).mean()
    return acc_knn, acc_cent


def run_B(seeds):
    print('\n' + '=' * 68)
    print('  B. 原型源消融  在线去噪子质心 vs 离线kmeans-30 vs 1-NN  (digits 30维)')
    print('=' * 68)
    sub_acc, raw_acc, knn_acc, cent_acc, sub_np = [], [], [], [], []
    for s in seeds:
        tasks, _ = build_split_digits(n_comp=30, seed=s)
        # 在线去噪子质心 k=1
        afm1 = PWAFM(T=0.3, sigma0=1.0, create_rule='dp', dp_alpha=1.0, base_scale=3.0,
                     dp_diag_cov=True, readout='density', density_bw=1.0, n_sub=1)
        a_sub, _, _ = eval_run(afm1, tasks)
        n_p = sum(len(b.sub_centroids()) for b in afm1.basins)
        # 原始噪声记忆 (n_sub=0)
        afm0 = PWAFM(T=0.3, sigma0=1.0, create_rule='dp', dp_alpha=1.0, base_scale=3.0,
                     dp_diag_cov=True, readout='density', density_bw=1.0, n_sub=0)
        a_raw, _, _ = eval_run(afm0, tasks)
        a_knn, a_cent = _offline_baselines(tasks)
        sub_acc.append(a_sub); raw_acc.append(a_raw); sub_np.append(n_p)
        knn_acc.append(a_knn); cent_acc.append(a_cent)
        print(f'  seed={s}: 子质心k=1({n_p}原型)={a_sub:.4f} | 原始记忆={a_raw:.4f} | '
              f'kmeans-30={a_cent:.4f} | 1-NN={a_knn:.4f}')
    print(f'\n  在线去噪子质心 k=1  ACC={ms(sub_acc)}  原型={ms(sub_np)}')
    print(f'  原始噪声记忆(蓄水池) ACC={ms(raw_acc)}')
    print(f'  离线 kmeans-30      ACC={ms(cent_acc)}')
    print(f'  1-NN (存全部)       ACC={ms(knn_acc)}')
    print(f'  Δ(子质心-kmeans30) = {ms(np.array(sub_acc)-np.array(cent_acc))}')
    return dict(sub=sub_acc, raw=raw_acc, knn=knn_acc, cent=cent_acc, sub_np=sub_np)


# ============================================================
# C. Split-MNIST 持续学习基准 (5 方法)
# ============================================================
def run_C(seeds):
    print('\n' + '=' * 68)
    print('  C. Split-MNIST 基准  DP-ProtOn vs EWC/Replay/Fine-tune/Joint')
    print('=' * 68)
    acc = {k: [] for k in ['DP-ProtOn', 'Replay', 'EWC', 'Fine-tune', 'Joint']}
    bwt = {k: [] for k in ['DP-ProtOn', 'Replay', 'EWC', 'Fine-tune']}
    for s in seeds:
        tasks, _ = build_mnist_tasks(n_comp=50, seed=s)
        a_dp, b_dp, _, np_ = eval_dpproton(tasks, dp_alpha=1.0, n_sub=1)
        mem_pc = max(1, np_ // N_CLASSES)
        a_rp, b_rp, _ = run_replay(tasks, mem_per_class=mem_pc)
        a_ew, b_ew = run_ewc(tasks, lambda_ewc=1000)
        a_ft, b_ft = run_finetune(tasks)
        a_jt = run_joint(tasks)
        acc['DP-ProtOn'].append(a_dp); bwt['DP-ProtOn'].append(b_dp)
        acc['Replay'].append(a_rp); bwt['Replay'].append(b_rp)
        acc['EWC'].append(a_ew); bwt['EWC'].append(b_ew)
        acc['Fine-tune'].append(a_ft); bwt['Fine-tune'].append(b_ft)
        acc['Joint'].append(a_jt)
        print(f'  seed={s}: DP={a_dp:.3f} Replay={a_rp:.3f} EWC={a_ew:.3f} '
              f'FT={a_ft:.3f} Joint={a_jt:.3f}')
    print('\n  方法            ACC(mean±std)        BWT(mean±std)')
    for k in ['Joint', 'DP-ProtOn', 'Replay', 'EWC', 'Fine-tune']:
        b_str = ms(bwt[k]) if k in bwt else '   --   '
        print(f'  {k:<14}{ms(acc[k]):<22}{b_str}')
    return dict(acc=acc, bwt=bwt)


# ============================================================
# 可视化: 三面板误差棒
# ============================================================
def plot_all(rA, rB, rC):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # A: DP vs kmeans (误差棒)
    ax = axes[0]
    labels = ['DP-ProtOn', '阈值k-means']
    means = [np.mean(rA['dp_acc']), np.mean(rA['km_acc'])]
    stds = [np.std(rA['dp_acc']), np.std(rA['km_acc'])]
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=['tab:green', 'tab:blue'])
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{m:.3f}', ha='center', fontsize=9)
    ax.set(ylabel='最终 ACC', ylim=(0, 1.05),
           title=f'A. 建盆准则 (digits, {len(rA["dp_acc"])} seeds)\n'
                 f'等原型数下 DP vs 距离阈值')

    # B: 原型源 (误差棒)
    ax = axes[1]
    keys = ['sub', 'raw', 'cent', 'knn']
    labs = ['子质心k=1', '原始记忆', 'kmeans-30', '1-NN(全存)']
    cols = ['tab:green', 'tab:red', 'purple', 'gray']
    m = [np.mean(rB[k]) for k in keys]; sd = [np.std(rB[k]) for k in keys]
    bars = ax.bar(labs, m, yerr=sd, capsize=6, color=cols)
    for bar, v in zip(bars, m):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.set(ylabel='最终 ACC', ylim=(0, 1.05),
           title=f'B. 原型源 (digits, {len(rB["sub"])} seeds)\n在线去噪 vs 离线上限')
    ax.tick_params(axis='x', rotation=15)

    # C: MNIST 5 方法 (误差棒)
    ax = axes[2]
    order = ['Joint', 'DP-ProtOn', 'Replay', 'EWC', 'Fine-tune']
    m = [np.mean(rC['acc'][k]) for k in order]
    sd = [np.std(rC['acc'][k]) for k in order]
    cols = ['gray', 'tab:green', 'tab:orange', 'tab:blue', 'tab:red']
    bars = ax.bar(order, m, yerr=sd, capsize=6, color=cols)
    for bar, v in zip(bars, m):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=8)
    ax.set(ylabel='最终 ACC', ylim=(0, 1.05),
           title=f'C. Split-MNIST ({len(rC["acc"]["Joint"])} seeds)\nDP-ProtOn vs 基准')
    ax.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    path = os.path.join(OUT, 'multiseed.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == '__main__':
    print('#' * 68)
    print('#  多 seed 稳健性验证 (mean±std)  —— 论文表格数据来源')
    print(f'#  digits seeds={SEEDS_DIGITS}   MNIST seeds={SEEDS_MNIST}')
    print('#' * 68)
    rA = run_A(SEEDS_DIGITS)
    rB = run_B(SEEDS_DIGITS)
    rC = run_C(SEEDS_MNIST)
    path = plot_all(rA, rB, rC)
    print(f'\n完成。三面板误差棒图已保存: {path}')
