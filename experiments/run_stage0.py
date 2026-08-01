# -*- coding: utf-8 -*-
"""
阶段0实验：低维合成景观 —— PW-AFM 机制存在性证明 (对应 all.md 实验 0.1 / 0.2 / 0.3)
产出: results/ 下的图表 + 控制台指标汇总
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pw_afm import PWAFM

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 数据
def make_task_clusters():
    """5 个顺序任务 A-E, 每任务一个 2D 高斯簇"""
    centers = np.array([[-4, 0], [0, 4], [4, 0], [0, -4], [-3, -3.5]], float)
    covs = [np.diag([0.35, 0.35])] * 5
    tasks = []
    for k, (c, S) in enumerate(zip(centers, covs)):
        Xtr = RNG.multivariate_normal(c, S, 200)
        Xte = RNG.multivariate_normal(c, S, 200)
        tasks.append((Xtr, np.full(200, k), Xte, np.full(200, k)))
    return tasks


# ---------------------------------------------------------------- 基线1: 在线 GMM (全局参数 + 增量EM -> 会漂移遗忘)
class OnlineGMM:
    def __init__(self, K=5, d=2, lr=0.05, seed=0):
        r = np.random.default_rng(seed)
        self.mu = r.normal(0, 1, (K, d))
        self.var = np.ones(K)
        self.pi = np.ones(K) / K
        self.lr = lr
        self.comp_label_score = {}  # (comp -> {label: soft count})

    def _resp(self, X):
        d2 = ((X[:, None, :] - self.mu[None]) ** 2).sum(-1)
        logp = -0.5 * d2 / self.var - X.shape[1] / 2 * np.log(self.var) + np.log(self.pi + 1e-12)
        logp -= logp.max(1, keepdims=True)
        R = np.exp(logp)
        return R / R.sum(1, keepdims=True)

    def partial_fit(self, X, y):
        for xi, yi in zip(X, y):
            r = self._resp(xi[None])[0]
            k = int(r.argmax())
            # 在线EM: 责任加权地把分量拉向当前数据 (全局共享参数 -> 旧簇被覆盖)
            self.mu += self.lr * r[:, None] * (xi - self.mu)
            self.var += self.lr * r * (((xi - self.mu) ** 2).sum(1) / 2 - self.var)
            self.var = np.clip(self.var, 0.05, None)
            self.pi = (1 - self.lr) * self.pi + self.lr * r
            self.comp_label_score.setdefault(k, {})
            self.comp_label_score[k][yi] = self.comp_label_score[k].get(yi, 0) * 0.999 + 1

    def predict(self, X):
        R = self._resp(X)
        comp2lab = {k: max(v, key=v.get) for k, v in self.comp_label_score.items()}
        return np.array([comp2lab.get(int(k), -1) for k in R.argmax(1)])


# ---------------------------------------------------------------- 基线2: 朴素SGD softmax (神经网络式全局梯度 -> 灾难性遗忘)
class NaiveSGD:
    def __init__(self, d=2, C=5, lr=0.1, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, 0.01, (C, d))
        self.b = np.zeros(C)
        self.lr = lr

    def partial_fit(self, X, y, epochs=5):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                z = self.W @ xi + self.b
                z -= z.max()
                p = np.exp(z) / np.exp(z).sum()
                p[yi] -= 1
                self.W -= self.lr * np.outer(p, xi)   # 全局参数更新 = 覆盖旧知识
                self.b -= self.lr * p

    def predict(self, X):
        return (X @ self.W.T + self.b).argmax(1)


# ================================================================ 实验 0.1
def exp01():
    print('\n========== 实验 0.1 顺序学习无遗忘 (PW-AFM vs 在线GMM vs 朴素SGD) ==========')
    tasks = make_task_clusters()
    afm = PWAFM(T=0.2, tau_novel=0.4, sigma0=0.5)
    gmm = OnlineGMM(K=5)
    sgd = NaiveSGD()
    names = ['PW-AFM', '在线GMM(增量EM)', '朴素SGD(下限)']
    acc_hist = {n: np.full((5, 5), np.nan) for n in names}  # [学完任务t, 测任务k]

    for t, (Xtr, ytr, _, _) in enumerate(tasks):
        idx = RNG.permutation(len(Xtr))
        afm.fit_stream(Xtr[idx], ytr[idx])
        gmm.partial_fit(Xtr[idx], ytr[idx])
        sgd.partial_fit(Xtr[idx], ytr[idx])
        for k in range(t + 1):
            Xte, yte = tasks[k][2], tasks[k][3]
            acc_hist['PW-AFM'][t, k] = (afm.classify(Xte) == yte).mean()
            acc_hist['在线GMM(增量EM)'][t, k] = (gmm.predict(Xte) == yte).mean()
            acc_hist['朴素SGD(下限)'][t, k] = (sgd.predict(Xte) == yte).mean()

    # 指标: 最终平均准确率 ACC 与后向迁移 BWT
    print(f"{'模型':<18}{'最终ACC':>10}{'BWT':>10}")
    for n in names:
        A = acc_hist[n]
        acc = np.nanmean(A[4])
        bwt = np.mean([A[4, k] - A[k, k] for k in range(4)])
        print(f'{n:<18}{acc:>10.3f}{bwt:>+10.3f}')
    print('PW-AFM 盆地状态:', afm.summary())

    # 图: 任务A准确率随训练推进的曲线 + 最终各任务准确率
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, style in zip(names, ['o-', 's--', '^:']):
        axes[0].plot(range(1, 6), [acc_hist[n][t, 0] for t in range(5)], style, label=n)
    axes[0].set(xlabel='已学任务数', ylabel='任务A 测试准确率', title='旧知识保持曲线 (任务A)',
                ylim=(-0.05, 1.05), xticks=range(1, 6))
    axes[0].axhline(0.95, color='gray', ls=':', lw=1)
    axes[0].legend()
    x = np.arange(5)
    for i, n in enumerate(names):
        axes[1].bar(x + (i - 1) * 0.26, acc_hist[n][4], 0.26, label=n)
    axes[1].set(xlabel='任务', ylabel='最终准确率', title='学完全部5任务后的各任务准确率',
                xticks=x, xticklabels=list('ABCDE'), ylim=(0, 1.05))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'exp01_no_forgetting.png'), dpi=150)
    plt.close(fig)
    return acc_hist


# ================================================================ 实验 0.2
def exp02():
    print('\n========== 实验 0.2 精度门控的必要性 (重叠簇 + 消融) ==========')
    # 两个重叠簇: 形状不同 (各向异性 vs 各向同性), 中心接近 -> 单纯距离易误判
    c0, c1 = np.array([-0.8, 0.0]), np.array([0.8, 0.0])
    S0 = np.array([[1.6, 0.0], [0.0, 0.08]])   # 扁平横条
    S1 = np.array([[0.12, 0.0], [0.0, 1.6]])   # 竖直长条
    X0 = RNG.multivariate_normal(c0, S0, 300); X1 = RNG.multivariate_normal(c1, S1, 300)
    X = np.vstack([X0, X1]); y = np.array([0] * 300 + [1] * 300)
    idx = RNG.permutation(600)
    Xte0 = RNG.multivariate_normal(c0, S0, 300); Xte1 = RNG.multivariate_normal(c1, S1, 300)
    Xte = np.vstack([Xte0, Xte1]); yte = np.array([0] * 300 + [1] * 300)
    overlap = (np.abs(Xte[:, 0]) < 1.2) & (np.abs(Xte[:, 1]) < 1.2)  # 重叠区

    results = {}
    models = {}
    for gate in (True, False):
        # sigma0 较大使距离能量项变"钝"(仅看远近不看结构), omega 较高使精度门控敏锐:
        # 从而分离"距离"与"结构可重构性"两种信号, 检验门控贡献
        m = PWAFM(T=0.2, tau_novel=0.4, sigma0=0.9, omega=18.0, use_precision_gate=gate)
        m.fit_stream(X[idx], y[idx])
        pred = m.classify(Xte)
        results[gate] = ((pred == yte).mean(), (pred[overlap] == yte[overlap]).mean())
        models[gate] = m
    print(f"{'配置':<22}{'总体ACC':>10}{'重叠区ACC':>12}")
    print(f"{'PW-AFM(精度门控)':<22}{results[True][0]:>10.3f}{results[True][1]:>12.3f}")
    print(f"{'纯AFM(均匀softmin)':<22}{results[False][0]:>10.3f}{results[False][1]:>12.3f}")

    # 决策边界可视化
    gx, gy = np.meshgrid(np.linspace(-4, 4, 220), np.linspace(-3.2, 3.2, 180))
    G = np.c_[gx.ravel(), gy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, gate, title in zip(axes, [False, True],
                               ['纯 AFM (无精度门控)', 'PW-AFM (精度门控)']):
        Z = models[gate].classify(G).reshape(gx.shape)
        ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=['#aec7e8', '#ffbb78'], alpha=0.7)
        ax.scatter(*X0.T, s=6, c='tab:blue', label='簇0 (横向)')
        ax.scatter(*X1.T, s=6, c='tab:orange', label='簇1 (纵向)')
        acc, oacc = results[gate]
        ax.set_title(f'{title}\n总体ACC={acc:.3f}  重叠区ACC={oacc:.3f}')
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'exp02_precision_gate.png'), dpi=150)
    plt.close(fig)
    return results


# ================================================================ 实验 0.3
def _run_conflict_once(seed):
    """单次矛盾实验: 返回 (afm, 事件列表, 动力学落入A的比例)"""
    rng = np.random.default_rng(seed)
    c = np.array([0.0, 0.0])
    afm = PWAFM(T=0.15, tau_novel=0.4, sigma0=0.45, barrier_height=1.4)
    events = []
    labs = ['concept-A'] * 150 + ['concept-B'] * 150
    rng.shuffle(labs)
    for lab in labs:
        offset = np.array([-0.55, 0]) if lab == 'concept-A' else np.array([0.55, 0])
        x = c + offset + rng.normal(0, 0.28, 2)
        events.append(afm.observe(x, lab))
    # 以真实对称中心采样, 沿能量景观梯度下降(Langevin 零温极限), 看收敛到哪个吸引子
    P0 = c + rng.normal(0, 0.15, (400, 2))
    h, step = 1e-4, 0.04
    for _ in range(200):
        gxn = (afm.global_energy(P0 + [h, 0]) - afm.global_energy(P0 - [h, 0])) / (2 * h)
        gyn = (afm.global_energy(P0 + [0, h]) - afm.global_energy(P0 - [0, h])) / (2 * h)
        g = np.c_[gxn, gyn]
        g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        P0 -= step * g
    centers = np.stack([b.center for b in afm.basins])
    blabels = np.array([b.label for b in afm.basins])
    near = ((P0[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
    pA = (blabels[near] == 'concept-A').mean()
    return afm, events, pA


def exp03():
    print('\n========== 实验 0.3 壁垒与矛盾概念共存 ==========')
    # 同一位置附近, 两个标签交替出现 ("猫有四条腿" vs "艺术猫有三条腿")
    # 单次景观天然随机不对称 -> 10 种子重复, 验证无系统性偏置
    pAs = []
    for seed in range(10):
        afm, events, pA = _run_conflict_once(seed)
        pAs.append(pA)
    pAs = np.array(pAs)
    print(f'10种子动力学落入比例: P(A) = {pAs.mean():.3f} ± {pAs.std():.3f}  (预期均值≈0.5)')
    print(f'各种子 P(A): {np.round(pAs, 2).tolist()}')

    # 取一个典型种子做可视化与结构指标
    afm, events, _ = _run_conflict_once(0)
    s = afm.summary()
    print('典型种子事件统计:', {e: events.count(e) for e in set(events)})
    print('典型种子盆地汇总:', s)
    A = [b for b in afm.basins if b.label == 'concept-A']
    B = [b for b in afm.basins if b.label == 'concept-B']
    ba = max(A, key=lambda b: b.w); bb = max(B, key=lambda b: b.w)
    dist = np.linalg.norm(ba.center - bb.center)
    print(f'主盆地中心距 = {dist:.3f}, 壁垒对数 = {s["n_conflict_barriers"]}')
    mid = np.array([0.0, 0.0])

    # 能量景观可视化: 双坑 + 中间壁垒脊
    gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 240), np.linspace(-1.8, 1.8, 200))
    G = np.c_[gx.ravel(), gy.ravel()]
    E = afm.global_energy(G).reshape(gx.shape)
    xs = np.linspace(-2.2, 2.2, 400)
    E1d = afm.global_energy(np.c_[xs, np.full(400, mid[1])])

    fig = plt.figure(figsize=(13, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.contourf(gx, gy, E, levels=30, cmap='viridis')
    ax1.plot(*ba.center, 'r*', ms=14, label='盆地A中心')
    ax1.plot(*bb.center, 'w*', ms=14, label='盆地B中心')
    fig.colorbar(im, ax=ax1, label='E(x)')
    ax1.set_title('全局能量景观 (双坑结构)'); ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(gx, gy, E, cmap='viridis', rstride=4, cstride=4, linewidth=0)
    ax2.set_title('能量曲面 3D')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(xs, E1d, lw=2)
    ax3.axvline(ba.center[0], color='r', ls='--', lw=1, label='A中心')
    ax3.axvline(bb.center[0], color='orange', ls='--', lw=1, label='B中心')
    ax3.axvline(mid[0], color='gray', ls=':', lw=1, label='壁垒中点')
    ax3.set(xlabel='x (过两中心连线)', ylabel='E(x)', title='一维剖面: 双坑 + 中间壁垒脊')
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'exp03_barrier_landscape.png'), dpi=150)
    plt.close(fig)
    return dist, s['n_conflict_barriers'], pAs.mean(), pAs.std()


# ================================================================ 附加: 知识编辑速验 (阶段3预演)
def exp_edit():
    print('\n========== 附加实验: 显式知识删除 (阶段3预演) ==========')
    tasks = make_task_clusters()
    afm = PWAFM(T=0.2, tau_novel=0.4, sigma0=0.5)
    for Xtr, ytr, _, _ in tasks:
        idx = RNG.permutation(len(Xtr))
        afm.fit_stream(Xtr[idx], ytr[idx])
    before = {k: (afm.classify(tasks[k][2]) == tasks[k][3]).mean() for k in range(5)}
    nov_before = afm.novelty(tasks[2][2]).mean()
    afm.delete_concept(2)   # 删除概念"任务C"
    after = {k: (afm.classify(tasks[k][2]) == tasks[k][3]).mean() for k in range(5) if k != 2}
    nov_after = afm.novelty(tasks[2][2]).mean()
    print(f'删除前各任务ACC: { {k: round(v,3) for k,v in before.items()} }')
    print(f'删除任务C盆地后, 其余任务ACC: { {k: round(v,3) for k,v in after.items()} }')
    print(f'任务C样本平均新奇分数: 删除前 {nov_before:.3f} -> 删除后 {nov_after:.3f} (应显著升高, 被识别为未知)')


if __name__ == '__main__':
    exp01()
    exp02()
    exp03()
    exp_edit()
    print(f'\n全部实验完成, 图表已保存至: {OUT}')
