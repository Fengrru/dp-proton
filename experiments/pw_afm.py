# -*- coding: utf-8 -*-
"""
PW-AFM: Precision-Weighted Attractor Field Machine —— 阶段0 低维原型实现
严格对应 all.md 中的数学定义：
  - 盆地能量:   E_i(x) = -w_i * exp(-0.5 * ||x-c_i||^2_{Sigma_i}) + B_i(x)   (径向基势阱 + 壁垒)
  - 全局能量:   E(x)  = -T * log( sum_i exp(-E_i(x)/T) )                     (softmin 概率混合)
  - 精度门控:   pi_i(x) = exp(-0.5 * eps^T Omega eps),  eps = x - g_i(x)     (生成器重建误差)
  - 生成器 g_i: 基于记忆样本集的无参数生成器 (最近邻原型), 完全局部存储
  - 新奇分数:   Novelty(x) = min_i [ 0.5*eps^T Omega eps + E_i(x) ]          (加权预测误差)
  - 深度更新:   w_i <- w_i + lr * pi_i(x)  (访问即强化, 精度加权);  未访问盆地指数衰减 (可逆遗忘)
  - 矛盾壁垒:   同位置不同标签 -> 创建新盆地并在两盆地间建立能量壁垒 B (双坑共存)
无全局参数、无反向传播、所有更新严格局部于单个盆地。
"""
import numpy as np


class Basin:
    """一个吸引盆地 = 一个可显式定位/编辑/删除的知识单元"""
    _next_id = 0

    def __init__(self, x, label, w0=1.0, sigma0=0.5, mem_cap=25,
                 model='gauss', proto_scale=1.0, rng_seed=0, n_sub=0):
        self.id = Basin._next_id
        Basin._next_id += 1
        d = len(x)
        self.center = np.array(x, dtype=float)      # c_i 吸引中心 (概念原型, 可读出)
        self.cov = np.eye(d) * sigma0 ** 2          # Sigma_i 形状张量 (泛化边界)
        self.cov_inv = np.linalg.inv(self.cov)
        self.w = float(w0)                          # w_i 深度 (记忆强度, 可衰减可恢复)
        self.label = label
        self.memory = [np.array(x, dtype=float)]    # 生成器 g_i 的记忆样本集 (无可训练权重)
        self.mem_cap = mem_cap
        self.model = model                          # 'gauss'(单高斯) 或 'mixture'(多原型 mini-GMM)
        self.proto_var = proto_scale ** 2           # mixture 模式下各原型势阱的带宽
        self.n_sub = n_sub                          # 每盆地在线去噪子质心数 (0=关闭, 用原始记忆)
        self.subc = [np.array(x, dtype=float)]      # 子质心 (在线竞争学习/均值去噪)
        self.subc_n = [1]                           # 各子质心归属计数 (供在线均值)
        self.barriers = []                          # [(midpoint, height, sigma_b)] 与矛盾盆地间的壁垒
        self.access = 1
        self._rng = np.random.default_rng(rng_seed)   # 局部播种, 不依赖全局创建顺序 (可复现)

    # ---- 能量项 ----
    def maha2(self, X):
        diff = np.atleast_2d(X) - self.center
        return np.einsum('ij,jk,ik->i', diff, self.cov_inv, diff)

    def barrier_term(self, X):
        """B_i(x): 壁垒在两矛盾盆地中点处抬升能量, 形成'护城河'"""
        X = np.atleast_2d(X)
        b = np.zeros(len(X))
        for mid, h, sb in self.barriers:
            d2 = np.sum((X - mid) ** 2, axis=1)
            b += h * np.exp(-0.5 * d2 / sb ** 2)
        return b

    def _proto_phi(self, X):
        """mixture 模式: phi(x) = max_k exp(-0.5||x-m_k||^2 / proto_var), 到最近记忆原型的势阱深度"""
        X = np.atleast_2d(X)
        M = np.asarray(self.memory)
        d2 = ((X[:, None, :] - M[None, :, :]) ** 2).sum(-1) / self.proto_var
        return np.exp(-0.5 * d2).max(1)

    def energy(self, X):
        """E_i(x) = -w_i * phi(x) + B_i(x)
        gauss:   phi = exp(-0.5 * 马氏距离^2)  (单高斯势阱)
        mixture: phi = 到最近记忆原型的势阱 (多峰, 适配高维非高斯流形)"""
        phi = self._proto_phi(X) if self.model == 'mixture' else np.exp(-0.5 * self.maha2(X))
        return -self.w * phi + self.barrier_term(X)

    # ---- 生成器与精度 ----
    def reconstruct(self, X):
        """g_i: 无参数生成器 —— 返回记忆样本集中的最近邻原型 x_hat"""
        X = np.atleast_2d(X)
        M = np.asarray(self.memory)
        d2 = ((X[:, None, :] - M[None, :, :]) ** 2).sum(-1)
        return M[np.argmin(d2, axis=1)]

    def precision(self, X, omega):
        """pi_i(x) = exp(-0.5 * ||x - x_hat||^2 * Omega)"""
        eps2 = np.sum((np.atleast_2d(X) - self.reconstruct(X)) ** 2, axis=1)
        return np.exp(-0.5 * eps2 * omega)

    def recon_err2(self, X, omega):
        """加权重建误差 0.5 * eps^T Omega eps"""
        eps2 = np.sum((np.atleast_2d(X) - self.reconstruct(X)) ** 2, axis=1)
        return 0.5 * eps2 * omega

    def gauss_loglik(self, x, diag=False):
        """log N(x | c_i, Sigma_i): 该盆地对 x 的生成似然, 供 DP 准则使用
        diag=True 时用对角协方差 (高维下更稳健: 少量样本难估计满秩协方差)"""
        d = len(self.center)
        diff = x - self.center
        if diag:
            var = np.clip(np.diag(self.cov), 1e-6, None)
            m2 = float(np.sum(diff * diff / var))
            logdet = float(np.sum(np.log(var)))
        else:
            _, logdet = np.linalg.slogdet(self.cov)
            m2 = float(self.maha2(x[None, :])[0])
        return -0.5 * (m2 + d * np.log(2 * np.pi) + logdet)

    # ---- 局部更新 (仅 Hebbian/EMA, 无梯度) ----
    def _update_subcentroids(self, x):
        """在线竞争学习 (在线 k-means): 前 n_sub 个点各自成质心, 之后就近均值去噪"""
        if self.n_sub <= 0:
            return
        if len(self.subc) < self.n_sub:
            self.subc.append(np.array(x, dtype=float))
            self.subc_n.append(1)
            return
        S = np.asarray(self.subc)
        j = int(((x - S) ** 2).sum(1).argmin())
        self.subc_n[j] += 1
        self.subc[j] += (x - self.subc[j]) / self.subc_n[j]   # 在线均值: 质心为归属样本均值(去噪)

    def sub_centroids(self):
        """去噪子质心 (n_sub>0) 或 原始记忆 (n_sub==0), 供 density 读出使用"""
        return np.asarray(self.subc) if self.n_sub > 0 else np.asarray(self.memory)

    def update(self, x, pi, lr_c=0.05, lr_w=0.08, w_max=3.0):
        self.access += 1
        self.center += lr_c * pi * (x - self.center)          # 中心轻微漂移
        diff = (x - self.center)[:, None]
        self.cov = 0.98 * self.cov + 0.02 * (diff @ diff.T + np.eye(len(x)) * 1e-3)
        self.cov_inv = np.linalg.inv(self.cov)
        self.w = min(w_max, self.w + lr_w * pi)               # 精度加权的深度强化
        self._update_subcentroids(x)                          # 在线去噪子质心
        if len(self.memory) < self.mem_cap:
            self.memory.append(np.array(x, dtype=float))
        elif self._rng.random() < self.mem_cap / self.access:  # 蓄水池采样保持代表性
            self.memory[self._rng.integers(self.mem_cap)] = np.array(x, dtype=float)


class PWAFM:
    """精度加权吸引子场机 (2D/低维原型)"""

    def __init__(self, T=0.2, tau_novel=0.4, sigma0=0.5, w0=1.0,
                 omega=None, decay=2e-4, w_floor=0.05,
                 use_precision_gate=True, barrier_height=1.2,
                 create_rule='threshold', dp_alpha=1.0, base_scale=3.0, dp_diag_cov=False,
                 basin_model='gauss', proto_scale=1.0, readout='energy', density_bw=1.0,
                 n_sub=0):
        self.T = T
        self.tau_novel = tau_novel        # 新奇阈值 (加权预测误差 + 能量) —— threshold 规则用
        self.sigma0 = sigma0
        self.w0 = w0
        self.omega = omega if omega is not None else 1.0 / sigma0 ** 2  # 精度矩阵(各向同性标量)
        self.decay = decay                # 未访问盆地的可逆衰减率
        self.w_floor = w_floor            # 深度下限 (蒸发阈值之上仍可恢复)
        self.gate = use_precision_gate    # False = 消融精度门控 (退化为纯 AFM softmin)
        self.barrier_height = barrier_height
        self.create_rule = create_rule   # 'threshold'(硬阈值) 或 'dp'(Dirichlet Process/CRP 准则)
        self.dp_alpha = dp_alpha          # DP 集中参数 alpha: 越小越不愿开新盆地
        self.base_scale = base_scale      # 基测度宽度 (相对数据方差的倍数)
        self.dp_diag_cov = dp_diag_cov    # DP 似然是否用对角协方差 (高维稳健)
        self.basin_model = basin_model    # 盆地能量模型: 'gauss'(单高斯) 或 'mixture'(多原型)
        self.proto_scale = proto_scale    # mixture 模式原型势阱带宽
        self.readout = readout            # 分类读出: 'energy'(能量 softmin 投票) 或 'density'(原型核密度)
        self.density_bw = density_bw      # density 读出的核带宽 h (h->0 退化为 1-NN)
        self.n_sub = n_sub                # 每盆地在线去噪子质心数 (0=density 用原始记忆; >0=用去噪子质心)
        self.basins = []
        self._basin_seq = 0               # 实例内盆地计数, 用于确定性播种 (免全局 id 干扰)
        self.conflict_pairs = []          # [(id_a, id_b)] 记录矛盾对
        # 在线估计数据分布 (Welford), 用于自适应基测度 G_0, 免魔数
        self._n_seen = 0
        self._data_mean = None
        self._data_M2 = None

    # ---------- 推断 ----------
    def _energies(self, X):
        return np.stack([b.energy(X) for b in self.basins])          # (n_basin, n_x)

    def _precisions(self, X):
        return np.stack([b.precision(X, self.omega) for b in self.basins])

    def global_energy(self, X):
        """E(x) = -T log sum_i exp(-E_i/T)  (含壁垒项)"""
        E = self._energies(X)
        return -self.T * np.log(np.exp(-E / self.T).sum(0) + 1e-300)

    def _proto_logits(self, X):
        """density 读出: 每盆地对 x 的软最近邻支持度 (log 域 Parzen 核密度)
        s_i(x) = logsumexp_k( -0.5 ||x - m_k||^2 / h^2 ),  m_k 为盆地 i 的原型
        不乘深度 w_i、不经温度 T 扭曲 —— 直接用原型密度作 label 证据
        原型源: n_sub>0 时用在线去噪子质心, 否则用原始记忆样本"""
        X = np.atleast_2d(X)
        h2 = self.density_bw ** 2
        out = []
        for b in self.basins:
            M = b.sub_centroids()
            d2 = ((X[:, None, :] - M[None, :, :]) ** 2).sum(-1)   # (n, m)
            a = -0.5 * d2 / h2
            amax = a.max(1)
            out.append(amax + np.log(np.exp(a - amax[:, None]).sum(1)))
        return np.stack(out)                                     # (n_basin, n)

    def posterior(self, X):
        """p(basin_i|x): energy 读出→精度加权能量混合; density 读出→原型核密度混合"""
        if self.readout == 'density':
            logits = self._proto_logits(np.atleast_2d(X))
        else:
            E = self._energies(X)
            logits = -E / self.T
        if self.gate:
            logits = logits + np.log(self._precisions(X) + 1e-12)
        logits -= logits.max(0, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(0, keepdims=True)

    def novelty(self, X):
        """Novelty(x) = min_i [0.5 eps^T Omega eps + E_i(x)]"""
        X = np.atleast_2d(X)
        if not self.basins:
            return np.full(len(X), np.inf)
        scores = np.stack([b.recon_err2(X, self.omega) + b.energy(X) for b in self.basins])
        return scores.min(0)

    def classify(self, X):
        """后验按标签聚合投票"""
        P = self.posterior(np.atleast_2d(X))
        labels = sorted({b.label for b in self.basins})
        votes = np.stack([P[[i for i, b in enumerate(self.basins) if b.label == lab]].sum(0)
                          for lab in labels])
        return np.array([labels[k] for k in votes.argmax(0)])

    def label_posterior(self, X):
        P = self.posterior(np.atleast_2d(X))
        labels = sorted({b.label for b in self.basins})
        votes = np.stack([P[[i for i, b in enumerate(self.basins) if b.label == lab]].sum(0)
                          for lab in labels])
        return labels, votes

    # ---------- 学习 (局部能量雕塑) ----------
    def _create(self, x, label):
        b = Basin(x, label, w0=self.w0, sigma0=self.sigma0,
                  model=self.basin_model, proto_scale=self.proto_scale,
                  rng_seed=self._basin_seq, n_sub=self.n_sub)
        self._basin_seq += 1
        self.basins.append(b)
        return b

    def _add_barrier(self, ba, bb):
        mid = (ba.center + bb.center) / 2
        sb = max(np.linalg.norm(ba.center - bb.center) / 2, self.sigma0 * 0.6)
        for b in (ba, bb):
            b.barriers.append((mid.copy(), self.barrier_height, sb))
        self.conflict_pairs.append((ba.id, bb.id))

    def _update_data_stats(self, x):
        """Welford 在线更新数据均值/方差, 供基测度自适应"""
        self._n_seen += 1
        if self._data_mean is None:
            self._data_mean = x.copy()
            self._data_M2 = np.zeros_like(x)
            return
        delta = x - self._data_mean
        self._data_mean += delta / self._n_seen
        self._data_M2 += delta * (x - self._data_mean)

    def _base_loglik(self, x):
        """log N(x | data_mean, base_scale^2 * data_var): 宽基测度 G_0 的先验预测似然"""
        d = len(x)
        if self._n_seen < 2:
            var = (self.base_scale * self.sigma0) ** 2
        else:
            data_var = float(self._data_M2.mean() / max(self._n_seen - 1, 1))
            var = self.base_scale ** 2 * data_var + 1e-6
        m2 = float(np.sum((x - self._data_mean) ** 2) / var)
        return -0.5 * (m2 + d * np.log(2 * np.pi * var))

    def _novelty_decision(self, x, scores):
        """返回 (is_novel, explain_mask): 是否需建新盆地 + 各盆地能否解释 x
        - threshold: 硬阈值, explain_i = scores_i <= tau_novel
        - dp/CRP:    比较 log(n_i)+logN(x|basin_i) 与 log(alpha)+logN_base(x), 后者大则新建
        """
        if self.create_rule == 'threshold':
            explain = scores <= self.tau_novel
            return (not explain.any()), explain
        logp_join = np.array([np.log(b.access) + b.gauss_loglik(x, diag=self.dp_diag_cov)
                              for b in self.basins])
        logp_new = np.log(self.dp_alpha) + self._base_loglik(x)
        explain = logp_join >= logp_new
        return (not explain.any()), explain

    def observe(self, x, label=None):
        """最小循环: Observe -> Detect Novelty -> Create/Connect -> Keep/Remove"""
        x = np.asarray(x, dtype=float)
        self._update_data_stats(x)
        if not self.basins:
            self._create(x, label)
            return 'create'
        X = x[None, :]
        scores = np.array([(b.recon_err2(X, self.omega) + b.energy(X))[0] for b in self.basins])
        best = int(scores.argmin())
        bb = self.basins[best]
        is_novel, explain = self._novelty_decision(x, scores)
        event = 'update'
        if is_novel:                                            # 新奇 -> 挖新坑
            self._create(x, label)
            event = 'create'
        elif label is not None and bb.label != label:           # 表面近但语义矛盾
            # 先检查同标签盆地能否解释该输入 (旧知识变体 -> 合并而非重复创建)
            same = [(i, scores[i]) for i in range(len(self.basins))
                    if self.basins[i].label == label and explain[i]]
            if same:
                k = min(same, key=lambda t: t[1])[0]
                sb = self.basins[k]
                sb.update(x, float(sb.precision(X, self.omega)[0]))
                if not any({bb.id, sb.id} <= {a, c} for a, c in self.conflict_pairs):
                    self._add_barrier(bb, sb)                   # 矛盾对仍需壁垒隔离
                event = 'conflict'
            else:                                               # 新概念 -> 新盆地 + 壁垒
                nb = self._create(x, label)
                self._add_barrier(bb, nb)
                event = 'conflict'
        else:                                                   # 归属现有盆地 -> 局部强化
            pi = float(bb.precision(X, self.omega)[0])
            bb.update(x, pi)
        for i, b in enumerate(self.basins):                     # 可逆衰减 (非覆盖)
            if i != best or event != 'update':
                b.w = max(self.w_floor, b.w * (1 - self.decay))
        return event

    def fit_stream(self, X, y):
        events = []
        for xi, yi in zip(X, y):
            events.append(self.observe(xi, yi))
        return events

    # ---------- 知识编辑接口 (显式可定位) ----------
    def delete_concept(self, label):
        self.basins = [b for b in self.basins if b.label != label]

    def summary(self):
        from collections import Counter
        c = Counter(b.label for b in self.basins)
        return {'n_basins': len(self.basins), 'per_label': dict(c),
                'depths': {b.id: round(b.w, 3) for b in self.basins},
                'n_conflict_barriers': len(self.conflict_pairs)}
