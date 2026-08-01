# -*- coding: utf-8 -*-
"""
Split-MNIST 基准: DP-ProtOn vs EWC / 经验重放 / 微调 / 联合训练。

所有方法接收相同的 PCA 白化 50 维特征（冻结特征提取器），保证公平比较。
"""
import os, sys, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pw_afm import PWAFM

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

N_CLASSES = 10
N_PER_TASK = 2
N_TASKS = N_CLASSES // N_PER_TASK  # 5

# ============================================================
# Part 1: 数据
# ============================================================
def build_mnist_tasks(n_comp=50, seed=0):
    """Split-MNIST, PCA 白化 50 维, 5 task x 2 class"""
    raw = datasets.MNIST(root='./data', train=True, download=False)
    X = raw.data.numpy().reshape(-1, 28 * 28).astype(float) / 255.0
    y = raw.targets.numpy()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    # 30% 预训练池
    n_pre = int(0.3 * len(X))
    Xpre = X[:n_pre]; Xrest, yrest = X[n_pre:], y[n_pre:]
    scaler = StandardScaler().fit(Xpre)
    pca = PCA(n_components=n_comp, whiten=True, random_state=seed).fit(scaler.transform(Xpre))
    proj = lambda Z: pca.transform(scaler.transform(Z))
    Xr = proj(Xrest)
    tasks = []
    for k in range(N_TASKS):
        cls = list(range(k * N_PER_TASK, (k + 1) * N_PER_TASK))
        m = np.isin(yrest, cls)
        Xk, yk = Xr[m], yrest[m]
        r = np.random.default_rng(100 + k)
        p = r.permutation(len(Xk)); Xk, yk = Xk[p], yk[p]
        s = int(0.7 * len(Xk))
        tasks.append((Xk[:s], yk[:s], Xk[s:], yk[s:]))
    return tasks, n_comp

def load_test_set(tasks):
    Xte = np.vstack([t[2] for t in tasks])
    yte = np.concatenate([t[3] for t in tasks])
    return Xte, yte

# ============================================================
# Part 2: 评估 & DP-ProtOn
# ============================================================
def eval_dpproton(tasks, dp_alpha=1.0, n_sub=1, sigma0=1.0, density_bw=1.0):
    """DP-ProtOn: DP + density + sub-centroids"""
    afm = PWAFM(T=0.3, sigma0=sigma0, create_rule='dp', dp_alpha=dp_alpha,
                base_scale=3.0, dp_diag_cov=True, readout='density',
                density_bw=density_bw, n_sub=n_sub)
    A = np.full((N_TASKS, N_TASKS), np.nan)
    for t in range(N_TASKS):
        afm.fit_stream(tasks[t][0], tasks[t][1])
        for k in range(t + 1):
            A[t, k] = (afm.classify(tasks[k][2]) == tasks[k][3]).mean()
    acc = float(np.nanmean(A[-1]))
    bwt = float(np.mean([A[-1, k] - A[k, k] for k in range(N_TASKS - 1)])) if N_TASKS > 1 else 0.0
    nb = len(afm.basins)
    np_ = sum(len(b.sub_centroids()) for b in afm.basins)
    return acc, bwt, nb, np_


# ============================================================
# Part 3: MLP baselines (PyTorch)
# ============================================================
class SimpleMLP(nn.Module):
    """简单 2 层 MLP"""
    def __init__(self, input_dim=50, hidden=256, n_class=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_class))
    def forward(self, x):
        return self.net(x)

def train_mlp(model, X, y, lr=1e-3, epochs=20, batch=256, callback=None):
    """训练 MLP，callback(model, epoch) 用于 EWC 额外 loss"""
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(bx), by)
            if callback is not None:
                loss = loss + callback(model, ep)
            loss.backward()
            opt.step()
            total_loss += loss.item()
    return total_loss / len(dl)

def eval_mlp(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(DEVICE)).argmax(1).cpu().numpy()
    return (pred == y).mean()

def compute_fisher(model, X, y):
    """对角 Fisher 信息矩阵"""
    model.eval()
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    ce = nn.CrossEntropyLoss(reduction='sum')
    total = 0
    for bx, by in dl:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        logits = model(bx)
        loss = ce(logits, by)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                fisher[name] += p.grad ** 2
            total += len(bx)
    for name in fisher:
        fisher[name] /= total
    return fisher


# ---- EWC ----
def run_ewc(tasks, lambda_ewc=1000, hidden=256, lr=1e-3, epochs=20):
    """EWC baseline (online EWC with single consolidate step per task)"""
    model = SimpleMLP(tasks[0][0].shape[1], hidden).to(DEVICE)
    A = np.full((N_TASKS, N_TASKS), np.nan)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    
    # Task 0: train normally
    X0, y0 = tasks[0][0], tasks[0][1]
    train_mlp(model, X0, y0, lr=lr, epochs=epochs)
    for k in range(1):
        A[0, k] = eval_mlp(model, tasks[k][2], tasks[k][3])
    
    # Consolidate after task 0
    fisher_prev = compute_fisher(model, X0, y0)
    params_prev = {name: p.detach().clone() for name, p in model.named_parameters()}
    
    for t in range(1, N_TASKS):
        Xt, yt = tasks[t][0], tasks[t][1]
        Xall = np.vstack([tasks[tt][0] for tt in range(t + 1)])
        yall = np.concatenate([tasks[tt][1] for tt in range(t + 1)])
        
        # EWC training
        def ewc_callback(model, epoch):
            loss = torch.tensor(0.0, device=DEVICE)
            for name, p in model.named_parameters():
                if name in fisher_prev:
                    loss += (lambda_ewc / 2.0) * (fisher_prev[name] * (p - params_prev[name]) ** 2).sum()
            return loss
        
        train_mlp(model, Xt, yt, lr=lr, epochs=epochs, callback=ewc_callback)
        
        # Re-consolidate (online EWC: running estimate)
        fisher_curr = compute_fisher(model, Xall, yall)
        for name in fisher_prev:
            fisher_prev[name] = 0.5 * (fisher_prev[name] + fisher_curr[name])
        params_prev = {name: p.detach().clone() for name, p in model.named_parameters()}
        
        for k in range(t + 1):
            A[t, k] = eval_mlp(model, tasks[k][2], tasks[k][3])
    
    acc = float(np.nanmean(A[-1]))
    bwt = float(np.mean([A[-1, k] - A[k, k] for k in range(N_TASKS - 1)]))
    return acc, bwt


# ---- Experience Replay ----
def run_replay(tasks, mem_per_class=3, hidden=256, lr=1e-3, epochs=20):
    """经验重放：每类存 mem_per_class 个 exemplar，总预算同 DP 原型数"""
    model = SimpleMLP(tasks[0][0].shape[1], hidden).to(DEVICE)
    A = np.full((N_TASKS, N_TASKS), np.nan)
    buffer_x, buffer_y = [], []
    
    for t in range(N_TASKS):
        Xt, yt = tasks[t][0], tasks[t][1]
        # Select exemplars for new classes (random, herding would be better)
        n_seen_classes = (t + 1) * N_PER_TASK
        seen_classes = list(range(n_seen_classes))
        
        # Add new exemplars
        for c in range(t * N_PER_TASK, (t + 1) * N_PER_TASK):
            mc = yt == c
            Xc = Xt[mc]
            if len(Xc) > mem_per_class:
                idx = np.random.default_rng(c).choice(len(Xc), mem_per_class, replace=False)
                buffer_x.append(Xc[idx])
            else:
                buffer_x.append(Xc)
            buffer_y.append(np.full(len(buffer_x[-1]), c))
        
        # Training data = current task + buffer
        if len(buffer_x) > 0:
            Xr = np.vstack(buffer_x); yr = np.concatenate(buffer_y)
            X_train = np.vstack([Xt, Xr]); y_train = np.concatenate([yt, yr])
        else:
            X_train, y_train = Xt, yt
        
        train_mlp(model, X_train, y_train, lr=lr, epochs=epochs)
        for k in range(t + 1):
            A[t, k] = eval_mlp(model, tasks[k][2], tasks[k][3])
    
    acc = float(np.nanmean(A[-1]))
    bwt = float(np.mean([A[-1, k] - A[k, k] for k in range(N_TASKS - 1)]))
    n_stored = sum(len(b) for b in buffer_x)
    return acc, bwt, n_stored


# ---- Fine-tuning (lower bound) ----
def run_finetune(tasks, hidden=256, lr=1e-3, epochs=20):
    model = SimpleMLP(tasks[0][0].shape[1], hidden).to(DEVICE)
    A = np.full((N_TASKS, N_TASKS), np.nan)
    for t in range(N_TASKS):
        Xt, yt = tasks[t][0], tasks[t][1]
        train_mlp(model, Xt, yt, lr=lr, epochs=epochs)
        for k in range(t + 1):
            A[t, k] = eval_mlp(model, tasks[k][2], tasks[k][3])
    acc = float(np.nanmean(A[-1]))
    bwt = float(np.mean([A[-1, k] - A[k, k] for k in range(N_TASKS - 1)]))
    return acc, bwt


# ---- Joint (upper bound) ----
def run_joint(tasks, hidden=256, lr=1e-3, epochs=40):
    model = SimpleMLP(tasks[0][0].shape[1], hidden).to(DEVICE)
    Xall = np.vstack([t[0] for t in tasks])
    yall = np.concatenate([t[1] for t in tasks])
    train_mlp(model, Xall, yall, lr=lr, epochs=epochs)
    Xte, yte = load_test_set(tasks)
    acc = eval_mlp(model, Xte, yte)
    return acc


# ============================================================
# Part 4: Run + Report
# ============================================================
if __name__ == '__main__':
    print('=' * 65)
    print('  Split-MNIST Benchmark: DP-ProtOn vs Baselines')
    print(f'  Device: {DEVICE}')
    print('=' * 65)
    
    tasks, dim = build_mnist_tasks(n_comp=50)
    print(f'\nData: {dim}D PCA-whitened, {N_TASKS} tasks x {N_PER_TASK} classes')
    for k, (Xtr, ytr, Xte, yte) in enumerate(tasks):
        print(f'  Task {k}: classes {set(ytr.tolist())}, train={len(Xtr)} test={len(Xte)}')
    
    results = {}
    
    # ---- DP-ProtOn ----
    print('\n--- DP-ProtOn (ours) ---')
    acc, bwt, nb, np_ = eval_dpproton(tasks, dp_alpha=1.0, n_sub=1)
    results['DP-ProtOn'] = (acc, bwt, np_)
    print(f'  ACC={acc:.4f}  BWT={bwt:+.4f}  prototypes={np_}  basins={nb}')
    
    # ---- Experience Replay (match prototype count) ----
    mem_pc = max(1, np_ // N_CLASSES)
    print(f'\n--- Experience Replay (mem={mem_pc}/class, total ~{mem_pc * N_CLASSES}) ---')
    acc, bwt, ns = run_replay(tasks, mem_per_class=mem_pc)
    results['Replay'] = (acc, bwt, ns)
    print(f'  ACC={acc:.4f}  BWT={bwt:+.4f}  stored={ns}')
    
    # ---- EWC ----
    print('\n--- EWC (lambda=1000) ---')
    acc, bwt = run_ewc(tasks, lambda_ewc=1000)
    results['EWC'] = (acc, bwt, 0)
    print(f'  ACC={acc:.4f}  BWT={bwt:+.4f}')
    
    # ---- Fine-tuning ----
    print('\n--- Fine-tuning (lower bound) ---')
    acc, bwt = run_finetune(tasks)
    results['Fine-tune'] = (acc, bwt, 0)
    print(f'  ACC={acc:.4f}  BWT={bwt:+.4f}')
    
    # ---- Joint (upper bound) ----
    print('\n--- Joint training (upper bound) ---')
    acc = run_joint(tasks)
    results['Joint'] = (acc, 0, 0)
    print(f'  ACC={acc:.4f}')
    
    # ---- Summary ----
    print('\n' + '=' * 65)
    print('  Summary (sorted by ACC)')
    print('=' * 65)
    print(f'  {"Method":<18}{"ACC":>8}{"BWT":>8}{"Memory":>8}')
    for name, (a, b, m) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f'  {name:<18}{a:>8.4f}{b:>+8.4f}{m:>8}')
    
    # ---- Visualization ----
    names = list(results.keys())
    accs = [results[n][0] for n in names]
    bwts = [results[n][1] for n in names]
    mems = [results[n][2] for n in names]
    colors = ['tab:green' if 'DP' in n else 'tab:orange' if 'Replay' in n else
              'tab:blue' if 'EWC' in n else 'tab:red' if 'Fine' in n else 'gray' for n in names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    # Left: ACC bar
    bars = ax1.bar(names, accs, color=colors)
    for bar, a in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{a:.3f}', ha='center', va='bottom', fontsize=8)
    ax1.axhline(results['Joint'][0], color='gray', ls='--', lw=1, label=f'Joint (upper)={results["Joint"][0]:.3f}')
    ax1.set_ylabel('Final ACC'); ax1.set_title('Split-MNIST: ACC Comparison')
    ax1.legend(fontsize=8)
    # Right: BWT bar
    ax2.bar(names, bwts, color=colors)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_ylabel('BWT'); ax2.set_title('Backward Transfer')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'mnist_bench.png'), dpi=150)
    plt.close(fig)
    print(f'\nChart saved to {OUT}/mnist_bench.png')
