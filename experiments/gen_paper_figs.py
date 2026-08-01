# -*- coding: utf-8 -*-
"""Generate publication-quality figures for the DP-ProtOn paper (TMLR).

Output:
  results/fig_multiseed.png   — 3-panel: build rule, prototype source, Split-MNIST
  results/fig_edit.png        — 3-panel: delete, relabel, inject
  results/fig_architecture.png — Pipeline conceptual diagram

All data verified against exp_multiseed.py / exp_edit.py output (RESULTS_MULTISEED.md).
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Global style: clean, publication-grade ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9.5, 'axes.labelsize': 10.5, 'axes.titlesize': 12,
    'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'figure.dpi': 150, 'savefig.dpi': 350, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'axes.axisbelow': True,
})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════
# VERIFIED DATA  (all per-seed arrays from experiment output)
# ═══════════════════════════════════════════════════

# ── A: Build rule ablation (digits 30-D PCA-white, 5 seeds) ──
dpA  = np.array([0.9284, 0.9366, 0.9313, 0.9144, 0.9407])   # DP-ProtOn ACC
kmA  = np.array([0.2152, 0.2263, 0.1684, 0.3132, 0.2480])   # threshold k-means ACC
dp_n = np.array([32, 29, 28, 25, 35])                        # DP prototypes
km_n = np.array([33, 33, 22, 27, 31])                        # threshold prototypes

# ── B: Prototype-source ablation (digits 30-D, 5 seeds, ≈30 protos) ──
subB = np.array([0.9284, 0.9366, 0.9313, 0.9144, 0.9407])   # sub-centroid k=1
rawB = np.array([0.8964, 0.9394, 0.9261, 0.8925, 0.9172])   # raw memory
k30B = np.array([0.8924, 0.8974, 0.9000, 0.8895, 0.9024])   # offline k-means-30
knnB = np.array([0.9528, 0.9579, 0.9474, 0.9368, 0.9631])   # 1-NN upper bound

# ── C: Split-MNIST continual (50-D PCA-white, 3 seeds) ──
accC = {'Joint':      np.array([0.972, 0.968, 0.968]),
        'DP-ProtOn':  np.array([0.874, 0.877, 0.873]),
        'Replay':     np.array([0.655, 0.697, 0.684]),
        'EWC':        np.array([0.195, 0.194, 0.196]),
        'Fine-tune':  np.array([0.198, 0.198, 0.198])}

# ── D: Knowledge editing (digits 30-D, 5 seeds, from exp_edit.py output) ──
# Delete
del_before  = np.array([0.8049, 0.9744, 0.8333, 0.7632, 0.9070])  # class-9 pre
del_after   = np.array([0.0,    0.0,    0.0,    0.0,    0.0   ])  # class-9 post
del_shift   = np.array([0.0,    0.1111, 0.0435, 0.0263, 0.0526])  # other-class max Δ
del_nov_pre = np.array([10.94,  9.60,   9.24,   9.06,   8.76  ])  # novelty pre
del_nov_pos = np.array([12.84,  13.01,  11.68,  10.80,  11.60 ])  # novelty post

# Relabel
rel_remap = np.array([1.0,    1.0,    1.0,    1.0,    1.0   ])  # remap-identical
rel_other = np.array([0.9881, 0.9970, 0.9971, 0.9883, 1.0000])  # non-target identical

# Inject
inj_before = np.array([0.0,    0.0,    0.0,    0.0,    0.0   ])  # class-9 pre-inject
inj_after  = np.array([0.7805, 0.9487, 0.9000, 0.8158, 0.8372])  # class-9 post-inject
inj_old_id = np.array([0.9912, 0.9824, 0.9914, 0.9854, 0.9911])  # old identical
inj_shift  = np.array([0.0303, 0.0833, 0.0286, 0.0294, 0.0500])  # old max Δ

# ── Color palette (colorblind-friendly) ──
C_DP      = '#2C7BB6'   # blue    — DP-ProtOn
C_BASE    = '#D7191C'   # red     — baselines / negative
C_ALT     = '#FDAE61'   # orange  — alternative methods
C_GREY    = '#ABD9E9'   # grey    — oracle / upper-bound
C_AFTER   = '#1A9641'   # green   — after-edit good outcome
C_BEFORE  = '#A6D96A'   # light green — before
C_DEL     = '#D7191C'   # red     — deletion target
C_GREY_DK = '#5E4FA2'   # purple  — joint oracle


# ═══════════════════════════════════════════════════
# FIGURE 1: Multi-seed validation  (3 panels)
# ═══════════════════════════════════════════════════
def fig_multiseed():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plt.subplots_adjust(wspace=0.32)

    # ---- Panel A: Build rule ----
    ax = axes[0]
    xs = np.array([0, 1.3]); w = 0.45
    m = [dpA.mean(), kmA.mean()]; s = [dpA.std(), kmA.std()]
    colors = [C_DP, C_BASE]
    labels_bar = ['DP-ProtOn\n(CRP)', 'Threshold\nk-means']

    bars = ax.bar(xs, m, w, yerr=s, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.6, error_kw={'linewidth': 1.2})
    for b, v, sd in zip(bars, m, s):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.025,
                f'{v:.4f}', ha='center', fontsize=9.5, fontweight='bold', color='#333')
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.07,
                f'±{sd:.4f}', ha='center', fontsize=7.5, color='#666')
    ax.set_xticks(xs); ax.set_xticklabels(labels_bar)
    ax.set_ylim(0, 1.10); ax.set_ylabel('Final Accuracy')
    ax.set_title('(a) Build rule', fontweight='bold', loc='left', pad=8)
    # Δ annotation
    delta = dpA.mean() - kmA.mean()
    delta_s = (dpA - kmA).std()
    ax.text(0.65, 0.93, f'Δ = {delta:.3f} ± {delta_s:.3f}',
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD',
                      edgecolor='#CCBB55', alpha=0.85))
    ax.text(0.99, 0.02, 'digits 30-D, 5 seeds', transform=ax.transAxes,
            ha='right', fontsize=7, color='#888')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # ---- Panel B: Prototype source ----
    ax = axes[1]
    labels = ['Sub-centroid\n(k=1)', 'Raw\nmemory', 'Offline\nk-means-30', '1-NN\n(upper bound)']
    xs = np.arange(len(labels)); w = 0.45
    m = [subB.mean(), rawB.mean(), k30B.mean(), knnB.mean()]
    s = [subB.std(), rawB.std(), k30B.std(), knnB.std()]
    colors = [C_DP, C_ALT, '#FEE08B', C_GREY]

    bars = ax.bar(xs, m, w, yerr=s, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.6, error_kw={'linewidth': 1.2})
    for b, v, sd in zip(bars, m, s):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.02,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold', color='#333')
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylim(0, 1.10)
    ax.set_title('(b) Prototype source', fontweight='bold', loc='left', pad=8)
    delta2 = subB.mean() - k30B.mean()
    delta2_s = (subB - k30B).std()
    ax.text(0.5, 0.93, f'Δ(sub − k-means) = {delta2:.3f} ± {delta2_s:.3f}',
            transform=ax.transAxes, fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD',
                      edgecolor='#CCBB55', alpha=0.85))
    ax.text(0.99, 0.02, 'digits 30-D, 5 seeds, ≈30 protos', transform=ax.transAxes,
            ha='right', fontsize=7, color='#888')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # ---- Panel C: Split-MNIST ----
    ax = axes[2]
    order = ['Joint', 'DP-ProtOn', 'Replay', 'EWC', 'Fine-tune']
    xs = np.arange(len(order)); w = 0.45
    m = [accC[k].mean() for k in order]; s = [accC[k].std() for k in order]
    colors = [C_GREY_DK, C_DP, C_ALT, '#F46D43', C_BASE]

    bars = ax.bar(xs, m, w, yerr=s, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.6, error_kw={'linewidth': 1.2})
    for b, v, sd in zip(bars, m, s):
        lbl = f'{v:.3f}±{sd:.3f}' if sd >= 0.001 else f'{v:.3f}'
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.025,
                lbl, ha='center', fontsize=7.5, fontweight='bold', color='#333',
                rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels(order); ax.set_ylim(0, 1.15)
    ax.set_title('(c) Split-MNIST continual', fontweight='bold', loc='left', pad=8)
    ax.axhline(y=0.20, color='#888', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(4.5, 0.202, 'random (0.2)', fontsize=6.5, color='#888', ha='right')
    ax.text(0.99, 0.02, '50-D PCA, 3 seeds', transform=ax.transAxes,
            ha='right', fontsize=7, color='#888')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    path = os.path.join(OUT, 'fig_multiseed.png')
    fig.savefig(path, pad_inches=0.15)
    plt.close(fig)
    print(f'[fig_multiseed] saved -> {path}')
    return path


# ═══════════════════════════════════════════════════
# FIGURE 2: Knowledge editing  (3 panels)
# ═══════════════════════════════════════════════════
def fig_edit():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    plt.subplots_adjust(wspace=0.32)

    # ---- Panel A: Delete class 9 ----
    ax = axes[0]
    xs = np.array([0, 1.3]); w = 0.45
    m = [del_before.mean(), del_after.mean()]
    s = [del_before.std(), del_after.std()]
    colors = [C_DP, C_DEL]

    bars = ax.bar(xs, m, w, yerr=s, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.6, error_kw={'linewidth': 1.2})
    for b, v, sd in zip(bars, m, s):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.025,
                f'{v:.4f}', ha='center', fontsize=10, fontweight='bold', color='#333')
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.065,
                f'±{sd:.4f}', ha='center', fontsize=7.5, color='#666')
    ax.set_xticks(xs); ax.set_xticklabels(['Before\ndeletion', 'After\ndeletion'])
    ax.set_ylim(0, 1.15); ax.set_ylabel('Class-9 Accuracy')
    ax.set_title('(a) Delete class 9', fontweight='bold', loc='left', pad=8)
    ax.text(0.65, 0.89, f'Other-class max Δ\n{del_shift.mean():.3f} ± {del_shift.std():.3f}',
            transform=ax.transAxes, fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD',
                      edgecolor='#CCBB55', alpha=0.85))
    ax.text(0.99, 0.02, 'digits 30-D, 5 seeds', transform=ax.transAxes,
            ha='right', fontsize=7, color='#888')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # ---- Panel B: Relabel 7 → 99 ----
    ax = axes[1]
    xs = np.array([0, 1.3]); w = 0.45
    m = [rel_remap.mean(), rel_other.mean()]
    s = [rel_remap.std(), rel_other.std()]
    colors = [C_DP, C_AFTER]

    bars = ax.bar(xs, m, w, yerr=s, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.6, error_kw={'linewidth': 1.2})
    for b, v, sd in zip(bars, m, s):
        fmt = f'{v:.4f}' if sd < 0.001 else f'{v:.4f}±{sd:.4f}'
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.006,
                fmt, ha='center', fontsize=9.5, fontweight='bold', color='#333')
    ax.set_xticks(xs); ax.set_xticklabels(['Remap-identical\n(99→7→compare)', 'Non-target\nidentical'])
    ax.set_ylim(0.95, 1.05); ax.set_ylabel('Prediction Agreement')
    ax.set_title('(b) Relabel 7 → 99', fontweight='bold', loc='left', pad=8)
    ax.axhline(y=1.0, color='#888', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(0.99, 0.02, 'Label = detachable pointer', transform=ax.transAxes,
            ha='right', fontsize=7, color='#888')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # ---- Panel C: Inject class 9 (20 samples) ----
    ax = axes[2]
    xs = np.array([0, 1.3]); w = 0.45
    m = [inj_before.mean(), inj_after.mean()]
    s = [inj_before.std(), inj_after.std()]
    colors = [C_DEL, C_DP]

    bars = ax.bar(xs, m, w, yerr=s, capsize=5, color=colors,
                  edgecolor='white', linewidth=0.6, error_kw={'linewidth': 1.2})
    for b, v, sd in zip(bars, m, s):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.025,
                f'{v:.4f}', ha='center', fontsize=10, fontweight='bold', color='#333')
        if sd > 0.001:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + sd + 0.07,
                    f'±{sd:.4f}', ha='center', fontsize=7.5, color='#666')
    ax.set_xticks(xs); ax.set_xticklabels(['Before\ninjection', 'After\ninjection'])
    ax.set_ylim(0, 1.15); ax.set_ylabel('Class-9 Accuracy')
    ax.set_title('(c) Inject class 9 (20 samples)', fontweight='bold', loc='left', pad=8)
    ax.text(0.6, 0.89,
            f'Old-class identical\n{inj_old_id.mean():.4f} ± {inj_old_id.std():.4f}',
            transform=ax.transAxes, fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD',
                      edgecolor='#CCBB55', alpha=0.85))
    ax.text(0.99, 0.02, 'digits 30-D, 5 seeds', transform=ax.transAxes,
            ha='right', fontsize=7, color='#888')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    path = os.path.join(OUT, 'fig_edit.png')
    fig.savefig(path, pad_inches=0.15)
    plt.close(fig)
    print(f'[fig_edit] saved -> {path}')
    return path


# ═══════════════════════════════════════════════════
# FIGURE 3: Architecture / pipeline diagram
# ═══════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    ax.axis('off')

    # ── Pipeline boxes with rounded corners ──
    box_w, box_h = 1.55, 1.5
    box_y = 2.8
    boxes_data = [
        (0.3,  'Streaming\ninput\n(x, y)',    '#E8F4FD', '#2C7BB6'),
        (2.3,  'CRP creation\ncriterion',      '#D1ECF9', '#2C7BB6'),
        (4.3,  'Online centroid\nupdate',       '#A8D8F0', '#2C7BB6'),
        (6.3,  'Density\nreadout',             '#74B9E0', '#2C7BB6'),
        (8.3,  'Prediction\nŷ',                '#4FA3D1', '#2C7BB6'),
    ]

    for x, label, face, edge in boxes_data:
        rect = FancyBboxPatch((x, box_y - box_h/2), box_w, box_h,
                               boxstyle='round,pad=0.15', facecolor=face,
                               edgecolor=edge, linewidth=1.3, zorder=2)
        ax.add_patch(rect)
        ax.text(x + box_w/2, box_y, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#1A3A5C', zorder=3)

    # ── Arrows between boxes ──
    for i in range(len(boxes_data) - 1):
        x1 = boxes_data[i][0] + box_w
        x2 = boxes_data[i+1][0]
        ax.annotate('', xy=(x2, box_y), xytext=(x1, box_y),
                    arrowprops=dict(arrowstyle='->', color='#2C7BB6',
                                    lw=2.2, connectionstyle='arc3,rad=0'))

    # ── Bottom annotations (formulas / explanations) ──
    ann_data = [
        (boxes_data[0][0] + box_w/2, 'Class-conditional\nallocation'),
        (boxes_data[1][0] + box_w/2, 'log n_i + log N(x|c_i,Σ_i)\nvs log α + log p_0(x)'),
        (boxes_data[2][0] + box_w/2, 'm += (x-m)/n\n(competitive mean)'),
        (boxes_data[3][0] + box_w/2, 'logsumexp_k\n(-1/2 ||x-m_k||^2 / h^2)'),
        (boxes_data[4][0] + box_w/2, 'argmax_i s_i(x)'),
    ]
    for x, label in ann_data:
        ax.text(x, 1.55, label, ha='center', va='top', fontsize=7, color='#4477AA',
                style='italic', zorder=1)

    # ── Editing operations row ──
    edit_y = 0.55
    ax.text(5.3, edit_y + 0.4, 'Knowledge Editing  —  O(1) set operations, zero retraining',
            ha='center', fontsize=10.5, fontweight='bold', color='#B26500',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E7',
                      edgecolor='#E6A817', alpha=0.92, linewidth=1.2))

    # Delete / Relabel / Inject sub-boxes
    for i, (label, color) in enumerate([
        ('Delete\nremove basins\nby label', '#FDD0D0'),
        ('Relabel\nreassign ℓ→ℓ′\ngeometry intact', '#D1ECF9'),
        ('Inject\nstream new\nsamples only', '#D4EDDA'),
    ]):
        bx = 0.6 + i * 3.5
        rect = FancyBboxPatch((bx, edit_y - 0.25), 2.6, 0.85,
                               boxstyle='round,pad=0.1', facecolor=color,
                               edgecolor='#999', linewidth=0.8, zorder=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(bx + 1.3, edit_y + 0.18, label, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#333', zorder=3)

    ax.set_title('DP-ProtOn Pipeline', fontsize=13, fontweight='bold',
                 color='#1A3A5C', pad=14)

    path = os.path.join(OUT, 'fig_architecture.png')
    fig.savefig(path, pad_inches=0.15)
    plt.close(fig)
    print(f'[fig_architecture] saved -> {path}')
    return path


# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    fig_multiseed()
    fig_edit()
    print('All figures regenerated with verified data + publication styling.')
