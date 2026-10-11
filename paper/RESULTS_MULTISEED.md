# DP-ProtOn multi-seed results (paper-ready numbers, mean+-std)

## A. Build-rule ablation: DP/CRP vs distance-threshold k-means (digits 30D PCA-white, 5 seeds)
Matched prototype budget per seed.
- DP-ProtOn      ACC = 0.9303 +- 0.0090   (prototypes 29.8 +- 3.4)
- Threshold kmeans ACC = 0.2342 +- 0.0473 (prototypes 29.2 +- 4.2)
- Delta(DP - kmeans) = 0.6960 +- 0.0528   -> DP stably dominates (std nowhere near 0)
Per-seed DP ACC: 0.9284, 0.9366, 0.9313, 0.9144, 0.9407

## B. Prototype-source ablation (digits 30D, 5 seeds, matched ~30 prototypes)
- Online denoised sub-centroid k=1 : ACC = 0.9303 +- 0.0090  (30 protos)
- Raw reservoir memory (n_sub=0)   : ACC = 0.9143 +- 0.0177
- Offline kmeans-30 (3/class)      : ACC = 0.8963 +- 0.0048
- 1-NN (store all ~4700)           : ACC = 0.9516 +- 0.0090  (upper bound)
- Delta(sub-centroid - kmeans30) = 0.0340 +- 0.0053 -> online denoising stably beats offline kmeans at equal budget

## C. Split-MNIST continual learning (50D PCA-white, 5 tasks x 2 classes, 3 seeds)
Final average accuracy (ACC) and backward transfer (BWT), mean +- std:
- Joint (oracle) : ACC = 0.9693 +- 0.0017   BWT = --
- DP-ProtOn      : ACC = 0.8747 +- 0.0019   BWT = -0.0471 +- 0.0049
- Replay         : ACC = 0.6787 +- 0.0174   BWT = -0.3921 +- 0.0226
- EWC            : ACC = 0.1949 +- 0.0004   BWT = -0.9927 +- 0.0005
- Fine-tune      : ACC = 0.1978 +- 0.0002   BWT = -0.9947 +- 0.0003
Per-seed DP-ProtOn ACC: 0.874, 0.877, 0.873  (extremely tight; single-seed TECH_REPORT 0.874 reproduced)
=> DP-ProtOn: +0.20 over Replay, near-zero forgetting (BWT ~ -0.05) vs total collapse of EWC/Fine-tune (BWT ~ -0.99)

## D. Knowledge editing: three surgeries, zero retraining (digits 30D, 5 seeds)
[1] DELETE class 9
    class-9 ACC : 0.8565 +- 0.0753  ->  0.0000 +- 0.0000  (surgically removed)
    other-class max ACC shift : 0.0467 +- 0.0369  (small; rerouted points only)
    class-9 novelty score : 9.52 +- 0.76  ->  11.98 +- 0.83  (rises: model flags as unknown)
[2] RELABEL class 7 -> 99 (fresh label)
    remap-identical (99->7 then compare) : 1.0000 +- 0.0000  (decision geometry byte-identical)
    non-target predictions identical : 0.9941 +- 0.0049
    => label is a detachable pointer, decoupled from the attractor geometry
[3] INJECT held-out class 9 from 20 samples (post-hoc)
    class-9 ACC : 0.0000 +- 0.0000  ->  0.8564 +- 0.0603  (recovers full accuracy)
    old-class predictions identical : 0.9883 +- 0.0037  (no catastrophic forgetting)
    old-class max ACC shift : 0.0443 +- 0.0211
Contrast: a softmax MLP cannot delete/relabel/add a class without retraining the whole net.

Figures: results/multiseed.png (A/B/C error bars), results/edit_surgery.png (not regenerated in compact edit run)
