# -*- coding: utf-8 -*-
"""DP-ProtOn: Dirichlet-Process-guided Prototype Online learner.

A minimal, gradient-free, single-pass prototype classifier for continual
learning under frozen features, with explicit knowledge-editing support.

Quick start:
    from dp_proton import DPProtOn
    model = DPProtOn(alpha=1.0, n_sub=1)
    model.fit_stream(X_train, y_train)
    preds = model.classify(X_test)
    model.delete_class(9)      # zero-retraining knowledge editing
"""

from .core import Basin, DPProtOn

__all__ = ['Basin', 'DPProtOn']
__version__ = '1.0.0'
