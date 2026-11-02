"""Metric functions the dispatcher needs that `core.metrics` doesn't provide
(precision/recall/F1, ROC-AUC). Kept dependency-light (numpy + scipy's
`rankdata`, both already platform dependencies) rather than requiring
scikit-learn, matching the pattern `core.metrics` and
`classical.machine_learning.baselines` already established -- the dispatcher
must be able to compute a full metric set even in an sklearn-less
environment, since several *algorithms* (not just metrics) already document
an sklearn-optional fallback path.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label) -> tuple[float, float, float]:
    """Binary precision/recall/F1 with `positive_label` as the positive class.
    Works for any label dtype/values (not just {0,1} or {-1,+1})."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == positive_label) & (y_true == positive_label)))
    fp = int(np.sum((y_pred == positive_label) & (y_true != positive_label)))
    fn = int(np.sum((y_pred != positive_label) & (y_true == positive_label)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def macro_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Macro-averaged precision/recall/F1 over every class present in
    y_true -- the standard multiclass generalization of `precision_recall_f1`
    (each class treated as "positive" in turn, then averaged unweighted, so
    minority regimes/classes are not swamped by a majority class)."""
    classes = np.unique(y_true)
    precisions, recalls, f1s = [], [], []
    for c in classes:
        p, r, f = precision_recall_f1(y_true, y_pred, c)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray, positive_label) -> float:
    """ROC-AUC via the Mann-Whitney U / rank-sum identity:

        AUC = (sum of ranks of positive-class scores - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Equivalent to sklearn's `roc_auc_score` for the binary case, without the
    dependency. Ties are handled via average ranking (`scipy.stats.rankdata`
    default). Returns NaN if either class is empty in `y_true` (AUC is
    undefined there, not zero -- and NaN propagates visibly rather than
    silently reading as "worst possible").
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos_mask = y_true == positive_label
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_ranks_pos = float(ranks[pos_mask].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


__all__ = ["precision_recall_f1", "macro_precision_recall_f1", "roc_auc_binary"]
