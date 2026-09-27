"""Evaluation metrics. Every metric is computed against the full synthetic ground truth: the target argument
must be an :class:`OracleTargets` (decision 0009, ground-truth-only evaluation rule). Passing training
targets, raw arrays or anything else raises ``TypeError``."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from fcip.labels.oracle import OracleTargets


def _check(targets: object, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(targets, OracleTargets):
        raise TypeError(f"metrics require OracleTargets (full ground truth), got {type(targets).__name__}")
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != targets.y.shape:
        raise ValueError("scores and targets differ in shape")
    if not np.isfinite(s).all():
        raise ValueError("non-finite scores")
    return targets.y.astype(bool), s


def average_precision(targets: OracleTargets, scores: np.ndarray) -> float:
    """PR-AUC estimated as average precision (the pre-registered estimator, EXP-M1-G)."""
    y, s = _check(targets, scores)
    if not y.any():
        raise ValueError("average precision is undefined without positives")
    return float(average_precision_score(y, s))


def recall_at_k(targets: OracleTargets, scores: np.ndarray, k: int) -> float:
    y, s = _check(targets, scores)
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / y.sum()) if y.any() else float("nan")


def positives(targets: OracleTargets) -> int:
    _check(targets, np.zeros(targets.y.shape))
    return int(targets.y.sum())
