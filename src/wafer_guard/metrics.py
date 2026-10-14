from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    return tn, fp, fn, tp


def precision_recall_fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> dict[str, float]:
    tn, fp, fn, tp = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    fbeta = (1.0 + b2) * precision * recall / (b2 * precision + recall) if (precision or recall) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision or recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": fbeta,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    y_sorted, s_sorted = y[order], s[order]
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    i = 0
    while i < len(y_sorted):
        j = i + 1
        while j < len(y_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        group = y_sorted[i:j]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return float(area)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, beta: float = 2.0) -> float:
    """Choose a recall-weighted threshold using validation labels only."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0, 1.0], s)))
    best = (float("-inf"), float("-inf"), 1.0)
    for threshold in candidates:
        m = precision_recall_fbeta(y, (s >= threshold).astype(np.int64), beta=beta)
        # Ties prefer higher precision, then the more conservative threshold.
        key = (m["f2"], m["precision"], float(threshold))
        if key > best:
            best = key
    return float(best[2])

