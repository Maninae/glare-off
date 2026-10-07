"""Binary-classification metrics for the glasses gate: confusion matrix, per-class precision/recall, AUC, threshold choice.

Convention: label 1 = wears glasses (the app runs the glare model), 0 = bare face (the app skips it).
Pure numpy, no sklearn.

- Threshold choice protects glasses recall: a glasses face the gate drops loses its glare fix (visible
  failure), while a bare face let through only costs one glare-model run that should change nothing.
"""

from dataclasses import asdict, dataclass

import numpy as np

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


@dataclass
class BinaryClassificationReport:
    """Counts and rates at one threshold. Confusion matrix rows = true class, columns = predicted class, order [bare, glasses]."""

    threshold: float
    sample_count: int
    accuracy: float
    balanced_accuracy: float
    glasses_precision: float
    glasses_recall: float
    bare_precision: float
    bare_recall: float
    confusion_matrix_bare_glasses: list[list[int]]

    def to_dict(self) -> dict:
        """JSON-friendly dict."""
        return asdict(self)


def safe_ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, NaN when the denominator is 0 (e.g. precision of a class never predicted)."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


def compute_binary_classification_report(glasses_probabilities: np.ndarray, has_glasses_labels: np.ndarray, threshold: float) -> BinaryClassificationReport:
    """Return the report for predictions `probability >= threshold` -> glasses."""
    predicted_glasses = np.asarray(glasses_probabilities) >= threshold
    true_glasses = np.asarray(has_glasses_labels) >= 0.5
    true_positive = int(np.sum(predicted_glasses & true_glasses))
    false_positive = int(np.sum(predicted_glasses & ~true_glasses))
    false_negative = int(np.sum(~predicted_glasses & true_glasses))
    true_negative = int(np.sum(~predicted_glasses & ~true_glasses))
    glasses_recall = safe_ratio(true_positive, true_positive + false_negative)
    bare_recall = safe_ratio(true_negative, true_negative + false_positive)
    return BinaryClassificationReport(
        threshold=float(threshold),
        sample_count=int(len(true_glasses)),
        accuracy=safe_ratio(true_positive + true_negative, len(true_glasses)),
        balanced_accuracy=float(np.nanmean([glasses_recall, bare_recall])),
        glasses_precision=safe_ratio(true_positive, true_positive + false_positive),
        glasses_recall=glasses_recall,
        bare_precision=safe_ratio(true_negative, true_negative + false_negative),
        bare_recall=bare_recall,
        confusion_matrix_bare_glasses=[[true_negative, false_positive], [false_negative, true_positive]],
    )


def compute_roc_auc(glasses_probabilities: np.ndarray, has_glasses_labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count half); NaN if only one class is present."""
    scores = np.asarray(glasses_probabilities, dtype=np.float64)
    positives = np.asarray(has_glasses_labels) >= 0.5
    positive_count, negative_count = int(positives.sum()), int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    # Average ranks over tied scores.
    unique_scores, first_index, tie_counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    average_ranks = first_index + (tie_counts + 1) / 2.0
    ranks[order] = np.repeat(average_ranks, tie_counts)
    positive_rank_sum = ranks[positives].sum()
    return float((positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count))


def choose_threshold_for_glasses_recall(
    glasses_probabilities: np.ndarray, has_glasses_labels: np.ndarray, min_glasses_recall: float, default_threshold: float
) -> float:
    """Keep `default_threshold` if its glasses recall is >= `min_glasses_recall`; otherwise the highest grid threshold that is.

    Lowering only when the recall floor demands it avoids tuning to a near-separable val set (any
    threshold inside a clean val gap scores the same there, so moving off the default would be noise).
    Falls back to the lowest grid value if no threshold reaches the floor.
    """
    if compute_binary_classification_report(glasses_probabilities, has_glasses_labels, default_threshold).glasses_recall >= min_glasses_recall:
        return float(default_threshold)
    qualifying = [
        threshold
        for threshold in THRESHOLD_GRID
        if threshold < default_threshold
        and compute_binary_classification_report(glasses_probabilities, has_glasses_labels, threshold).glasses_recall >= min_glasses_recall
    ]
    return float(max(qualifying)) if qualifying else float(THRESHOLD_GRID[0])
