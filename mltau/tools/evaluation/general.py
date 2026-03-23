import numpy as np
import boost_histogram as bh


def binary_classifier_metrics(
    predictions: np.array, targets: np.array, threshold: float
) -> dict:
    """Compute scalar classification metrics for a binary classifier at a given threshold.

    Args:
        predictions: 1-D array of predicted scores.
        targets: 1-D array of ground-truth binary labels (0 or 1).
        threshold: Decision threshold; scores > threshold are classified as positive.

    Returns:
        Dict with TPR, TNR, FPR, FNR, precision, recall, F1, and accuracy.
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    tp = float(np.sum((predictions > threshold) & (targets == 1)))
    fn = float(np.sum((predictions <= threshold) & (targets == 1)))
    tn = float(np.sum((predictions <= threshold) & (targets == 0)))
    fp = float(np.sum((predictions > threshold) & (targets == 0)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "TPR": tpr,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "accuracy": accuracy,
    }


def calculate_bin_centers(edges: np.array) -> np.array:
    bin_widths = np.array([edges[i + 1] - edges[i] for i in range(len(edges) - 1)])
    bin_centers = []
    for i in range(len(edges) - 1):
        bin_centers.append(edges[i] + (bin_widths[i] / 2))
    return np.array(bin_centers), bin_widths / 2


def to_bh(data, bins, cumulative=False):
    h1 = bh.Histogram(bh.axis.Variable(bins))
    h1.fill(data)
    if cumulative:
        h1[:] = np.sum(h1.values()) - np.cumsum(h1)
    return h1
