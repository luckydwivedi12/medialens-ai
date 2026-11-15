"""Binary classification metrics with fake as positive and no extra dependency."""

import math
from collections.abc import Sequence
from typing import Any


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum, index = 0.0, 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], loss: float,
                    threshold: float = .5) -> dict[str, Any]:
    if not labels or len(labels) != len(probabilities) or any(y not in (0, 1) for y in labels):
        raise ValueError('Expected equally sized nonempty binary labels and probabilities')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Probabilities must be finite and in [0,1]')
    if not math.isfinite(loss) or loss < 0 or not 0 < threshold < 1:
        raise ValueError('Invalid loss or decision threshold')
    predicted = [int(p >= threshold) for p in probabilities]
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'sample_count': len(labels), 'loss': loss, 'accuracy': (tp + tn) / len(labels),
        'precision': precision, 'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'specificity': tn / (tn + fp) if tn + fp else None,
        'roc_auc': roc_auc(labels, probabilities), 'confusion_matrix': [[tn, fp], [fn, tp]],
        'confusion_matrix_order': 'rows=true [real,fake], columns=predicted [real,fake]',
        'positive_class': 'fake', 'decision_threshold': threshold,
        'interpretation': 'Crop-level development metrics against provisional paired-region video labels.'}
