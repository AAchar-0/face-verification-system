"""Accuracy measurement for a face verification system. Pure functions: no camera, no OpenCV.

Two kinds of mistake matter, and they trade off against each other as the decision threshold moves:
  FAR (False Accept Rate): how often two DIFFERENT people are wrongly accepted. The security error.
  FRR (False Reject Rate): how often the SAME person is wrongly rejected. The inconvenience error.
EER (Equal Error Rate) is the point where the two are equal: a one-number summary of accuracy.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pair:
    fold: int            # which of the 10 groups the pair belongs to (0 to 9)
    same: bool           # True if both photos show the same person
    first: str           # image path relative to the lfw folder, e.g. Aaron_Peirsol/Aaron_Peirsol_0001.jpg
    second: str


def image_relpath(name: str, index: int) -> str:
    return f"{name}/{name}_{index:04d}.jpg"


def parse_pairs(text: str) -> list[Pair]:
    """Read the official LFW pairs.txt: 10 folds, each with N matching pairs then N non-matching pairs."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    folds, per_kind = (int(value) for value in lines[0].split())
    pairs, position = [], 1
    for fold in range(folds):
        for _ in range(per_kind):                                    # name, photo number, photo number
            name, a, b = lines[position].split()
            position += 1
            pairs.append(Pair(fold, True, image_relpath(name, int(a)), image_relpath(name, int(b))))
        for _ in range(per_kind):                                    # name, photo number, name, photo number
            name1, a, name2, b = lines[position].split()
            position += 1
            pairs.append(Pair(fold, False, image_relpath(name1, int(a)), image_relpath(name2, int(b))))
    return pairs


def error_rates(same_scores, different_scores, thresholds):
    """FAR and FRR at each threshold. A pair is ACCEPTED when its score is at or above the threshold."""
    same = np.sort(np.asarray(same_scores, dtype=float))
    different = np.sort(np.asarray(different_scores, dtype=float))
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    frr = np.searchsorted(same, thresholds, side="left") / len(same)                        # same-person pairs below
    far = (len(different) - np.searchsorted(different, thresholds, side="left")) / len(different)   # different-person pairs at or above
    return far, frr


def equal_error_rate(same_scores, different_scores) -> tuple[float, float]:
    """The error rate where FAR equals FRR, and the threshold that gives it."""
    thresholds = np.unique(np.concatenate([same_scores, different_scores]))
    far, frr = error_rates(same_scores, different_scores, thresholds)
    best = int(np.argmin(np.abs(far - frr)))
    return float((far[best] + frr[best]) / 2), float(thresholds[best])


def threshold_for_far(different_scores, target_far: float) -> float:
    """The lowest threshold that lets through at most `target_far` of different-person pairs."""
    ranked = np.sort(np.asarray(different_scores, dtype=float))[::-1]        # highest score first
    allowed = int(math.floor(target_far * len(ranked)))                      # how many false accepts we may have
    if allowed >= len(ranked):
        return float(ranked[-1])
    return float(np.nextafter(ranked[allowed], np.inf))                      # just above the first score that must be rejected


def accuracy_at(same_scores, different_scores, threshold: float) -> float:
    far, frr = error_rates(same_scores, different_scores, threshold)
    n_same, n_different = len(same_scores), len(different_scores)
    correct = (1 - frr[0]) * n_same + (1 - far[0]) * n_different
    return float(correct / (n_same + n_different))


def best_accuracy_threshold(same_scores, different_scores) -> float:
    """The threshold with the highest accuracy, placed in the MIDDLE of the best range.

    Candidates are midpoints between neighbouring scores. If several candidates tie for best, the
    middle one is used, so the threshold does not sit on the edge of the training data.
    """
    scores = np.unique(np.concatenate([same_scores, different_scores]))
    candidates = np.concatenate([[scores[0] - 1e-9], (scores[:-1] + scores[1:]) / 2, [scores[-1] + 1e-9]])
    far, frr = error_rates(same_scores, different_scores, candidates)
    n_same, n_different = len(same_scores), len(different_scores)
    accuracy = ((1 - frr) * n_same + (1 - far) * n_different) / (n_same + n_different)
    best = np.flatnonzero(accuracy == accuracy.max())
    return float(candidates[best[len(best) // 2]])


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% confidence range for a rate such as FAR. It stays sensible even when the count is 0."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)          # exact 0 and 1 at the extremes,
    high = 1.0 if successes == total else min(1.0, centre + half)     # not rounding noise such as 1e-19
    return low, high


def summarise(values) -> tuple[float, float]:
    """Mean and sample standard deviation."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def cross_validate(folds, target_far: float) -> dict:
    """The standard LFW protocol: choose thresholds on 9 folds, measure on the held-out 10th.

    `folds` is a list of (same_scores, different_scores), one per fold.
    """
    far_list, frr_list, accuracy_list = [], [], []
    for held_out in range(len(folds)):
        train_same = np.concatenate([s for k, (s, _) in enumerate(folds) if k != held_out])
        train_diff = np.concatenate([d for k, (_, d) in enumerate(folds) if k != held_out])
        test_same, test_diff = folds[held_out]

        threshold = threshold_for_far(train_diff, target_far)
        far, frr = error_rates(test_same, test_diff, threshold)
        far_list.append(float(far[0]))
        frr_list.append(float(frr[0]))

        accuracy_threshold = best_accuracy_threshold(train_same, train_diff)
        accuracy_list.append(accuracy_at(test_same, test_diff, accuracy_threshold))
    return {"far": far_list, "frr": frr_list, "accuracy": accuracy_list}
