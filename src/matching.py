"""Matching maths for face verification. Pure functions: no camera, no files, easy to test.

A face is turned into an "embedding": a list of 128 numbers. Two photos of the same person give
embeddings that point in nearly the same direction. We measure that with cosine similarity:
1.0 means the same direction, 0 means unrelated, -1 means opposite.
"""
from dataclasses import dataclass

import numpy as np


def l2_normalize(vector) -> np.ndarray:
    """Scale a vector to length 1, so that only its direction matters."""
    flat = np.asarray(vector, dtype=np.float64).ravel()
    length = np.linalg.norm(flat)
    if length == 0:
        raise ValueError("Cannot normalise a vector of zeros")
    return flat / length


def cosine_similarity(first, second) -> float:
    """Similarity between two embeddings, from -1 to 1 (higher means more alike)."""
    return float(np.dot(l2_normalize(first), l2_normalize(second)))


def average_template(embeddings) -> np.ndarray:
    """Combine several embeddings of one person into one steadier template.

    Each embedding is scaled to length 1, averaged, and scaled again, so a single
    unusual frame (odd light, odd angle) has little influence.
    """
    if len(embeddings) == 0:
        raise ValueError("Need at least one embedding")
    stacked = np.stack([l2_normalize(e) for e in embeddings])
    return l2_normalize(stacked.mean(axis=0))


@dataclass(frozen=True)
class Decision:
    accepted: bool
    score: float | None        # median similarity over the frames, or None if too few frames
    frames_used: int
    reason: str                # "ok" or "not_enough_frames"


def decide(template, probes, threshold: float, min_frames: int = 3) -> Decision:
    """Accept if the MEDIAN similarity of several camera frames reaches the threshold.

    The median ignores a single bad frame, and requiring several frames stops one lucky
    frame from being enough.
    """
    scores = [cosine_similarity(template, probe) for probe in probes]
    if len(scores) < min_frames:
        return Decision(False, None, len(scores), "not_enough_frames")
    score = float(np.median(scores))
    return Decision(score >= threshold, score, len(scores), "ok")
