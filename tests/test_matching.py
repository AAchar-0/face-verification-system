import numpy as np
import pytest

from matching import average_template, cosine_similarity, decide, l2_normalize


def test_normalised_vector_has_length_one():
    assert np.linalg.norm(l2_normalize([3, 4])) == pytest.approx(1.0)


def test_zero_vector_is_rejected():
    with pytest.raises(ValueError):
        l2_normalize([0, 0, 0])


def test_cosine_similarity_known_values():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_ignores_scale():
    assert cosine_similarity([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_average_template_has_length_one_and_lies_between_inputs():
    template = average_template([[1, 0], [0, 1]])
    assert np.linalg.norm(template) == pytest.approx(1.0)
    assert template[0] == pytest.approx(template[1])


def test_average_template_needs_input():
    with pytest.raises(ValueError):
        average_template([])


def test_decision_accepts_when_median_reaches_threshold():
    template = [1.0, 0.0]
    probes = [[1.0, 0.05], [1.0, 0.1], [1.0, 0.0]]
    decision = decide(template, probes, threshold=0.9)
    assert decision.accepted and decision.reason == "ok" and decision.frames_used == 3


def test_decision_rejects_a_different_person():
    decision = decide([1.0, 0.0], [[0.0, 1.0]] * 4, threshold=0.5)
    assert not decision.accepted and decision.score == pytest.approx(0.0)


def test_decision_needs_enough_frames():
    decision = decide([1.0, 0.0], [[1.0, 0.0]] * 2, threshold=0.5, min_frames=3)
    assert not decision.accepted and decision.reason == "not_enough_frames" and decision.score is None


def test_one_bad_frame_does_not_change_the_median_decision():
    probes = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]      # three good frames, one outlier
    assert decide([1.0, 0.0], probes, threshold=0.9).accepted
