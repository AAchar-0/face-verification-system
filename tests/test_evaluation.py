import numpy as np
import pytest

from evaluation import (accuracy_at, cross_validate, equal_error_rate, error_rates, image_relpath,
                        parse_pairs, summarise, threshold_for_far, wilson_interval)

PAIRS_TEXT = """2\t2
Alice_Smith\t1\t2
Alice_Smith\t2\t3
Alice_Smith\t1\tBob_Jones\t1
Carol_White\t4\tBob_Jones\t2
Dave_Lee\t1\t2
Dave_Lee\t2\t3
Dave_Lee\t1\tAlice_Smith\t3
Eve_Ray\t1\tBob_Jones\t3
"""


def test_image_path_format():
    assert image_relpath("Alice_Smith", 7) == "Alice_Smith/Alice_Smith_0007.jpg"


def test_pairs_are_parsed_by_fold_and_kind():
    pairs = parse_pairs(PAIRS_TEXT)
    assert len(pairs) == 8
    assert [p.fold for p in pairs] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [p.same for p in pairs] == [True, True, False, False, True, True, False, False]
    assert pairs[0].first == "Alice_Smith/Alice_Smith_0001.jpg"
    assert pairs[2].second == "Bob_Jones/Bob_Jones_0001.jpg"


def test_error_rates_on_a_small_example():
    same = [0.9, 0.8, 0.4, 0.3]            # two of the four are below 0.5
    different = [0.6, 0.2, 0.1, 0.0]       # one of the four is at or above 0.5
    far, frr = error_rates(same, different, 0.5)
    assert far[0] == pytest.approx(0.25) and frr[0] == pytest.approx(0.5)


def test_threshold_equal_to_a_score_accepts_that_score():
    far, _ = error_rates([1.0], [0.5], 0.5)
    assert far[0] == 1.0


def test_eer_is_zero_when_scores_separate_perfectly():
    eer, _ = equal_error_rate([0.8, 0.9, 0.7], [0.1, 0.2, 0.3])
    assert eer == pytest.approx(0.0)


def test_eer_is_near_half_when_scores_overlap_completely():
    rng = np.random.default_rng(1)
    eer, _ = equal_error_rate(rng.normal(size=4000), rng.normal(size=4000))
    assert 0.45 < eer < 0.55


def test_threshold_for_far_limits_false_accepts():
    different = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold = threshold_for_far(different, 0.2)             # at most one of five may be accepted
    far, _ = error_rates([0.9], different, threshold)
    assert far[0] == pytest.approx(0.2)
    zero = threshold_for_far(different, 0.0)                   # none may be accepted
    far_zero, _ = error_rates([0.9], different, zero)
    assert far_zero[0] == 0.0


def test_threshold_for_far_never_exceeds_the_target_even_with_ties():
    different = [0.5, 0.5, 0.5, 0.2]
    threshold = threshold_for_far(different, 0.25)
    far, _ = error_rates([0.9], different, threshold)
    assert far[0] <= 0.25


def test_accuracy_at_a_threshold():
    assert accuracy_at([0.9, 0.8], [0.1, 0.2], 0.5) == pytest.approx(1.0)
    assert accuracy_at([0.9, 0.4], [0.6, 0.2], 0.5) == pytest.approx(0.5)


def test_wilson_interval_with_zero_events():
    low, high = wilson_interval(0, 3000)
    assert low == 0.0
    assert high == pytest.approx(3.8416 / (3000 + 3.8416), rel=1e-3)     # about 0.13%


def test_wilson_interval_contains_the_observed_rate():
    low, high = wilson_interval(30, 1000)
    assert low < 0.03 < high


def test_summarise():
    mean, sd = summarise([1, 2, 3])
    assert mean == pytest.approx(2.0) and sd == pytest.approx(1.0)


def test_cross_validation_on_perfectly_separable_folds():
    rng = np.random.default_rng(0)
    folds = [(rng.uniform(0.6, 1.0, 50), rng.uniform(-0.2, 0.4, 50)) for _ in range(10)]
    result = cross_validate(folds, target_far=0.01)
    assert max(result["frr"]) == 0                      # the threshold sits well below every same-person score
    assert min(result["accuracy"]) == 1.0               # the best-accuracy threshold separates the groups perfectly
    # A 1% target deliberately lets a few different-person scores through, so FAR is small but not always 0.
    assert np.mean(result["far"]) < 0.05


def test_wilson_interval_is_exact_at_the_extremes():
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0
