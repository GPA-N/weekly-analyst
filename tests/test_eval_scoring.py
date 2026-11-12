import pytest

from eval.scoring import (
    ac002_pass,
    bootstrap_ci,
    clopper_pearson,
    precision_at_k,
    recall_at_k,
)


def test_clopper_pearson_10_of_10_lower_bound_matches_reference():
    lower, _upper = clopper_pearson(10, 10)
    assert abs(lower - 0.6915) < 0.001


def test_clopper_pearson_0_of_10_lower_bound_is_zero():
    lower, _upper = clopper_pearson(0, 10)
    assert lower == 0.0


def test_recall_at_k_counts_hits_in_top_k_over_truth_size():
    assert recall_at_k(["a", "b"], ["a", "x", "b"], 2) == 0.5
    assert recall_at_k(["a", "b"], ["a", "x", "b"], 3) == 1.0


def test_precision_at_k_counts_hits_in_top_k_over_k():
    assert precision_at_k(["a", "b"], ["a", "x", "b"], 2) == 0.5
    assert precision_at_k(["a", "b"], ["a", "x", "b"], 3) == pytest.approx(2 / 3)


def test_ac002_rule_false_when_fewer_than_30_events():
    assert ac002_pass(29, 29) is False


def test_ac002_rule_true_for_30_of_30():
    assert ac002_pass(30, 30) is True


def test_ac002_rule_false_for_27_of_30():
    assert ac002_pass(27, 30) is False


def test_bootstrap_ci_is_deterministic_for_a_seed():
    values = [0.1, 0.4, 0.2, 0.9, 0.5, 0.7, 0.3, 0.6]
    first = bootstrap_ci(values, n_boot=1000, seed=42)
    second = bootstrap_ci(values, n_boot=1000, seed=42)
    assert first == second
    lo, hi = first
    assert lo <= hi
