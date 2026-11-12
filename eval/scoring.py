"""Scoring helpers for the eval harness (spec 7.3)."""

import numpy as np
from scipy.stats import beta


def recall_at_k(truth: list, alerts: list, k: int) -> float:
    truth_set = set(truth)
    if not truth_set:
        raise ValueError("truth must contain at least one event")
    return len(set(alerts[:k]) & truth_set) / len(truth_set)


def precision_at_k(truth: list, alerts: list, k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    return len(set(alerts[:k]) & set(truth)) / k


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("need n >= 1 and 0 <= successes <= n")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, data.size, size=(n_boot, data.size))
    means = data[resample_idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def ac002_pass(successes: int, n: int) -> bool:
    return n >= 30 and clopper_pearson(successes, n)[0] >= 0.80
