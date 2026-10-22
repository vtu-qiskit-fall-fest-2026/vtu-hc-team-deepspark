"""Small, dependency-free statistics helpers (binomial proportions)."""
from __future__ import annotations
import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (default 95%)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def improvement_p_value(k_base: int, n_base: int, k_new: int, n_new: int) -> float:
    """One-sided two-proportion z-test.

    H0: failure rates are equal.  H1: the new strategy has a LOWER failure rate
    than the baseline.  Returns the p-value (small => evidence of improvement).
    If both counts are identical/zero the test is uninformative and 0.5 is returned.
    """
    if n_base <= 0 or n_new <= 0:
        return 1.0
    p_b, p_n = k_base / n_base, k_new / n_new
    pooled = (k_base + k_new) / (n_base + n_new)
    var = pooled * (1 - pooled) * (1 / n_base + 1 / n_new)
    if var <= 0:
        return 0.5
    return normal_sf((p_b - p_n) / math.sqrt(var))


def smoothed_variance(k: int, n: int) -> float:
    """Variance of a proportion with a +0.5 continuity smoothing (never exactly 0)."""
    if n <= 0:
        return 1.0
    p = (k + 0.5) / (n + 1)
    return p * (1 - p) / n
