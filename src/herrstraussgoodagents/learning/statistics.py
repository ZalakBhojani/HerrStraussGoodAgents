"""Bootstrap confidence interval for comparing prompt candidates.

Pure Python implementation (stdlib only).  Compares baseline vs candidate
fitness score distributions and recommends adoption when the 95% CI
lower bound of the improvement is > 0.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of a bootstrap CI comparison."""

    effect_size: float  # mean(candidate) - mean(baseline)
    ci_lower: float  # lower bound of confidence interval
    ci_upper: float  # upper bound of confidence interval
    p_value: float  # approximate p-value
    recommend_adoption: bool  # True if ci_lower > 0
    n_baseline: int
    n_candidate: int
    n_resamples: int


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _resample_mean(xs: list[float], rng: random.Random) -> float:
    """Draw len(xs) samples with replacement and return the mean."""
    n = len(xs)
    return sum(xs[rng.randint(0, n - 1)] for _ in range(n)) / n


def bootstrap_ci(
    baseline: list[float],
    candidate: list[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compare baseline and candidate fitness scores using bootstrap CI.

    Args:
        baseline: Fitness scores from the current prompt version.
        candidate: Fitness scores from the mutated prompt version.
        n_resamples: Number of bootstrap resamples (default 10,000).
        confidence: Confidence level for the interval (default 0.95).
        seed: RNG seed for reproducibility.

    Returns:
        BootstrapResult with effect size, CI bounds, p-value, and
        adoption recommendation.

    Raises:
        ValueError: If either list has fewer than 2 scores.
    """
    if len(baseline) < 2 or len(candidate) < 2:
        raise ValueError(
            f"Need at least 2 scores per group (got {len(baseline)} baseline, "
            f"{len(candidate)} candidate)"
        )

    rng = random.Random(seed)

    observed_diff = _mean(candidate) - _mean(baseline)

    # Bootstrap: resample each group independently, compute difference of means
    diffs: list[float] = []
    for _ in range(n_resamples):
        b_mean = _resample_mean(baseline, rng)
        c_mean = _resample_mean(candidate, rng)
        diffs.append(c_mean - b_mean)

    diffs.sort()

    # Confidence interval from percentiles
    alpha = 1.0 - confidence
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1.0 - alpha / 2)) - 1
    ci_lower = diffs[lo_idx]
    ci_upper = diffs[hi_idx]

    # Approximate p-value: proportion of resampled diffs <= 0
    n_leq_zero = sum(1 for d in diffs if d <= 0)
    p_value = n_leq_zero / n_resamples

    recommend = ci_lower > 0

    logger.info(
        "Bootstrap CI: effect=%.4f, CI=[%.4f, %.4f], p=%.4f, adopt=%s "
        "(n_base=%d, n_cand=%d, resamples=%d)",
        observed_diff,
        ci_lower,
        ci_upper,
        p_value,
        recommend,
        len(baseline),
        len(candidate),
        n_resamples,
    )

    return BootstrapResult(
        effect_size=round(observed_diff, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        recommend_adoption=recommend,
        n_baseline=len(baseline),
        n_candidate=len(candidate),
        n_resamples=n_resamples,
    )
