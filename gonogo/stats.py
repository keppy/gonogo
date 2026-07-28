"""Statistics for small case sets.

Pilot evaluations run on 40-100 real cases, not 10,000. At that size the point
estimate is not the answer -- the interval is. Everything here exists to keep
the reported numbers honest at that scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z for a two-sided normal interval, by confidence level.
_Z = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def z_for(level: float) -> float:
    """z value for a two-sided interval at `level`."""
    if level in _Z:
        return _Z[level]
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence level must be in (0, 1), got {level}")
    # Acklam-style rational approximation to the normal quantile, plenty
    # accurate for reporting intervals.
    p = (1.0 + level) / 2.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


@dataclass(frozen=True)
class Interval:
    """A proportion with a confidence interval."""

    point: float
    low: float
    high: float
    level: float
    n: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sanely at small n and at proportions near 0 or 1 -- exactly the
    regime a pilot lives in.
    """
    if n < 0 or successes < 0:
        raise ValueError("successes and n must be non-negative")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")
    if n == 0:
        return Interval(point=float("nan"), low=0.0, high=1.0, level=level, n=0)

    z = z_for(level)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(point=p, low=max(0.0, center - half), high=min(1.0, center + half), level=level, n=n)


def required_n(observed_rate: float, target: float, level: float = 0.95, max_n: int = 100_000) -> int | None:
    """Smallest n at which `observed_rate` would clear `target` as a lower bound.

    Answers "how many more cases do I need before I can claim this?" Returns
    None if the observed rate is at or below the target, where no amount of
    additional data would establish the claim.
    """
    if not 0.0 <= observed_rate <= 1.0:
        raise ValueError("observed_rate must be in [0, 1]")
    if observed_rate <= target:
        return None
    lo, hi = 1, 1
    while hi <= max_n:
        if wilson(round(observed_rate * hi), hi, level).low >= target:
            break
        lo, hi = hi, hi * 2
    else:
        return None
    # The predicate is not perfectly monotone in n because round(rate * n)
    # jitters between adjacent n, so the binary search can land a few cases
    # off the true minimum. Fine for its purpose: "roughly how many more
    # cases", not a sample-size guarantee.
    while lo < hi:  # binary search the crossing point
        mid = (lo + hi) // 2
        if wilson(round(observed_rate * mid), mid, level).low >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass(frozen=True)
class OperatingPoint:
    """What you get if the agent abstains below `threshold`."""

    threshold: float
    coverage: float           # fraction of cases the agent still answers
    n_covered: int
    precision: Interval       # accuracy among answered cases
    n_deferred: int           # cases routed to a human

    def __str__(self) -> str:
        return (f"abstain below {self.threshold:.2f} -> {self.coverage:.0%} coverage "
                f"at {self.precision.point:.1%} precision, {self.n_deferred} to review")


def risk_coverage(results: list[tuple[float, bool]], level: float = 0.95) -> list[OperatingPoint]:
    """Risk-coverage curve over abstention thresholds.

    `results` is (confidence, passed) per case. Returns one OperatingPoint per
    distinct confidence value, from full coverage to most selective. This is the
    curve that answers the real question: not "how accurate is it" but "how much
    can I automate, at what precision, with the rest going to a human".
    """
    if not results:
        return []
    for conf, _ in results:
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {conf}")

    n = len(results)
    points: list[OperatingPoint] = []
    for threshold in sorted({c for c, _ in results}):
        covered = [passed for conf, passed in results if conf >= threshold]
        if not covered:
            continue
        points.append(OperatingPoint(
            threshold=threshold,
            coverage=len(covered) / n,
            n_covered=len(covered),
            precision=wilson(sum(covered), len(covered), level),
            n_deferred=n - len(covered),
        ))
    return points


def best_operating_point(
    results: list[tuple[float, bool]],
    target_precision: float,
    min_coverage: float = 0.0,
    level: float = 0.95,
) -> OperatingPoint | None:
    """Highest-coverage threshold whose precision *lower bound* clears the target.

    Uses the lower bound rather than the point estimate on purpose: a threshold
    that hits 100% on 3 cases is not a defensible operating point.
    """
    candidates = [
        p for p in risk_coverage(results, level)
        if p.precision.low >= target_precision and p.coverage >= min_coverage
    ]
    return max(candidates, key=lambda p: p.coverage) if candidates else None


def expected_calibration_error(results: list[tuple[float, bool]], bins: int = 10) -> float:
    """Expected calibration error: does stated confidence match observed accuracy?

    A well-calibrated 0.9 means right about 90% of the time. Poor calibration is
    what makes an abstention threshold meaningless, so this gates whether the
    operating point above can be trusted at all.
    """
    if not results:
        return float("nan")
    if bins < 1:
        raise ValueError("bins must be >= 1")

    n = len(results)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        # Include the right edge in the last bin so confidence 1.0 is counted.
        bucket = [(c, p) for c, p in results if (lo <= c < hi or (b == bins - 1 and c == hi))]
        if not bucket:
            continue
        acc = sum(1 for _, p in bucket if p) / len(bucket)
        conf = sum(c for c, _ in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(acc - conf)
    return total


def reliability_table(results: list[tuple[float, bool]], bins: int = 5) -> list[dict]:
    """Per-bin confidence vs accuracy, for showing calibration in a report."""
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, p) for c, p in results if (lo <= c < hi or (b == bins - 1 and c == hi))]
        if not bucket:
            continue
        rows.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": len(bucket),
            "mean_confidence": sum(c for c, _ in bucket) / len(bucket),
            "accuracy": sum(1 for _, p in bucket if p) / len(bucket),
        })
    return rows
