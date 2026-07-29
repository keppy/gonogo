"""Head-to-head comparison of two agents on the same cases.

Swapping in a new model and watching the pass rate rise is the most common way
a team convinces itself of an improvement that isn't there. Two independent
intervals that overlap tell you very little, and two that don't overlap is a
cruder test than the data supports.

When both agents ran the *same* cases, the results are paired, and the paired
test is strictly more powerful: a case both agents got right carries no
information about which is better, so it should not be in the denominator.
McNemar's test throws those away and looks only at the disagreements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .stats import z_for

# Above this many disagreements the exact binomial sum gets slow and the
# chi-square approximation is indistinguishable from it anyway.
_EXACT_LIMIT = 1000


@dataclass(frozen=True)
class Comparison:
    """Whether two agents actually differ on the cases they share."""

    n_shared: int
    both_passed: int
    both_failed: int
    only_a: int              # a passed, b failed
    only_b: int              # b passed, a failed
    rate_a: float
    rate_b: float
    difference: float        # rate_b - rate_a
    low: float               # interval on the difference
    high: float
    level: float
    p_value: float
    exact: bool              # False when the chi-square approximation was used

    @property
    def significant(self) -> bool:
        return self.p_value < (1.0 - self.level)

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b

    def __str__(self) -> str:
        verdict = "distinguishable" if self.significant else "not distinguishable"
        return (f"{self.rate_b:.1%} vs {self.rate_a:.1%} "
                f"({self.difference:+.1%} [{self.low:+.1%}, {self.high:+.1%}], "
                f"p={self.p_value:.3f}) — {verdict} on {self.n_shared} shared cases")

    def summary(self) -> str:
        """A few lines suitable for dropping into a report."""
        lines = [
            f"Shared cases        {self.n_shared}",
            f"Agent A pass rate   {self.rate_a:.1%}",
            f"Agent B pass rate   {self.rate_b:.1%}",
            f"Difference          {self.difference:+.1%} "
            f"[{self.low:+.1%}, {self.high:+.1%}] at {self.level:.0%}",
            f"Disagreements       {self.discordant} "
            f"({self.only_a} only-A, {self.only_b} only-B)",
            f"McNemar p           {self.p_value:.4f}"
            f"{'' if self.exact else ' (chi-square approximation)'}",
        ]
        if self.significant:
            better = "B" if self.difference > 0 else "A"
            lines.append(f"Verdict             agent {better} is better on these cases")
        else:
            lines.append("Verdict             no detectable difference; "
                         "the sample cannot separate them")
        return "\n".join(lines)


def outcomes(report: Any) -> dict[str, bool]:
    """Pull `{case_id: passed}` out of a Report or a `Report.to_dict()` payload.

    Accepting the serialized form matters: comparing two runs usually means
    comparing something measured today against a JSON file written last week,
    not two live objects.
    """
    cases = report.get("cases") if isinstance(report, Mapping) else getattr(report, "results", None)
    if cases is None:
        raise ValueError("expected a Report or a Report.to_dict() payload with per-case results")

    out: dict[str, bool] = {}
    for entry in cases:
        if isinstance(entry, Mapping):
            case_id, passed = entry.get("id"), entry.get("passed")
        else:  # a CaseResult
            case_id, passed = entry.case.id, entry.passed
        if case_id is None:
            raise ValueError("every case needs an id to pair on")
        if case_id in out:
            raise ValueError(f"duplicate case id {case_id!r}; ids must be unique to pair on")
        out[str(case_id)] = bool(passed)
    return out


def compare(report_a: Any, report_b: Any, level: float = 0.95) -> Comparison:
    """McNemar's test on the cases two agents share.

    Pairs by case id, ignores any case only one of them ran, and reports whether
    the difference in pass rate is bigger than the disagreement pattern would
    produce by chance.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    a, b = outcomes(report_a), outcomes(report_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two reports share no case ids; nothing to compare")

    both_passed = sum(1 for i in shared if a[i] and b[i])
    both_failed = sum(1 for i in shared if not a[i] and not b[i])
    only_a = sum(1 for i in shared if a[i] and not b[i])
    only_b = sum(1 for i in shared if b[i] and not a[i])

    n = len(shared)
    rate_a = (both_passed + only_a) / n
    rate_b = (both_passed + only_b) / n
    difference = rate_b - rate_a

    p_value, exact = _mcnemar(only_a, only_b)

    # Wald interval on the paired difference. Only the discordant pairs carry
    # information, so the concordant ones drop out of the variance.
    discordant = only_a + only_b
    if discordant == 0:
        low = high = 0.0
    else:
        z = z_for(level)
        var = (discordant - (only_b - only_a) ** 2 / n) / (n * n)
        half = z * math.sqrt(max(0.0, var))
        low, high = max(-1.0, difference - half), min(1.0, difference + half)

    return Comparison(
        n_shared=n, both_passed=both_passed, both_failed=both_failed,
        only_a=only_a, only_b=only_b, rate_a=rate_a, rate_b=rate_b,
        difference=difference, low=low, high=high, level=level,
        p_value=p_value, exact=exact,
    )


def _mcnemar(only_a: int, only_b: int) -> tuple[float, bool]:
    """Two-sided McNemar p-value from the two disagreement counts."""
    n = only_a + only_b
    if n == 0:
        # The agents agreed on every case; there is nothing to test.
        return 1.0, True
    if n <= _EXACT_LIMIT:
        k = max(only_a, only_b)
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
        return min(1.0, 2 * tail), True
    # Chi-square with continuity correction, via the normal survival function.
    chi = (abs(only_a - only_b) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2)), False
