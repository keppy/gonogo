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

from .stats import wilson

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
    unit: str = "cases"

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
                f"p={self.p_value:.3f}) — {verdict} on {self.n_shared} shared {self.unit}")

    def summary(self) -> str:
        """A few lines suitable for dropping into a report."""
        lines = [
            f"Shared {self.unit:<12} {self.n_shared}",
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
            lines.append(f"Verdict             agent {better} is better on these {self.unit}")
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
        if not isinstance(passed, bool):
            raise ValueError(f"case {case_id!r}: passed must be a boolean")
        if str(case_id) in out:
            raise ValueError(f"duplicate case id {case_id!r}; ids must be unique to pair on")
        out[str(case_id)] = passed
    return out

def _trial_outcomes(report: Any) -> tuple[dict[str, bool], dict[str, set[str]] | None]:
    """Fold grouped runs to independent group trials, preserving membership."""
    cases = report.get("cases") if isinstance(report, Mapping) else getattr(report, "results", None)
    if cases is None:
        raise ValueError("expected a Report or a Report.to_dict() payload with per-case results")
    entries = list(cases)
    explicit = (report.get("n_groups") is not None if isinstance(report, Mapping) and "n_groups" in report
                else getattr(getattr(report, "decision", None), "n_groups", None) is not None)
    # Raw JSONL payloads have no n_groups marker; group on every row means grouped.
    implicit = isinstance(report, Mapping) and "n_groups" not in report and any(
        e.get("group") is not None for e in entries if isinstance(e, Mapping))
    if not (explicit or implicit):
        return outcomes(report), None
    passed_by_group: dict[str, bool] = {}
    membership: dict[str, set[str]] = {}
    for entry in entries:
        if isinstance(entry, Mapping):
            group, cid, passed = entry.get("group"), entry.get("id"), entry.get("passed")
        else:
            group, cid, passed = entry.case.group, entry.case.id, entry.passed
        if group is None or cid is None:
            raise ValueError("grouped comparison requires a group and id on every case")
        if not isinstance(passed, bool):
            raise ValueError(f"case {cid!r}: passed must be a boolean")
        key = str(group)
        ids = membership.setdefault(key, set())
        if str(cid) in ids:
            raise ValueError(f"duplicate case id {cid!r} within group {key!r}")
        ids.add(str(cid))
        passed_by_group[key] = passed_by_group.get(key, True) and passed
    return passed_by_group, membership


def compare(report_a: Any, report_b: Any, level: float = 0.95) -> Comparison:
    """McNemar's test on shared case or group trials.

    Ungrouped runs pair by case id. Grouped runs pair by group id, require
    matching case membership within each shared group, and pass a group only
    when all its cases pass. Unmatched trials are excluded. The difference
    interval uses simultaneous Wilson bounds, not a boundary-degenerate Wald CI.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")

    a, groups_a = _trial_outcomes(report_a)
    b, groups_b = _trial_outcomes(report_b)
    if (groups_a is None) != (groups_b is None):
        raise ValueError("cannot compare grouped trials with ungrouped cases")
    unit = "groups" if groups_a is not None else "cases"
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(f"the two reports share no {unit[:-1]} ids; nothing to compare")
    if groups_a is not None and groups_b is not None:
        for group in shared:
            if groups_a[group] != groups_b[group]:
                raise ValueError(f"group {group!r} has different case membership across runs")

    both_passed = sum(1 for i in shared if a[i] and b[i])
    both_failed = sum(1 for i in shared if not a[i] and not b[i])
    only_a = sum(1 for i in shared if a[i] and not b[i])
    only_b = sum(1 for i in shared if b[i] and not a[i])

    n = len(shared)
    rate_a = (both_passed + only_a) / n
    rate_b = (both_passed + only_b) / n
    difference = rate_b - rate_a

    p_value, exact = _mcnemar(only_a, only_b)

    # Paired difference is P(B-only) - P(A-only). Simultaneous Wilson
    # intervals (Bonferroni) for those two probabilities give a conservative
    # interval that does not collapse to width zero at boundary outcomes.
    component_level = 1.0 - (1.0 - level) / 2.0
    a_only = wilson(only_a, n, component_level)
    b_only = wilson(only_b, n, component_level)
    low = max(-1.0, b_only.low - a_only.high)
    high = min(1.0, b_only.high - a_only.low)

    return Comparison(
        n_shared=n, both_passed=both_passed, both_failed=both_failed,
        only_a=only_a, only_b=only_b, rate_a=rate_a, rate_b=rate_b,
        difference=difference, low=low, high=high, level=level,
        p_value=p_value, exact=exact, unit=unit,
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
