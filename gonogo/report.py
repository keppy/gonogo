"""The score report.

Written for the person who signs off on the pilot, not for the engineer who
built it. Plain language, the interval next to every rate, and the decision
stated in the first three lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cases import CaseResult
from .decide import Decision, Verdict
from .stats import OperatingPoint, reliability_table, risk_coverage

_MARK = {
    Verdict.AUTOMATE: "Ship it.",
    Verdict.AUTOMATE_WITH_REVIEW: "Ship it behind a confidence threshold.",
    Verdict.ASSIST_ONLY: "Use it to draft, keep a human on every case.",
    Verdict.DO_NOT_AUTOMATE: "Don't automate this.",
    Verdict.INSUFFICIENT_EVIDENCE: "Not enough cases to decide yet.",
}


@dataclass
class Report:
    task: str
    results: list[CaseResult]
    decision: Decision
    metadata: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def errors(self) -> list[CaseResult]:
        return [r for r in self.results if r.prediction.error]

    def failures(self, limit: int | None = None) -> list[CaseResult]:
        """Failed cases, worst score first -- the list to actually go read."""
        worst = sorted((r for r in self.results if not r.passed), key=lambda r: r.score)
        return worst[:limit] if limit else worst

    def curve(self) -> list[OperatingPoint]:
        return risk_coverage([(r.confidence, r.passed) for r in self.results])

    def summary(self) -> str:
        """One line, for a terminal or a commit message."""
        d = self.decision
        return f"{self.task}: {d.verdict.value} ({d.pass_rate} on {self.n} cases)"

    def markdown(self, show_failures: int = 5) -> str:
        d = self.decision
        pairs = [(r.confidence, r.passed) for r in self.results]
        out: list[str] = []

        out.append(f"# Score report: {self.task}")
        out.append("")
        out.append(f"**{d.verdict.value}**: {_MARK[d.verdict]}")
        out.append("")
        out.append(d.reason.capitalize() + ".")
        out.append("")

        out.append("| | |")
        out.append("| --- | --- |")
        out.append(f"| Cases evaluated | {self.n} |")
        out.append(f"| Passed | {self.n_passed} |")
        out.append(f"| Pass rate | {d.pass_rate} |")
        out.append(f"| Target | {d.target:.0%} |")
        if d.calibration_error == d.calibration_error:  # not NaN
            usable = "usable" if d.calibration_error <= 0.15 else "too high to threshold on"
            out.append(f"| Calibration error | {d.calibration_error:.2f} ({usable}) |")
        if self.errors:
            out.append(f"| Cases that errored | {len(self.errors)} |")
        out.append("")

        out.append(f"The pass rate is reported with a {d.pass_rate.level:.0%} confidence interval. "
                   f"With {self.n} cases the true rate could plausibly be anywhere in that range, "
                   f"which is why the interval and not the headline number drives the decision.")
        out.append("")

        if d.operating_point:
            p = d.operating_point
            out.append("## Recommended operating point")
            out.append("")
            out.append(f"Let the agent answer only when its confidence is at least "
                       f"**{p.threshold:.2f}**, and route the rest to a person.")
            out.append("")
            out.append("| | |")
            out.append("| --- | --- |")
            out.append(f"| Cases handled automatically | {p.n_covered} of {self.n} ({p.coverage:.0%}) |")
            out.append(f"| Precision on those | {p.precision} |")
            out.append(f"| Cases sent to review | {p.n_deferred} |")
            out.append("")

        curve = self.curve()
        if len(curve) > 1:
            out.append("## Coverage vs precision")
            out.append("")
            out.append("| Confidence floor | Handled | Precision | To review |")
            out.append("| --- | --- | --- | --- |")
            for p in _thin(curve, 8):
                out.append(f"| {p.threshold:.2f} | {p.coverage:.0%} | {p.precision} | {p.n_deferred} |")
            out.append("")

        table = reliability_table(pairs)
        if len(table) > 1:
            out.append("## Calibration")
            out.append("")
            out.append("| Stated confidence | Cases | Mean confidence | Actual accuracy |")
            out.append("| --- | --- | --- | --- |")
            for row in table:
                out.append(f"| {row['range']} | {row['n']} | {row['mean_confidence']:.2f} | "
                           f"{row['accuracy']:.0%} |")
            out.append("")

        if d.notes:
            out.append("## Caveats")
            out.append("")
            for note in d.notes:
                out.append(f"- {note}")
            out.append("")

        failures = self.failures(show_failures)
        if failures:
            out.append(f"## Failures worth reading ({len(self.failures())} total)")
            out.append("")
            for r in failures:
                detail = r.detail or (r.prediction.error or "")
                out.append(f"- `{r.case.id}` (confidence {r.confidence:.2f}): {detail or 'no detail'}")
            out.append("")

        return "\n".join(out).rstrip() + "\n"

    def to_dict(self) -> dict:
        d = self.decision
        return {
            "task": self.task,
            "verdict": d.verdict.value,
            "reason": d.reason,
            "n": self.n,
            "n_passed": self.n_passed,
            "pass_rate": {"point": d.pass_rate.point, "low": d.pass_rate.low,
                          "high": d.pass_rate.high, "level": d.pass_rate.level},
            "target": d.target,
            "calibration_error": d.calibration_error,
            "needed_n": d.needed_n,
            "operating_point": None if not d.operating_point else {
                "threshold": d.operating_point.threshold,
                "coverage": d.operating_point.coverage,
                "n_covered": d.operating_point.n_covered,
                "precision": d.operating_point.precision.point,
                "precision_low": d.operating_point.precision.low,
                "n_deferred": d.operating_point.n_deferred,
            },
            "notes": d.notes,
            "metadata": self.metadata,
        }


def _thin(points: list[OperatingPoint], limit: int) -> list[OperatingPoint]:
    """Evenly sample a long curve down to `limit` rows, always keeping the ends."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]
