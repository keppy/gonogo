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
        """Failed cases, worst score first -- the list to actually go read.

        `limit=None` returns every failure; `limit=0` returns none.
        """
        worst = sorted((r for r in self.results if not r.passed), key=lambda r: r.score)
        return worst if limit is None else worst[:limit]

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
            usable = "well calibrated" if d.calibration_error <= 0.15 else "ranks cases, scale unreliable"
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

    def html(self, standalone: bool = True, show_failures: int = 5) -> str:
        """A self-contained HTML score report.

        Styled with CSS custom properties and no external assets, so it can be
        opened directly, emailed as a file, or dropped into a page that
        overrides the variables. `standalone=False` returns just the fragment.
        """
        d = self.decision
        e = _esc
        slug = d.verdict.name.lower().replace("_", "-")
        rows: list[str] = []

        rows.append(f'<section class="gng" data-verdict="{slug}">')
        rows.append(f'<header class="gng-head">')
        rows.append(f'<p class="gng-task">{e(self.task)}</p>')
        rows.append(f'<p class="gng-verdict">{e(d.verdict.value)}</p>')
        rows.append(f'<p class="gng-gloss">{e(_MARK[d.verdict])}</p>')
        rows.append('</header>')

        rows.append(f'<p class="gng-reason">{e(d.reason[:1].upper() + d.reason[1:])}.</p>')

        rows.append('<dl class="gng-facts">')
        facts = [("Cases evaluated", str(self.n)), ("Passed", str(self.n_passed)),
                 ("Pass rate", str(d.pass_rate)), ("Target", f"{d.target:.0%}")]
        if d.calibration_error == d.calibration_error:
            usable = "well calibrated" if d.calibration_error <= 0.15 else "ranks cases, scale unreliable"
            facts.append(("Calibration error", f"{d.calibration_error:.2f} ({usable})"))
        if self.errors:
            facts.append(("Cases that errored", str(len(self.errors))))
        if d.needed_n:
            facts.append(("Cases needed for the claim", f"~{d.needed_n}"))
        for label, value in facts:
            rows.append(f'<div><dt>{e(label)}</dt><dd>{e(value)}</dd></div>')
        rows.append('</dl>')

        rows.append(f'<p class="gng-note">The pass rate carries a {d.pass_rate.level:.0%} '
                    f'confidence interval. With {self.n} cases the true rate could plausibly sit '
                    f'anywhere in that range, which is why the interval and not the headline '
                    f'number drives the decision.</p>')

        if d.operating_point:
            p = d.operating_point
            rows.append('<div class="gng-op">')
            rows.append('<p class="gng-op-title">Recommended operating point</p>')
            rows.append(f'<p>Answer only above confidence <b>{p.threshold:.2f}</b>; '
                        f'route the rest to a person.</p>')
            rows.append(f'<p class="gng-op-nums">{p.n_covered} of {self.n} handled '
                        f'({p.coverage:.0%}) at {p.precision} precision, '
                        f'{p.n_deferred} to review.</p>')
            rows.append('</div>')

        curve = self.curve()
        if len(curve) > 1:
            rows.append('<table class="gng-table"><caption>Coverage vs precision</caption><thead><tr>'
                        '<th scope="col">Confidence floor</th><th scope="col">Handled</th>'
                        '<th scope="col">Precision</th><th scope="col">To review</th>'
                        '</tr></thead><tbody>')
            for p in _thin(curve, 8):
                meets = p.precision.low >= d.target
                flag = "ok" if meets else "under"
                rows.append(
                    f'<tr><td>{p.threshold:.2f}</td>'
                    f'<td><span class="gng-bar" style="--pct:{p.coverage:.0%}"><i></i></span>'
                    f'{p.coverage:.0%}</td>'
                    f'<td class="gng-{flag}">{p.precision.point:.1%} '
                    f'<span class="gng-ci">[{p.precision.low:.1%}, {p.precision.high:.1%}]</span></td>'
                    f'<td>{p.n_deferred}</td></tr>')
            rows.append('</tbody></table>')

        table = reliability_table([(r.confidence, r.passed) for r in self.results])
        if len(table) > 1:
            rows.append('<table class="gng-table"><caption>Calibration</caption><thead><tr>'
                        '<th scope="col">Stated confidence</th><th scope="col">Cases</th>'
                        '<th scope="col">Mean confidence</th><th scope="col">Actual accuracy</th>'
                        '</tr></thead><tbody>')
            for row in table:
                gap = abs(row["mean_confidence"] - row["accuracy"])
                flag = "under" if gap > 0.15 else "ok"
                rows.append(f'<tr><td>{e(row["range"])}</td><td>{row["n"]}</td>'
                            f'<td>{row["mean_confidence"]:.2f}</td>'
                            f'<td class="gng-{flag}">{row["accuracy"]:.0%}</td></tr>')
            rows.append('</tbody></table>')

        if d.notes:
            rows.append('<div class="gng-caveats"><p class="gng-op-title">Caveats</p><ul>')
            for note in d.notes:
                rows.append(f'<li>{e(note)}</li>')
            rows.append('</ul></div>')

        failures = self.failures(show_failures)
        if failures:
            total = len(self.failures())
            rows.append(f'<div class="gng-caveats"><p class="gng-op-title">'
                        f'Failures worth reading ({total} total)</p><ul>')
            for r in failures:
                detail = r.detail or r.prediction.error or "no detail"
                rows.append(f'<li><code>{e(str(r.case.id))}</code> '
                            f'(confidence {r.confidence:.2f}): {e(detail)}</li>')
            rows.append('</ul></div>')

        rows.append('</section>')
        fragment = "\n".join(rows)
        if not standalone:
            return fragment
        return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                f'<title>Score report: {e(self.task)}</title>\n<style>\n{GNG_CSS}\n</style>\n'
                f'</head>\n<body>\n{fragment}\n</body>\n</html>\n')

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


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


# Everything is driven by custom properties so a host page can restyle the
# report by overriding the variables rather than fighting the selectors.
GNG_CSS = """
.gng {
  --gng-ink: #111827;
  --gng-muted: #4b5563;
  --gng-border: #d1d5db;
  --gng-wash: #f8fafc;
  --gng-ok: #0f766e;
  --gng-under: #b45309;
  --gng-accent: var(--gng-ok);
  max-width: 44rem;
  margin: 0 auto;
  padding: 1.25rem;
  border: 1px solid var(--gng-border);
  border-radius: 8px;
  background: #fff;
  color: var(--gng-ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.gng[data-verdict="automate"] { --gng-accent: #0f766e; }
.gng[data-verdict="automate-with-review"] { --gng-accent: #0f766e; }
.gng[data-verdict="assist-only"] { --gng-accent: #b45309; }
.gng[data-verdict="do-not-automate"] { --gng-accent: #9f1239; }
.gng[data-verdict="insufficient-evidence"] { --gng-accent: #4b5563; }
.gng-head { padding-left: .75rem; border-left: 3px solid var(--gng-accent); }
.gng-task { margin: 0; color: var(--gng-muted); font-size: .72rem; font-weight: 700;
  letter-spacing: .04em; text-transform: uppercase; }
.gng-verdict { margin: .15rem 0 0; color: var(--gng-accent); font-size: 1.5rem;
  font-weight: 800; line-height: 1.1; }
.gng-gloss { margin: .15rem 0 0; color: var(--gng-muted); font-size: .95rem; }
.gng-reason { margin: 1rem 0 0; color: var(--gng-ink); }
.gng-note, .gng-op-nums { color: var(--gng-muted); font-size: .82rem; }
.gng-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .5rem 1rem; margin: 1rem 0 0; }
.gng-facts > div { display: flex; justify-content: space-between; gap: .5rem;
  padding-bottom: .35rem; border-bottom: 1px solid var(--gng-border); }
.gng-facts dt { color: var(--gng-muted); font-size: .8rem; }
.gng-facts dd { margin: 0; font-size: .85rem; font-weight: 700;
  font-variant-numeric: tabular-nums; }
.gng-op { margin: 1.25rem 0 0; padding: .85rem 1rem; border-left: 3px solid var(--gng-accent);
  border-radius: 0 8px 8px 0; background: var(--gng-wash); }
.gng-op p { margin: .2rem 0 0; }
.gng-op-title { margin: 0 !important; color: var(--gng-muted); font-size: .72rem;
  font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.gng-table { width: 100%; margin: 1.5rem 0 0; border-collapse: collapse;
  font-variant-numeric: tabular-nums; }
.gng-table caption { margin-bottom: .4rem; color: var(--gng-muted); font-size: .72rem;
  font-weight: 700; letter-spacing: .04em; text-align: left; text-transform: uppercase; }
.gng-table th { padding: 0 .5rem .35rem 0; border-bottom: 1px solid var(--gng-border);
  color: var(--gng-muted); font-size: .7rem; font-weight: 700; text-align: left;
  text-transform: uppercase; }
.gng-table td { padding: .45rem .5rem .45rem 0; border-bottom: 1px solid var(--gng-border);
  font-size: .85rem; }
.gng-table th:last-child, .gng-table td:last-child { padding-right: 0; text-align: right; }
.gng-ci { color: var(--gng-muted); font-size: .75rem; }
.gng-ok { color: var(--gng-ok); }
.gng-under { color: var(--gng-under); }
.gng-bar { display: inline-block; width: 2.75rem; height: .35rem; margin-right: .45rem;
  overflow: hidden; border-radius: 999px; background: rgba(17,24,39,.12);
  vertical-align: middle; }
.gng-bar i { display: block; width: var(--pct, 0%); height: 100%; background: var(--gng-accent); }
.gng-caveats { margin: 1.25rem 0 0; }
.gng-caveats ul { margin: .35rem 0 0; padding-left: 1.1rem; }
.gng-caveats li { color: var(--gng-muted); font-size: .82rem; margin-bottom: .3rem; }
.gng code { font-size: .8em; }
@media (max-width: 34rem) {
  .gng-facts { grid-template-columns: 1fr; }
  .gng-bar { display: none; }
}
@media (prefers-color-scheme: dark) {
  .gng:not([data-theme]) {
    --gng-ink: #e5e7eb; --gng-muted: #9ca3af; --gng-border: #374151; --gng-wash: #111827;
    --gng-ok: #5eead4; --gng-under: #fbbf24;
    background: #0b1220;
  }
}
""".strip()


def _thin(points: list[OperatingPoint], limit: int) -> list[OperatingPoint]:
    """Evenly sample a long curve down to `limit` rows, always keeping the ends."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]
