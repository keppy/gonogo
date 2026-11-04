"""The shell surface: `gonogo 47/50`, `gonogo decide results.jsonl`, `gonogo compare a.json b.json`.

Every verb speaks `--json` so a client in another language reads an object, never prose."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

from .compare import compare
from .decide import MIN_USEFUL_COVERAGE, Decision, Verdict, decide_counts
from .evaluate import GROUP_RULES, report_from_results
from .outcomes import results_from_jsonl
from .style import paint

_VERBS = ("decide", "compare")


def _fraction(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected PASSED/TOTAL like 47/50, got {text!r}")
    passed, total = int(m.group(1)), int(m.group(2))
    if total < 1 or passed > total:
        raise argparse.ArgumentTypeError(f"need TOTAL >= 1 and PASSED <= TOTAL, got {text}")
    return passed, total


def _probability(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number in (0, 1), got {text!r}") from None
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"expected a number in (0, 1), got {text}")
    return value


def format_counts(passed: int, total: int, target: float = 0.95, level: float = 0.95) -> str:
    return _format_decision(decide_counts(passed, total, target=target, level=level), passed, total)


def _format_decision(d: Decision, passed: int, total: int) -> str:
    """Verdict, reason, and how many trials the claim needs: the same three lines for a count and a file."""
    lines = [d.verdict.value, d.reason]
    if d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needed_n:
        lines.append(f"{passed}/{total} looks like {passed / total:.0%}; you'd need about "
                     f"{d.needed_n} {d.unit} at this rate to claim {d.target:.0%}.")
    return "\n".join(lines)


def _decision_json(d: Decision, passed: int, total: int) -> dict:
    """The decision as one JSON object, for a script or a non-Python client.

    Mirrors the decision fields of `Report.to_dict()` so a consumer that reads a
    score report sees the same shape here; the count has no per-case outcomes,
    so `passed`/`total` stand in for `cases`. `calibration_error` is `null` for
    a count (there is no per-case confidence to calibrate).
    """
    return {
        "verdict": d.verdict.value,
        "can_automate": d.can_automate,
        "reason": d.reason,
        "passed": passed,
        "total": total,
        "unit": d.unit,
        "pass_rate": {
            "point": d.pass_rate.point,
            "low": d.pass_rate.low,
            "high": d.pass_rate.high,
            "level": d.pass_rate.level,
        },
        "target": d.target,
        "calibration_error": _finite(d.calibration_error),
        "needed_n": d.needed_n,
        "notes": d.notes,
    }


def _finite(value: float) -> float | None:
    """Non-finite floats (NaN) are not valid JSON; emit `null` instead."""
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _VERBS:
        return _verb_main(argv)
    ap = argparse.ArgumentParser(
        prog="gonogo",
        description="Is PASSED out of TOTAL good enough to claim the target? "
                    "Prints the verdict, the interval, and how many cases the claim needs.",
        epilog="Per-case results: gonogo decide RESULTS.jsonl. Two runs: gonogo compare A.json B.json.")
    ap.add_argument("counts", type=_fraction, metavar="PASSED/TOTAL", help="e.g. 47/50")
    ap.add_argument("--target", type=_probability, default=0.95,
                    help="pass rate you want to claim (default 0.95)")
    ap.add_argument("--level", type=_probability, default=0.95,
                    help="confidence level of the interval (default 0.95)")
    ap.add_argument("--version", action="version", version=f"gonogo-eval {_version()}")
    ap.add_argument("--json", action="store_true",
                    help="print the decision as one JSON object (scripts, non-Python clients)")
    args = ap.parse_args(argv)
    passed, total = args.counts
    if args.json:
        d = decide_counts(passed, total, target=args.target, level=args.level)
        print(json.dumps(_decision_json(d, passed, total)))
    else:
        print(paint(format_counts(passed, total, target=args.target, level=args.level),
                    target=args.target))
    return 0


def _verb_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gonogo")
    sub = ap.add_subparsers(dest="verb", required=True)

    dp = sub.add_parser("decide", help="the verdict for per-case results in a JSONL file",
                        description="One JSON object per line: passed (true/false), and optionally id, "
                                    "confidence, expected_action, action, group, score, detail, error.")
    dp.add_argument("results", metavar="RESULTS.jsonl")
    dp.add_argument("--target", type=_probability, default=0.95, help="pass rate to claim (default 0.95)")
    dp.add_argument("--level", type=_probability, default=0.95, help="interval confidence level (default 0.95)")
    dp.add_argument("--min-coverage", type=float, default=MIN_USEFUL_COVERAGE,
                    help=f"least coverage a review threshold is worth (default {MIN_USEFUL_COVERAGE})")
    dp.add_argument("--group-rule", choices=GROUP_RULES, default="none",
                    help="'all': one trial per group, passing only if every case in it passed")
    dp.add_argument("--task", default="task", help="name on the report")
    dp.add_argument("--json", action="store_true",
                    help="print Report.to_dict() as one JSON object, with per-case outcomes for compare")

    cp = sub.add_parser("compare", help="paired comparison of two saved reports on their shared cases",
                        description="A and B are files written by `gonogo decide --json` (Report.to_dict()).")
    cp.add_argument("a", metavar="A.json")
    cp.add_argument("b", metavar="B.json")
    cp.add_argument("--level", type=_probability, default=0.95, help="interval confidence level (default 0.95)")
    cp.add_argument("--json", action="store_true", help="print the comparison as one JSON object")

    args = ap.parse_args(argv)
    try:
        if args.verb == "decide":
            report = report_from_results(
                results_from_jsonl(args.results), task=args.task, target=args.target,
                level=args.level, min_coverage=args.min_coverage, group_rule=args.group_rule,
                metadata={"source": args.results})
            if args.json:
                print(json.dumps(report.to_dict()))
            else:
                d = report.decision
                passed, total = ((report.n_groups_passed(), d.n_groups) if report.grouped
                                 else (report.n_passed, report.n))
                print(paint(_format_decision(d, passed, total), target=args.target))
        else:
            with open(args.a, encoding="utf-8") as fa, open(args.b, encoding="utf-8") as fb:
                c = compare(json.load(fa), json.load(fb), level=args.level)
            print(json.dumps(c.to_dict()) if args.json else c.summary())
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    return 0


def _version() -> str:
    from . import __version__
    return __version__


if __name__ == "__main__":
    sys.exit(main())
