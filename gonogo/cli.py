"""`gonogo 47/50 --target 0.9`: the verdict for a pass count, from the shell."""

from __future__ import annotations

import argparse
import re
import sys

from .decide import Verdict, decide_counts


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
    d = decide_counts(passed, total, target=target, level=level)
    lines = [d.verdict.value, d.reason]
    if d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needed_n:
        lines.append(f"{passed}/{total} looks like {passed / total:.0%}; you'd need about "
                     f"{d.needed_n} cases at this rate to claim {target:.0%}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gonogo",
        description="Is PASSED out of TOTAL good enough to claim the target? "
                    "Prints the verdict, the interval, and how many cases the claim needs.")
    ap.add_argument("counts", type=_fraction, metavar="PASSED/TOTAL", help="e.g. 47/50")
    ap.add_argument("--target", type=_probability, default=0.95,
                    help="pass rate you want to claim (default 0.95)")
    ap.add_argument("--level", type=_probability, default=0.95,
                    help="confidence level of the interval (default 0.95)")
    ap.add_argument("--version", action="version", version=f"gonogo-eval {_version()}")
    args = ap.parse_args(argv)
    passed, total = args.counts
    print(format_counts(passed, total, target=args.target, level=args.level))
    return 0


def _version() -> str:
    from . import __version__
    return __version__


if __name__ == "__main__":
    sys.exit(main())
