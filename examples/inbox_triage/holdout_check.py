"""Re-test a threshold chosen on eval, fixed, on the fresh holdout.

    python examples/inbox_triage/holdout_check.py --gate claude-opus-5-5

gonogo's AUTOMATE WITH REVIEW verdict picks its threshold on the cases it measured,
so that precision is optimistic. This reads the threshold from the eval report and
applies it unchanged to the holdout report: coverage, precision with its Wilson
interval, and whether the lower bound still clears the target. Nothing is re-picked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gonogo import wilson
from gonogo.style import paint

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "data" / "reports"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", default="claude-opus-5-5")
    ap.add_argument("--mix", default="enriched")
    ap.add_argument("--level", type=float, default=0.95)
    args = ap.parse_args()

    ev = json.loads((REPORTS / f"{args.gate}-eval-{args.mix}.json").read_text())
    ho = json.loads((REPORTS / f"{args.gate}-holdout-{args.mix}.json").read_text())
    op = ev.get("operating_point")
    if not op:
        raise SystemExit("the eval report has no operating point; nothing to re-test")
    t, target = op["threshold"], ev["target"]
    covered = [c["passed"] for c in ho["cases"] if c["confidence"] is not None and c["confidence"] >= t]
    n = len(ho["cases"])
    iv = wilson(sum(covered), len(covered), args.level)

    def say(text: str) -> None:
        print(paint(text, target=target))  # colored on a terminal, plain in a pipe

    say(f"threshold {t} chosen on eval ({op['coverage']:.1%} coverage, precision "
        f"{op['precision']:.1%} [{op['precision_low']:.1%}, {op['precision_high']:.1%}])")
    say(f"holdout, same threshold: {len(covered)}/{n} handled ({len(covered) / n:.1%}), "
        f"precision {iv.point:.1%} [{iv.low:.1%}, {iv.high:.1%}], {n - len(covered)} to the owner")
    ok = iv.low >= target
    say(f"lower bound {iv.low:.1%} {'clears' if ok else 'misses'} the {target:.0%} target: "
        + ("the threshold holds on fresh cases" if ok else "it does not hold; don't route by it"))
    say(f"holdout overall: {ho['pass_rate']['point']:.1%} [{ho['pass_rate']['low']:.1%}, "
        f"{ho['pass_rate']['high']:.1%}] on {n} cases ({ho['verdict']}, threshold re-picked there: "
        f"{(ho.get('operating_point') or {}).get('threshold')}, not used)")


if __name__ == "__main__":
    main()
