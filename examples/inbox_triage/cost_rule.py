"""Escalate on P(ask_human) at a threshold set by costs, not picked on data.

    python examples/inbox_triage/cost_rule.py                       # eval sweep + holdout at r=4
    python examples/inbox_triage/cost_rule.py --ratio 4 --split holdout

A message that needed the owner and didn't reach them (a silent miss, or a drafted reply
where the owner had to answer) costs r; sending the owner a message they didn't need costs
1. If P(ask_human) is calibrated, the cost-minimising rule is: route to ask_human when
P(ask_human) >= 1 / (1 + r), otherwise take the gate's own choice. r is declared up front
(default 4, so the threshold is 0.2); nothing is tuned on eval or holdout.

The encoders and Jev give every option's probability. Claude returns one tool call with a
confidence, so for Claude P(ask_human) is its confidence when it chose ask_human and
1 - confidence otherwise, an upper bound: Claude's row escalates at least as often as the
true rule would.

Each gate's replayed decisions go through gonogo like run_gate.py, saved as
data/reports/<gate>-cost<r>-<split>-<mix>.json, then compare() pairs the gates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from gonogo import Case, Prediction, compare, evaluate
from gonogo.style import paint
from run_gate import LABEL_SOURCE, decided, load_cases

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SHA = "ec1a66c4"
GATES = ("claude-opus-5-5", "jev-1.14.0", "encoder-inbox-triage-v4")


def p_ask(d: dict) -> float:
    if "probabilities" in d:
        return d["probabilities"]["ask_human"]
    return d["confidence"] if d["action"] == "ask_human" else 1.0 - d["confidence"]


def apply_rule(d: dict, t: float) -> tuple[str, float]:
    p = p_ask(d)
    if d["action"] != "ask_human" and p >= t:
        return "ask_human", p
    return d["action"], d["confidence"]


def run(gate: str, split: str, mix: str, ratio: float, target: float):
    t = 1.0 / (1.0 + ratio)
    cache = {d["id"]: d for d in (json.loads(l) for l in
             (DATA / "cache" / f"{gate}.{SHA}.jsonl").read_text().splitlines() if l)}
    cases = load_cases(split, mix, include_spot=False)
    taken = {}

    def agent(case: Case):
        d = cache[str(case.id)]
        if d.get("error"):
            return Prediction(None, error=d["error"])
        action, conf = apply_rule(d, t)
        taken[str(case.id)] = action
        if action == "reply":
            return "(reply)", conf
        return Prediction(None, confidence=conf, action=action)

    report = evaluate(
        agent,
        [Case(input=c.get("input", c["id"]), expected=c["expected"], id=c["id"], expected_action=c["expected_action"])
         for c in cases],
        scorer=decided,
        task=f"Inbox triage gate: {gate} + cost rule r={ratio:g} (escalate at P(ask_human) >= {t:.3f}) "
             f"({split}/{mix}; labels: {LABEL_SOURCE})",
        target=target,
        workers=1,
    )
    payload = report.to_dict()
    payload.setdefault("metadata", {})["label_source"] = LABEL_SOURCE
    out = DATA / "reports" / f"{gate}-cost{ratio:g}-{split}-{mix}"
    Path(f"{out}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(f"{out}.html").write_text(report.html(), encoding="utf-8")

    exp = {c["id"]: c["expected"] for c in cases}
    owner_missed = sum(exp[i] == "ask_human" and a != "ask_human" for i, a in taken.items())
    replied_instead = sum(exp[i] == "ask_human" and a == "reply" for i, a in taken.items())
    over = sum(exp[i] != "ask_human" and a == "ask_human" for i, a in taken.items())
    other = sum(exp[i] != "ask_human" and a not in (exp[i], "ask_human") for i, a in taken.items())
    cost = ratio * owner_missed + over + other
    per_case = {i: (ratio if exp[i] == "ask_human" else 1.0) * (a != exp[i] and not (exp[i] == "reply" and a == "reply"))
                for i, a in taken.items()}
    row = dict(gate=gate, t=t, n=len(cases), per_case=per_case, verdict=payload["verdict"],
               owner_missed=owner_missed, replied_instead=replied_instead, over=over, other=other,
               cost_per_100=100 * cost / len(cases), owner_load=sum(a == "ask_human" for a in taken.values()))
    return payload, row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=float, nargs="+", default=[1, 2, 4, 9])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--mix", default="enriched")
    ap.add_argument("--target", type=float, default=0.90)
    args = ap.parse_args()
    for r in args.ratio:
        print(f"\n== {args.split}/{args.mix}, r={r:g} (escalate at P(ask_human) >= {1 / (1 + r):.3f})")
        print(f"{'gate':26} {'pass rate':22} {'owner missed':>12} {'(replied)':>9} {'over-esc':>8} "
              f"{'other':>5} {'to owner':>8} {'cost/100':>8}  verdict")
        reports, costs = {}, {}
        for g in GATES:
            payload, row = run(g, args.split, args.mix, r, args.target)
            reports[g], costs[g] = payload, row["per_case"]
            pr = payload["pass_rate"]
            print(paint(f"{g:26} {pr['point']:.1%} [{pr['low']:.1%}, {pr['high']:.1%}]   {row['owner_missed']:>12} "
                        f"{row['replied_instead']:>9} {row['over']:>8} {row['other']:>5} {row['owner_load']:>8} "
                        f"{row['cost_per_100']:>8.1f}  {row['verdict']}", target=args.target))
        for a, b in itertools.combinations(GATES, 2):
            lines = compare(reports[a], reports[b]).summary().splitlines()
            keep = [l.split(None, 1)[1].strip() for l in lines if l.startswith(("Difference", "McNemar", "Verdict"))]
            lo, hi = cost_interval(costs[a], costs[b])
            diff = 100 * (sum(costs[b].values()) - sum(costs[a].values())) / len(costs[a])
            print(f"  {a} vs {b}: " + " | ".join(keep)
                  + f" || cost/100 B-A {diff:+.1f} [{lo:+.1f}, {hi:+.1f}] (paired bootstrap 95%)")


def cost_interval(a: dict, b: dict, reps: int = 10_000, seed: int = 11) -> tuple[float, float]:
    """Paired bootstrap over cases of (cost B - cost A) per 100 messages."""
    rng = random.Random(seed)
    d = [b[i] - a[i] for i in a]
    n = len(d)
    means = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


if __name__ == "__main__":
    main()
