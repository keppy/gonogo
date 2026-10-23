"""Grade a gate on the panel's cases with gonogo, and save the report for pairing.

    python examples/inbox_triage/run_gate.py --gate claude --dry-run          # free: calls + cost
    python examples/inbox_triage/run_gate.py --gate claude --mix balanced     # paid (cached after)
    python examples/inbox_triage/run_gate.py --gate claude --mix natural --offline

Mixes, all on the same split (default eval):

- natural:  the random sample only, the inbox as it arrives (mostly no_reply);
- enriched: random + enriched rows;
- balanced: enriched, with no_reply downsampled (seeded) to the largest other class, so
  "always do nothing" can't pass on volume.

A `reply` case passes when the gate replies (the draft isn't graded); every other case
passes only when the gate takes that action (`Case.expected_action`), so the report has
the actions table. Spot-check rows stay out until the human sheet is read back, so a
report can't show a labeler's answer for them; pass --include-spot after that.

Reports go to data/reports/<gate>-<split>-<mix>.json|html. compare() pairs them.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from gonogo import Case, Prediction, evaluate

from gates import (ACTIONS, CLAUDE_PRICES, JEV_PRICE_IN, ClaudeGate, EncoderGate, JevGate, api_key,
                   load_messages, message_text)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LABEL_SOURCE = "model panel (deepseek-v4-pro + GLM-5.3, agreement only), not human labels"


def load_cases(split: str, mix: str, include_spot: bool, seed: int = 1103) -> list[dict]:
    path = DATA / "cases" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing")
    cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    if not include_spot:
        cases = [c for c in cases if not c["spot"]]
    if mix == "natural":
        cases = [c for c in cases if c["sampling"] == "random"]
    if mix == "balanced":
        by = collections.defaultdict(list)
        for c in cases:
            by[c["expected"]].append(c)
        cap = max(len(v) for k, v in by.items() if k != "no_reply")
        rng = random.Random(seed)
        by["no_reply"] = rng.sample(by["no_reply"], min(cap, len(by["no_reply"])))
        cases = [c for k in ACTIONS for c in by[k]]
    return cases


def gonogo_agent(gate, rows_by_id: dict[str, dict]):
    def agent(case: Case):
        d = gate.decide(rows_by_id.get(str(case.id), {"id": str(case.id)}))
        if d.error:
            return Prediction(None, error=d.error)
        if d.action == "reply":
            return d.args.get("draft") or "(reply)", d.confidence
        return Prediction(None, confidence=d.confidence, action=d.action)
    return agent


def decided(output, expected):
    ok = output is not None
    return ok, 1.0 if ok else 0.0, "" if ok else "no reply drafted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", choices=("claude", "jev", "encoder"), default="claude")
    ap.add_argument("--model", help="default: claude-opus-5-5 / jev-1.14.0 / encoder run inbox-triage-v1")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--mix", choices=("natural", "enriched", "balanced"), default="balanced")
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--include-spot", action="store_true")
    ap.add_argument("--limit", type=int, help="a seeded random subset of this many cases (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="count uncached calls and estimate cost; no calls")
    ap.add_argument("--offline", action="store_true", help="replay the cache only")
    args = ap.parse_args()

    rows = load_messages()
    if args.gate != "encoder" and (not args.offline or args.dry_run) and not rows:
        raise SystemExit("data/messages.jsonl is missing: run fetch_text.py first (the repo ships no "
                         "email text), or pass --offline to replay the cached answers")
    cases = load_cases(args.split, args.mix, args.include_spot)
    if args.limit:
        cases = random.Random(7).sample(cases, min(args.limit, len(cases)))
    dist = collections.Counter(c["expected"] for c in cases)
    print(f"{args.split}/{args.mix}: {len(cases)} cases (" + ", ".join(f"{a}={dist[a]}" for a in ACTIONS) + ")")

    client = None
    online = not (args.dry_run or args.offline)
    if args.gate == "claude":
        args.model = args.model or "claude-opus-5-5"
        if online:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"), max_retries=5)
        gate = ClaudeGate(args.model, client)
    elif args.gate == "encoder":
        args.model = args.model or "inbox-triage-v1"
        gate = EncoderGate(args.model)  # replay only: its scores were computed once, locally
    else:
        args.model = args.model or "jev-1.14.0"
        if online:
            from typesafe_sdk import TypeSafeClient
            client = TypeSafeClient(api_key=api_key("TYPESAFE_API_KEY"))
        gate = JevGate(args.model, client)

    todo = [c for c in cases if not gate.cache.get(c["id"])]
    if args.dry_run and args.gate == "jev":
        chars = sum(len(gate.instructions) + sum(map(len, gate.criteria.values()))
                    + len(message_text(rows[c["id"]])) for c in todo)
        tok = chars / 3.5 * 1.2
        print(f"{gate.name}: {len(cases) - len(todo)} cached, {len(todo)} to call; estimate "
              f"~{tok:,.0f} input tokens = ~${tok * JEV_PRICE_IN / 1e6:.3f} (output is free)")
        return
    if args.dry_run:
        p_in, p_out, p_hit = CLAUDE_PRICES[args.model]
        sys_tok = len(gate.system[0]["text"]) / 3.5 * 1.3 + 900   # + tool schemas
        msg_tok = sum(len(message_text(rows[c["id"]])) for c in todo) / 3.5 * 1.3
        est = (msg_tok * p_in + len(todo) * sys_tok * p_hit + len(todo) * 400 * p_out) / 1e6
        print(f"{gate.name}: {len(cases) - len(todo)} cached, {len(todo)} to call; "
              f"estimate ~${est:.2f} (~{msg_tok / max(1, len(todo)):.0f} message tokens + cached system "
              f"prompt per call, ~400 output tokens incl. low-effort thinking; ${p_in}/${p_out} per 1M)")
        return

    report = evaluate(
        gonogo_agent(gate, rows),
        [Case(input=c.get("input", c["id"]), expected=c["expected"], id=c["id"], expected_action=c["expected_action"],
              metadata={"sampling": c["sampling"], "mailbox": c["mailbox"]}) for c in cases],
        scorer=decided,
        task=f"Inbox triage gate: {gate.name} ({args.split}/{args.mix}; labels: {LABEL_SOURCE})",
        target=args.target,
        workers=args.workers,
    )
    out = DATA / "reports" / (f"{gate.name}-{args.split}-{args.mix}"
                              + (f"-limit{args.limit}" if args.limit else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload.setdefault("metadata", {})["label_source"] = LABEL_SOURCE
    # not with_suffix: "jev-1.14.0-eval-enriched" would lose ".0-eval-enriched" as a suffix
    Path(f"{out}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(f"{out}.html").write_text(report.html(), encoding="utf-8")
    print(report.summary())
    a = report.actions()
    print(f"acted when it should have held back: {a['acted_when_should_hold_back']}; "
          f"held back when it should have acted: {a['held_back_when_should_act']}; "
          f"errors: {len(report.errors)}")
    print(f"report: {out}.json / .html; spend so far on {gate.name}: ${gate.spent():.2f} at list price")


if __name__ == "__main__":
    main()
