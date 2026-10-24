"""The inbox-triage harness: every decision is a tool call, and every tool call is a row.

    python examples/inbox_triage/agent.py --gate claude --limit 20            # the trace
    python examples/inbox_triage/agent.py --gate claude --limit 20 --legacy   # the "before" log

The "before" harness is the usual one: it logs what it sent. A message it decided to
leave alone produces no row, so the log can't tell a correct silence from a missed
reply, and nobody can grade it. The "after" harness makes `wait`, `ask_human` and
`no_reply` tool calls like `reply`, each with a reason and a confidence, so every
message leaves exactly one row. That trace is what `run_gate.py` grades.

The tools here are stubs that record instead of sending: this is an audit rig, not a
mail client.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gonogo.style import color

from gates import ACTIONS, Decision

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


class Harness:
    def __init__(self, gate, trace: Path, legacy: bool = False):
        self.gate, self.trace, self.legacy = gate, trace, legacy
        trace.parent.mkdir(parents=True, exist_ok=True)

    def _row(self, row: dict, d: Decision, tool: str) -> dict:
        return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "message": row["id"], "tool": tool,
                "args": d.args, "reason": d.reason, "confidence": d.confidence,
                "gate": getattr(self.gate, "name", type(self.gate).__name__), **({"error": d.error} if d.error else {})}

    def handle(self, row: dict) -> Decision:
        d = self.gate.decide(row)
        if d.action not in ACTIONS:
            raise ValueError(f"gate returned {d.action!r}")
        tool = "error" if d.error else d.action  # a failed decision is not a silent no_reply
        if self.legacy and tool != "reply":
            return d  # the old harness: nothing sent, nothing written
        with self.trace.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(self._row(row, d, "send_reply" if self.legacy else tool),
                                ensure_ascii=False) + "\n")
        return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", choices=("claude",), default="claude")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--legacy", action="store_true", help="the before harness: log sent replies only")
    ap.add_argument("--offline", action="store_true", help="replay the cache only; never call the API")
    args = ap.parse_args()

    if not (DATA / "messages.jsonl").exists():
        raise SystemExit("data/messages.jsonl is missing: run fetch_text.py first (the repo ships no email text)")
    from gates import ClaudeGate
    client = None
    if not args.offline:
        import anthropic
        from gates import api_key
        client = anthropic.Anthropic(api_key=api_key("ANTHROPIC_API_KEY"), max_retries=5)
    gate = ClaudeGate(args.model, client)
    rows = [json.loads(l) for l in (DATA / "messages.jsonl").read_text(encoding="utf-8").splitlines() if l]
    rows = [r for r in rows if r["split"] == args.split and not r.get("spot")][: args.limit]
    trace = DATA / "traces" / f"{gate.name}-{'legacy' if args.legacy else 'tools'}.jsonl"
    trace.unlink(missing_ok=True)
    h = Harness(gate, trace, legacy=args.legacy)
    for r in rows:
        h.handle(r)
    n_rows = sum(1 for _ in trace.open()) if trace.exists() else 0
    print(f"{len(rows)} messages in, {color(f'{n_rows} trace rows out', 'good' if n_rows == len(rows) else 'poor')}"
          f" {color(f'-> {trace}', 'dim')}")
    if args.legacy:
        print(color(f"{len(rows) - n_rows} messages left no row", "hazard") + ": from this log, a correct "
              "silence and a missed reply look the same")


if __name__ == "__main__":
    main()
