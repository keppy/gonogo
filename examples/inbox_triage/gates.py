"""Gates: the component that makes the triage decision, three ways.

Every gate takes one inbox message and returns a Decision: the action (reply,
ask_human, wait, no_reply), the gate's confidence in it, and its reason. The
harness (`agent.py`) turns that decision into a tool call and a trace row; the
eval (`run_gate.py`) turns it into a gonogo Prediction. The gates are:

- `ClaudeGate`: a frontier model choosing one of four tools, the agent itself.
- `JevGate`: TypeSafe's Jev with the choice array passed at runtime.
- `EncoderGate`: a thomas encoder (ModernBERT-base) fine-tuned on the panel's labels; its
  scores are cached like the others (the model: see README).

Every response is cached to `data/cache/<gate>.jsonl` by message id, so a report
can be re-rendered and the demo replayed offline without paying again.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "cache"
ACTIONS = ("reply", "ask_human", "wait", "no_reply")


@dataclass
class Decision:
    action: str
    confidence: float
    reason: str = ""
    args: dict = field(default_factory=dict)
    error: str | None = None


class Cache:
    """Append-only JSONL cache keyed by message id; the last good row wins."""

    def __init__(self, name: str):
        # keyed by the guidelines version too: a gate prompted with other guidelines is another gate
        import hashlib
        version = hashlib.sha1((HERE / "guidelines.md").read_bytes()).hexdigest()[:8]
        self.path = CACHE / f"{name}.{version}.jsonl"
        self.rows: dict[str, dict] = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line) if line.strip() else {}
                if row.get("action") in ACTIONS:
                    self.rows[row["id"]] = row

    def get(self, row_id: str) -> dict | None:
        return self.rows.get(row_id)

    def put(self, row: dict) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row.get("action") in ACTIONS:
                self.rows[row["id"]] = row


def api_key(name: str) -> str:
    """A provider key from the environment. Only live (paid) runs need one; --offline never does."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set; live runs call the provider. Use --offline to replay the cache.")
    return value


def load_messages() -> dict[str, dict]:
    """The email text by case id, from data/messages.jsonl (built by fetch_text.py).

    The repo ships no email text. Offline replay doesn't need it: every gate's answers
    are cached by message id. Live runs, --dry-run and agent.py read the text.
    """
    path = HERE / "data" / "messages.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l)}


def message_text(row: dict) -> str:
    owner = row["user"].split("-")[0].title()
    return (f"Mailbox owner: {owner}\nFrom: {row['from']}\nTo: {row['to']}\nCc: {row['cc']}\n"
            f"Date: {row['date']}\nSubject: {row['subject']}\n\n{row['body']}"
            + ("\n[message truncated]" if row.get("truncated") else ""))


# --------------------------------------------------------------------- Claude

CONFIDENCE = {
    "type": "number",
    "description": ("Probability, 0 to 1, that this is the action the mailbox owner would want. "
                    "Calibrate it honestly: on a genuinely ambiguous message something near 0.5 is "
                    "the right answer. Don't inflate it, and don't give every message the same value."),
}
REASON = {"type": "string", "description": "One sentence naming the part of the message that decided it."}

TOOLS = [
    {"name": "reply", "description": "Send a reply now. Only for routine replies an assistant can write.",
     "input_schema": {"type": "object", "properties": {
         "draft": {"type": "string", "description": "The reply to send, short."},
         "reason": REASON, "confidence": CONFIDENCE}, "required": ["draft", "reason", "confidence"]}},
    {"name": "ask_human", "description": "Hand the message to the mailbox owner: only they should decide it.",
     "input_schema": {"type": "object", "properties": {
         "question": {"type": "string", "description": "What the owner needs to decide, one line."},
         "reason": REASON, "confidence": CONFIDENCE}, "required": ["question", "reason", "confidence"]}},
    {"name": "wait", "description": "Send nothing yet: the thread is live, and something is still coming.",
     "input_schema": {"type": "object", "properties": {
         "waiting_for": {"type": "string", "description": "What has to happen before anything is sent."},
         "reason": REASON, "confidence": CONFIDENCE}, "required": ["waiting_for", "reason", "confidence"]}},
    {"name": "no_reply", "description": "Nothing to do: archive it.",
     "input_schema": {"type": "object", "properties": {
         "reason": REASON, "confidence": CONFIDENCE}, "required": ["reason", "confidence"]}},
]

# USD per 1M tokens, platform.claude.com/docs/en/about-claude/pricing, checked 2026-10-21.
CLAUDE_PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 0.10)}


class ClaudeGate:
    """The frontier-model gate: Claude picks exactly one of the four tools.

    The decision *is* the tool call, so this is also the agent: nothing is
    decided outside a recorded call, including saying nothing.
    """

    def __init__(self, model: str = "claude-opus-5-5", client=None, effort: str = "low"):
        self.model = model
        self.name = f"claude-{model.removeprefix('claude-')}"
        self.effort = effort
        self.cache = Cache(self.name)
        self.client = client
        self.system = [{
            "type": "text",
            "text": ((HERE / "guidelines.md").read_text(encoding="utf-8")
                     + "\n\nDecide by calling exactly one of the four tools. Don't answer in text."),
            "cache_control": {"type": "ephemeral"},
        }]

    def decide(self, row: dict) -> Decision:
        hit = self.cache.get(row["id"])
        if hit:
            return Decision(hit["action"], hit["confidence"], hit.get("reason", ""), hit.get("args", {}))
        if self.client is None:
            return Decision("no_reply", 0.0, error="not cached and no client (offline replay)")
        resp = self.client.messages.create(
            model=self.model, max_tokens=4096, system=self.system, tools=TOOLS,
            tool_choice={"type": "auto"}, output_config={"effort": self.effort},
            messages=[{"role": "user", "content": message_text(row)}],
        )
        calls = [b for b in resp.content if b.type == "tool_use"]
        usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens,
                 "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0}
        if len(calls) != 1 or calls[0].name not in ACTIONS:
            err = f"expected one tool call, got {[b.type for b in resp.content]} (stop={resp.stop_reason})"
            self.cache.put({"id": row["id"], "error": err, "model": resp.model, **usage})
            return Decision("no_reply", 0.0, error=err)
        call = calls[0]
        args = dict(call.input)
        conf = min(1.0, max(0.0, float(args.pop("confidence", 0.0))))
        reason = str(args.pop("reason", ""))
        self.cache.put({"id": row["id"], "action": call.name, "confidence": round(conf, 3),
                        "reason": reason, "args": args, "model": resp.model, **usage})
        return Decision(call.name, round(conf, 3), reason, args)

    def spent(self) -> float:
        p_in, p_out, p_hit = CLAUDE_PRICES.get(self.model, (0.0, 0.0, 0.0))
        rows = self.cache.rows.values()
        return sum((r.get("in", 0) * p_in + r.get("cache_write", 0) * p_in * 1.25
                    + r.get("cache_read", 0) * p_hit + r.get("out", 0) * p_out) / 1e6 for r in rows)


# ------------------------------------------------------------------------ Jev

# USD per 1M input tokens; output is free (docs.typesafe.ai/models, checked 2026-10-22).
JEV_PRICE_IN = 0.042


def guideline_parts() -> tuple[str, dict[str, str]]:
    """guidelines.md split for a Choice: the prose as instructions, the table rows as criteria."""
    text = (HERE / "guidelines.md").read_text(encoding="utf-8")
    criteria = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0].strip("`") in ACTIONS:
            criteria[cells[0].strip("`")] = cells[1]
    if set(criteria) != set(ACTIONS):
        raise ValueError(f"guidelines.md table gave {sorted(criteria)}, expected {ACTIONS}")
    prose = "\n".join(l for l in text.splitlines()
                      if not l.startswith("|") and not l.startswith("Write one short sentence"))
    return prose.strip(), criteria


class JevGate:
    """TypeSafe's Jev with the choice array passed at runtime: one Choice per message.

    The instructions are the same guidelines every other gate and labeler reads, and
    the four options carry the same definitions. The gonogo confidence is the
    probability Jev puts on its chosen option (the "calibrated probability" being
    tested); TypeSafe's own `confidence` statistic, derived from how the probability
    is spread, is cached alongside.
    """

    def __init__(self, model: str = "jev-1.14.0", client=None):
        self.model = model
        self.name = model
        self.cache = Cache(self.name)
        self.client = client
        self.instructions, self.criteria = guideline_parts()

    def decide(self, row: dict) -> Decision:
        hit = self.cache.get(row["id"])
        if hit:
            return Decision(hit["action"], hit["confidence"], "", {"jev_confidence": hit.get("jev_confidence")})
        if self.client is None:
            return Decision("no_reply", 0.0, error="not cached and no client (offline replay)")
        from typesafe_sdk import Choice
        try:
            resp = self.client.system_one(
                state={"mailbox_owner": row["user"].split("-")[0].title(), "from": row["from"],
                       "to": row["to"], "cc": row["cc"], "date": row["date"],
                       "subject": row["subject"],
                       "body": row["body"] + ("\n[message truncated]" if row.get("truncated") else "")},
                questions={"action": Choice(instructions=self.instructions, criteria=self.criteria)},
                model=self.model,
            )
        except Exception as exc:  # recorded, never a silent no_reply
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.cache.put({"id": row["id"], "error": err})
            return Decision("no_reply", 0.0, error=err)
        ans = resp.answers["action"]
        probs = {k: float(v) for k, v in dict(ans.probabilities).items()}
        p = round(probs.get(ans.choice, 0.0), 4)
        usage = getattr(resp, "usage", None)
        self.cache.put({"id": row["id"], "action": ans.choice, "confidence": p,
                        "jev_confidence": round(float(ans.confidence), 4), "probabilities": probs,
                        "model": resp.model, "in": getattr(usage, "input_tokens", 0) or 0,
                        "out": getattr(usage, "output_tokens", 0) or 0})
        return Decision(ans.choice, p, "", {"jev_confidence": float(ans.confidence)})

    def spent(self) -> float:
        return sum(r.get("in", 0) for r in self.cache.rows.values()) * JEV_PRICE_IN / 1e6


# -------------------------------------------------------------------- encoder

class EncoderGate:
    """A thomas encoder fine-tuned on the panel's labels, replayed from its cached scores.

    Inference ran once, locally, and landed in the same cache the other gates use, so this
    gate never calls anything: a message it wasn't scored on is an error.
    """

    def __init__(self, run: str):
        self.name = f"encoder-{run}"
        self.cache = Cache(self.name)

    def decide(self, row: dict) -> Decision:
        hit = self.cache.get(row["id"])
        if not hit:
            return Decision("no_reply", 0.0, error="not in the encoder's cached scores")
        return Decision(hit["action"], hit["confidence"])

    def spent(self) -> float:
        return 0.0  # inference was local; training cost is in the README
