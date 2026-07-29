"""A real LLM agent, measured on the same cases as the baseline.

`banking77_routing.py` scores a TF-IDF nearest-centroid classifier on 250
Banking77 messages. This runs a Claude agent over the *same* 250 cases with the
*same* scorer, so the two reports are directly comparable -- which is the point.
A harness that only ever sees one agent tells you nothing about the agent; it
tells you about the harness.

    pip install anthropic
    python examples/banking77_claude_agent.py

Requires credentials: an ANTHROPIC_API_KEY, or an `ant auth login` profile (the
SDK resolves either automatically -- the zero-arg client below picks up
whichever is present).

Cost: 250 short classification calls. The 77-label system prompt is identical on
every request and marked for caching, so it is written once and read back at a
tenth of the price for the remaining 249.
"""

from __future__ import annotations

import argparse
import json
import os

from gonogo import Case, Prediction, evaluate
from gonogo.scoring import exact

from banking77_routing import PILOT_SIZE, SEED, fetch

try:
    import anthropic
except ImportError:  # pragma: no cover - the message is the point
    raise SystemExit("this example needs the Anthropic SDK: pip install anthropic")

import random

# Thinking is on by default on Claude Opus 5. Classification does not need deep
# reasoning, so effort is dialed down rather than switching thinking off --
# disabled thinking on this model has failure modes that low effort does not.
DEFAULT_MODEL = "claude-opus-5"
EFFORT = "low"
WORKERS = 8

SYSTEM = """You route customer messages for a retail bank to the correct intent queue.

Reply with the single best-matching intent from the list below, plus your
confidence that it is correct.

Calibrate the confidence honestly. It should be the probability that your chosen
intent is the one a trained human agent would pick. Many of these intents overlap
(for example `card_payment_fee_charged` and `transaction_fee_charged`), and on a
genuinely ambiguous message a confidence near 0.5 is the correct answer, not a
hedge. Do not inflate it, and do not flatten everything to the same value -- the
number is only useful if it separates the cases you are sure about from the ones
you are not.

Valid intents:
{intents}"""


def build_agent(client: anthropic.Anthropic, model: str, intents: list[str]):
    """A classification agent: one message in, one intent + confidence out.

    Structured outputs guarantee the reply parses and that `intent` is one of the
    77 labels, so there is no regex, no retry-on-malformed-JSON, and no chance of
    scoring a hallucinated label as a miss when it was really a format error.
    """
    system = [{
        "type": "text",
        "text": SYSTEM.format(intents="\n".join(f"- {i}" for i in intents)),
        # Identical on all 250 requests: write once, read back at ~0.1x.
        "cache_control": {"type": "ephemeral"},
    }]
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": intents},
            # Structured outputs do not enforce numeric bounds, so this is
            # clamped below rather than declared with minimum/maximum.
            "confidence": {"type": "number"},
        },
        "required": ["intent", "confidence"],
        "additionalProperties": False,
    }

    def agent(case: Case) -> Prediction:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": case.input}],
        )
        if response.stop_reason == "refusal":
            return Prediction(output=None, error="model declined to answer")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            return Prediction(output=None, error=f"no text block (stop_reason={response.stop_reason})")
        data = json.loads(text)
        confidence = min(1.0, max(0.0, float(data["confidence"])))
        return Prediction(output=data["intent"], confidence=round(confidence, 3))

    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"model to evaluate (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=PILOT_SIZE,
                        help=f"number of cases (default: {PILOT_SIZE})")
    parser.add_argument("--target", type=float, default=0.95,
                        help="precision target the agent must clear (default: 0.95)")
    args = parser.parse_args()

    train, test = fetch("train"), fetch("test")
    intents = sorted({label for _, label in train})

    # Same seed and same slice as banking77_routing.py, so the two runs score
    # the identical case set and the reports can be read side by side.
    pilot = random.Random(SEED).sample(test, args.limit)
    cases = [Case(input=t, expected=l, id=f"msg-{i:04d}") for i, (t, l) in enumerate(pilot)]

    # A burst of 250 concurrent-ish requests will brush rate limits; the SDK
    # backs off and retries on 429 and 5xx.
    client = anthropic.Anthropic(max_retries=5)
    report = evaluate(
        build_agent(client, args.model, intents),
        cases,
        scorer=exact(),
        task=f"Route customer message to the right queue ({args.model})",
        target=args.target,
        workers=WORKERS,
    )
    print(report.markdown())

    here = os.path.dirname(__file__)
    out = os.path.join(here, "banking77_claude_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report.html())

    # Per-case outcomes, so this run can be paired against the baseline later:
    #   from gonogo import compare
    #   compare(json.load(open("banking77_baseline.json")),
    #           json.load(open("banking77_claude.json")))
    data = os.path.join(here, "banking77_claude.json")
    with open(data, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    print(f"\n[html report written to {out}]")
    print(f"[per-case results written to {data}]")


if __name__ == "__main__":
    main()
