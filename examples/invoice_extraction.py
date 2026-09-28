"""A worked example you can run with no API key.

Simulates a document-extraction pilot: 60 real-ish invoices, an agent that is
good but not perfect and reports a confidence, and the resulting decision.

    python examples/invoice_extraction.py

The point of the example is the *shape of the answer*. The agent here gets about
88% of cases right, which looks shippable until you see the interval -- and then
you see that abstaining on its low-confidence cases buys you a defensible
operating point instead.
"""

from __future__ import annotations

import random

from gonogo import Case, Prediction, evaluate
from gonogo.scoring import fields

VENDORS = ["Acme Supply", "Northwind Freight", "Globex Paper", "Initech Legal", "Umbrella Labs"]


def build_cases(n: int = 60, seed: int = 7) -> list[Case]:
    """Stand-in for a real case set. In a pilot this is a JSONL file of your invoices."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        vendor = rng.choice(VENDORS)
        total = round(rng.uniform(40, 9000), 2)
        # Some invoices are genuinely hard: handwritten, multi-page, or foreign currency.
        hard = rng.random() < 0.25
        cases.append(Case(
            id=f"inv-{i:03d}",
            input={"scan": f"invoice_{i:03d}.pdf", "pages": 3 if hard else 1},
            expected={"vendor": vendor, "total": total, "currency": "USD"},
            metadata={"hard": hard},
        ))
    return cases


def agent(case: Case) -> Prediction:
    """A plausible extraction agent: usually right, shakier on hard documents.

    Crucially it reports a calibrated-ish confidence, which is what lets the
    harness find an abstention threshold. An agent that always says 0.99 gives
    you nothing to threshold on.
    """
    rng = random.Random(hash(case.id) & 0xFFFF)
    hard = case.metadata.get("hard", False)
    correct = rng.random() > (0.35 if hard else 0.04)

    expected = case.expected
    if correct:
        output = dict(expected)
    else:
        # Realistic failure modes: wrong total, or a truncated vendor name.
        output = dict(expected)
        if rng.random() < 0.6:
            output["total"] = round(expected["total"] * rng.uniform(1.05, 1.4), 2)
        else:
            output["vendor"] = expected["vendor"].split()[0]

    confidence = rng.uniform(0.35, 0.75) if hard else rng.uniform(0.85, 0.99)
    return Prediction(output=output, confidence=round(confidence, 2))


def main() -> None:
    cases = build_cases()
    report = evaluate(
        agent,
        cases,
        scorer=fields(required=["vendor", "total", "currency"]),
        task="Extract fields from invoice",
        target=0.95,
    )
    print(report.markdown())


if __name__ == "__main__":
    main()
