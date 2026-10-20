"""Real cases, loaded from disk.

A case set is the pilot's ground truth. It is deliberately a plain JSONL file
so a domain expert can build and audit one in a spreadsheet without touching
Python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# The action that produces an output for the scorer. Every other action
# ("abstain", "ask", "wait", or any name you choose) is a decision not to
# answer, and is graded by whether it matches the case's expected action.
ANSWER = "answer"
ABSTAIN = "abstain"

# Top-level JSONL keys that are Case fields rather than metadata.
_RESERVED = ("input", "expected", "id", "group", "expected_action")


def _check_action(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


@dataclass
class Case:
    """One real case: an input, the expected output, and whatever context helps."""

    input: Any
    expected: Any
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Cases that share a group came from one draw of the system under test:
    # one run of a scenario, one document's fields, one conversation's turns.
    # They are not independent trials. With evaluate(group_rule="all") the
    # pass rate and its interval are over groups, and the case count is context.
    group: str | None = None
    # What the agent should do with this case. "answer" (the default) means
    # produce an output for the scorer. Anything else -- "abstain", "ask",
    # "wait" -- means the correct behaviour is to not answer, and the case
    # passes only if the agent takes exactly that action; the scorer and
    # `expected` are not consulted.
    expected_action: str = ANSWER

    def __post_init__(self) -> None:
        _check_action(self.expected_action, "expected_action")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> list["Case"]:
        """Load cases from a JSONL file with `input` and `expected` keys.

        An optional `group` key marks cases that share one draw (see `group`).
        An optional `expected_action` key says the correct behaviour is not to
        answer (see `expected_action`); absent means "answer".
        Any other keys land in `metadata`, so you can slice a report by
        customer, document type, or difficulty later.
        """
        path = Path(path)
        cases: list[Case] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
                for required in ("input", "expected"):
                    if required not in row:
                        raise ValueError(f"{path}:{lineno}: missing required key {required!r}")
                group = row.get("group")
                action = row.get("expected_action", ANSWER)
                try:
                    _check_action(action, "expected_action")
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
                cases.append(cls(
                    input=row["input"],
                    expected=row["expected"],
                    id=row.get("id") or f"case-{lineno}",
                    metadata={k: v for k, v in row.items() if k not in _RESERVED},
                    group=None if group is None else str(group),
                    expected_action=action,
                ))
        if not cases:
            raise ValueError(f"{path}: no cases found")
        return cases

    @staticmethod
    def to_jsonl(cases: list["Case"], path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for c in cases:
                row = {"id": c.id, "input": c.input, "expected": c.expected, **c.metadata}
                if c.group is not None:
                    row["group"] = c.group
                if c.expected_action != ANSWER:  # omitted so legacy files round-trip unchanged
                    row["expected_action"] = c.expected_action
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Prediction:
    """What the agent returned for one case.

    `confidence` is optional but it is what unlocks selective automation; without
    it the harness can only report an overall pass rate.

    `action` is what the agent chose to do: "answer" (the default) or a
    decision not to answer, such as "abstain", "ask" or "wait". `abstained`
    means "did not answer" and is kept in step with `action`: any action other
    than "answer" sets it, and `abstained=True` on its own reads as
    `action="abstain"`.
    """

    output: Any
    confidence: float | None = None
    abstained: bool = False
    error: str | None = None
    action: str = ANSWER

    def __post_init__(self) -> None:
        _check_action(self.action, "action")
        if self.action != ANSWER:
            self.abstained = True
        elif self.abstained:
            self.action = ABSTAIN
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class CaseResult:
    """A scored case: what was expected, what came back, and whether it passed."""

    case: Case
    prediction: Prediction
    passed: bool
    score: float
    detail: str = ""

    @property
    def confidence(self) -> float:
        # Only an entirely confidence-free set may use this fallback. Mixed
        # missing/present confidence is rejected by the evaluation/report paths.
        return 1.0 if self.prediction.confidence is None else self.prediction.confidence


def answered(results: list[CaseResult]) -> list[CaseResult]:
    """Results whose confidence is the agent's own, for the mixed-confidence check.

    That is every non-errored result except a plain abstention on a case that
    expects an answer: that one is a deferral and carries a synthesized 0.0.
    Every other action -- an answer, a non-answer on a case that expects one,
    or a decision such as "wait" or "ask" on a case that expected an answer --
    is a decision the agent made, so its confidence counts.
    """
    return [r for r in results if not r.prediction.error
            and not (r.prediction.action == ABSTAIN and r.case.expected_action == ANSWER)]


def iter_batches(cases: list[Case], size: int) -> Iterator[list[Case]]:
    for i in range(0, len(cases), size):
        yield cases[i:i + size]
