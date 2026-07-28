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


@dataclass
class Case:
    """One real case: an input, the expected output, and whatever context helps."""

    input: Any
    expected: Any
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> list["Case"]:
        """Load cases from a JSONL file with `input` and `expected` keys.

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
                cases.append(cls(
                    input=row["input"],
                    expected=row["expected"],
                    id=row.get("id") or f"case-{lineno}",
                    metadata={k: v for k, v in row.items() if k not in ("input", "expected", "id")},
                ))
        if not cases:
            raise ValueError(f"{path}: no cases found")
        return cases

    @staticmethod
    def to_jsonl(cases: list["Case"], path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for c in cases:
                row = {"id": c.id, "input": c.input, "expected": c.expected, **c.metadata}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Prediction:
    """What the agent returned for one case.

    `confidence` is optional but it is what unlocks selective automation; without
    it the harness can only report an overall pass rate.
    """

    output: Any
    confidence: float | None = None
    abstained: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
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
        # A case with no stated confidence is treated as fully confident, which
        # keeps it in every coverage bucket rather than silently excusing it.
        return 1.0 if self.prediction.confidence is None else self.prediction.confidence


def iter_batches(cases: list[Case], size: int) -> Iterator[list[Case]]:
    for i in range(0, len(cases), size):
        yield cases[i:i + size]
