"""The runner: point it at your agent and a case file, get a decision."""

from __future__ import annotations

import concurrent.futures as futures
from typing import Any, Callable

from .cases import Case, CaseResult, Prediction
from .decide import MIN_USEFUL_COVERAGE, decide
from .report import Report
from .scoring import Scorer, exact

# An agent is any callable from a Case to its answer. Returning a bare value is
# fine; return a Prediction when you can also report confidence.
Agent = Callable[[Case], Any]


def evaluate(
    agent: Agent,
    cases: list[Case],
    scorer: Scorer | None = None,
    task: str = "task",
    target: float = 0.95,
    level: float = 0.95,
    min_coverage: float = MIN_USEFUL_COVERAGE,
    workers: int = 1,
) -> Report:
    """Run `agent` over `cases`, score it, and decide whether it can ship.

    An agent that raises is recorded as a failed case rather than aborting the
    run: a crash on 3 of 60 cases is a result about reliability, not a reason to
    lose the other 57.
    """
    if not cases:
        raise ValueError("no cases to evaluate")
    scorer = scorer or exact()

    if workers > 1:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            predictions = list(pool.map(lambda c: _run_one(agent, c), cases))
    else:
        predictions = [_run_one(agent, c) for c in cases]

    results: list[CaseResult] = []
    for case, pred in zip(cases, predictions):
        if pred.error is not None:
            results.append(CaseResult(case, pred, passed=False, score=0.0,
                                      detail=f"agent error: {pred.error}"))
            continue
        if pred.abstained:
            # An explicit abstention is not a correct answer, but it is honest.
            # It lands at zero confidence so thresholding defers it first.
            results.append(CaseResult(
                case, Prediction(pred.output, confidence=0.0, abstained=True),
                passed=False, score=0.0, detail="agent abstained",
            ))
            continue
        try:
            passed, score, detail = scorer(pred.output, case.expected)
        except Exception as exc:  # a broken scorer shouldn't look like a broken agent
            results.append(CaseResult(case, pred, passed=False, score=0.0,
                                      detail=f"scorer error: {exc!r}"))
            continue
        results.append(CaseResult(case, pred, passed=passed, score=score, detail=detail))

    decision = decide(
        [(r.confidence, r.passed) for r in results],
        target=target, level=level, min_coverage=min_coverage,
    )
    return Report(task=task, results=results, decision=decision,
                  metadata={"scorer": getattr(scorer, "__qualname__", repr(scorer))})


def _run_one(agent: Agent, case: Case) -> Prediction:
    try:
        raw = agent(case)
    except Exception as exc:
        return Prediction(output=None, error=f"{type(exc).__name__}: {exc}")
    if isinstance(raw, Prediction):
        return raw
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], (int, float)):
        return Prediction(output=raw[0], confidence=float(raw[1]))
    return Prediction(output=raw)
