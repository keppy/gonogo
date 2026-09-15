"""The runner: point it at your agent and a case file, get a decision."""

from __future__ import annotations

import concurrent.futures as futures
import inspect
from typing import Any, Callable

from .cases import Case, CaseResult, Prediction
from .decide import MIN_USEFUL_COVERAGE, decide
from .report import Report
from .scoring import Score, Scorer, exact

# An agent is any callable from a Case to its answer. Returning a bare value is
# fine; return a Prediction when you can also report confidence.
Agent = Callable[[Case], Any]

# How cases that share a `group` are folded into one trial.
#   none  every case is its own trial (the default, and the only honest choice
#         when cases really are independent)
#   all   one trial per group, passing only if every case in it passed; the
#         group's confidence is the least confident case in it
GROUP_RULES = ("none", "all")


def evaluate(
    agent: Agent,
    cases: list[Case],
    scorer: Scorer | None = None,
    task: str = "task",
    target: float = 0.95,
    level: float = 0.95,
    min_coverage: float = MIN_USEFUL_COVERAGE,
    workers: int = 1,
    group_rule: str = "none",
) -> Report:
    """Run `agent` over `cases`, score it, and decide whether it can ship.

    An agent that raises is recorded as a failed case rather than aborting the
    run: a crash on 3 of 60 cases is a result about reliability, not a reason to
    lose the other 57.

    Cases are treated as independent trials unless `group_rule` says otherwise.
    Several checks on one run of a scenario, or several fields from one
    document, share a draw of the system under test, and an interval that
    counts them separately is narrower than the evidence supports. With
    `group_rule="all"` and a `group` on every case, the pass rate and its
    interval are over groups -- one trial per group, passing only if every case
    in it passed -- and the case count is reported as context.

    A scorer may take `(output, expected)` or `(output, expected, case)`. The
    second form lets one run route different cases to different scorers, or
    read `case.metadata` while scoring.
    """
    if not cases:
        raise ValueError("no cases to evaluate")
    if group_rule not in GROUP_RULES:
        raise ValueError(f"group_rule must be one of {GROUP_RULES}, got {group_rule!r}")
    score = _adapt_scorer(scorer or exact())

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
            passed, value, detail = score(pred.output, case.expected, case)
        except Exception as exc:  # a broken scorer shouldn't look like a broken agent
            results.append(CaseResult(case, pred, passed=False, score=0.0,
                                      detail=f"scorer error: {exc!r}"))
            continue
        results.append(CaseResult(case, pred, passed=passed, score=value, detail=detail))

    metadata = {"scorer": getattr(scorer, "__qualname__", repr(scorer)), "group_rule": group_rule}

    if group_rule == "none":
        decision = decide(
            [(r.confidence, r.passed) for r in results],
            target=target, level=level, min_coverage=min_coverage,
        )
        return Report(task=task, results=results, decision=decision, metadata=metadata)

    groups = group_results(results)
    trials = [(min(r.confidence for r in rs), all(r.passed for r in rs)) for rs in groups.values()]
    decision = decide(trials, target=target, level=level, min_coverage=min_coverage, unit="groups")
    decision.n_groups = len(groups)
    return Report(task=task, results=results, decision=decision, metadata=metadata)


def group_results(results: list[CaseResult]) -> dict[str, list[CaseResult]]:
    """Results by `case.group`, in first-seen order.

    Every case must carry a group. A mix of grouped and ungrouped cases has no
    honest interpretation -- the ungrouped ones would each count as a full
    trial next to groups that count once -- so it is an error, not a guess.
    """
    missing = [r.case.id for r in results if r.case.group is None]
    if len(missing) == len(results):
        raise ValueError("group_rule is set but no case has a group; set Case.group or use group_rule='none'")
    if missing:
        shown = ", ".join(str(m) for m in missing[:3]) + (", ..." if len(missing) > 3 else "")
        raise ValueError(f"group_rule is set but {len(missing)} case(s) have no group: {shown}")
    out: dict[str, list[CaseResult]] = {}
    for r in results:
        out.setdefault(r.case.group, []).append(r)  # type: ignore[arg-type]
    return out


def _adapt_scorer(scorer: Scorer) -> Callable[[Any, Any, Case], Score]:
    """Wrap a scorer so it is always called as (output, expected, case).

    A scorer written as (output, expected) keeps working untouched. One that
    declares a third positional parameter receives the Case as well.
    """
    wants_case = False
    try:
        params = inspect.signature(scorer).parameters.values()
        positional = [p for p in params
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        wants_case = len(positional) >= 3
    except (TypeError, ValueError):
        pass  # builtins and some callables have no inspectable signature
    if wants_case:
        return lambda output, expected, case: scorer(output, expected, case)  # type: ignore[call-arg]
    return lambda output, expected, case: scorer(output, expected)


def _run_one(agent: Agent, case: Case) -> Prediction:
    # Normalization stays inside the try: an agent returning an out-of-range
    # confidence is a per-case failure, not a reason to abort the whole run.
    try:
        raw = agent(case)
        if isinstance(raw, Prediction):
            return raw
        # bool is an int subclass; ("answer", True) is not a confidence of 1.0.
        if (isinstance(raw, tuple) and len(raw) == 2
                and isinstance(raw[1], (int, float)) and not isinstance(raw[1], bool)):
            return Prediction(output=raw[0], confidence=float(raw[1]))
        return Prediction(output=raw)
    except Exception as exc:
        return Prediction(output=None, error=f"{type(exc).__name__}: {exc}")
