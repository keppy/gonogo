"""Results scored somewhere else, loaded from JSONL.

`evaluate` runs the agent and scores it. A harness in another language, or one
that scores as it runs, already holds the outcomes. This loads them as the same
`CaseResult`s, so `gonogo decide results.jsonl` takes the one decision path.
"""

from __future__ import annotations

from pathlib import Path

from .cases import ABSTAIN, ANSWER, ERROR, Case, CaseResult, Prediction, _check_action, read_jsonl_rows

# Top-level keys that are result fields; anything else lands in case metadata.
_RESERVED = ("id", "passed", "confidence", "expected_action", "action", "group", "score", "detail", "error")


def results_from_jsonl(path: str | Path) -> list[CaseResult]:
    """Load scored results: one JSON object per line.

    Keys: `passed` (required, true/false); `id` (default `case-<line>`);
    `confidence` in [0, 1] (all rows or none); `expected_action` and `action`
    (default "answer"); `group`; `score`; `detail`; `error`. The rules `evaluate`
    applies after scoring hold here too: a case that expects a non-answer passes
    only on that action, an abstention on an answer case is a deferral at
    confidence 0.0, and any other non-answer on an answer case fails. A `passed`
    that contradicts those is refused with its line number, not corrected.

    `error` (a non-empty string) marks a case where the agent or the harness
    crashed: the case `evaluate` records for an agent that raised, failed at
    score 0 with detail `agent error: <error>`. `action: "error"`, which is what
    `decide --json` writes for such a case, means the same. An errored row
    cannot pass. Its confidence is optional and stays out of the
    mixed-confidence check; left out, it sits at 1.0 in a run with confidence,
    as a crash does in `evaluate`.
    """
    path = Path(path)
    results: list[CaseResult] = []
    for lineno, row in read_jsonl_rows(path):
        where = f"{path}:{lineno}"
        passed = row.get("passed")
        if not isinstance(passed, bool):
            raise ValueError(f"{where}: passed must be true or false, got {passed!r}")
        try:
            expected_action = _check_action(row.get("expected_action", ANSWER), "expected_action")
            action = _check_action(row.get("action", ANSWER), "action")
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        confidence = row.get("confidence")
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))):
            raise ValueError(f"{where}: confidence must be a number in [0, 1], got {confidence!r}")

        error = row.get("error")
        if error is not None:
            if not isinstance(error, str) or not error:
                raise ValueError(f"{where}: error must be a non-empty string, got {error!r}")
            if "action" in row and action != ERROR:
                raise ValueError(f"{where}: error is set, but action is {action!r}; an errored case "
                                 f"took no action (leave action out, or write {ERROR!r})")
        elif action == ERROR:
            error = ERROR
        if error is not None:
            if passed:
                raise ValueError(f"{where}: an errored case cannot pass")
            action = ANSWER  # what evaluate's crash records; the error is what marks it
        elif expected_action != ANSWER:
            expect = action == expected_action
            if passed != expect:
                raise ValueError(
                    f"{where}: passed is {str(passed).lower()}, but action {action!r} against "
                    f"expected_action {expected_action!r} means {str(expect).lower()}")
        elif action == ABSTAIN:
            if passed:
                raise ValueError(f"{where}: an abstention on a case that expects an answer cannot pass")
            confidence = 0.0  # a deferral sits at the bottom of every threshold
        elif action != ANSWER and passed:
            raise ValueError(f"{where}: action {action!r} on a case that expects an answer cannot pass")

        group = row.get("group")
        score = row.get("score", 1.0 if passed else 0.0)
        try:
            prediction = Prediction(output=None, confidence=None if confidence is None else float(confidence),
                                    action=action, error=error)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        results.append(CaseResult(
            case=Case(input=None, expected=None, id=str(row.get("id") or f"case-{lineno}"),
                      metadata={k: v for k, v in row.items() if k not in _RESERVED},
                      group=None if group is None else str(group), expected_action=expected_action),
            prediction=prediction,
            passed=passed,
            score=float(score),
            detail=str(row.get("detail", f"agent error: {error}" if error is not None else "")),
        ))
    if not results:
        raise ValueError(f"{path}: no results found")
    return results
