"""Scorers: did this prediction match the expected output?

A scorer is just `(prediction_output, expected) -> (passed, score, detail)`.
Bring your own if none of these fit -- that is the whole contract.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

Score = tuple[bool, float, str]


class Scorer(Protocol):
    def __call__(self, output: Any, expected: Any) -> Score: ...


def exact() -> Scorer:
    """Strict equality after string normalization."""

    def score(output: Any, expected: Any) -> Score:
        if isinstance(output, str) and isinstance(expected, str):
            ok = _normalize(output) == _normalize(expected)
        else:
            ok = output == expected
        return ok, 1.0 if ok else 0.0, "" if ok else f"expected {expected!r}, got {output!r}"

    return score


def numeric(tolerance: float = 0.01, relative: bool = True) -> Scorer:
    """Numeric match within a tolerance -- for totals, counts, amounts."""

    def score(output: Any, expected: Any) -> Score:
        try:
            got, want = float(output), float(expected)
        except (TypeError, ValueError):
            return False, 0.0, f"not numeric: {output!r}"
        if math.isnan(got) or math.isnan(want):
            return False, 0.0, "NaN value"
        limit = abs(want) * tolerance if relative else tolerance
        delta = abs(got - want)
        ok = delta <= limit
        return ok, 1.0 if ok else 0.0, "" if ok else f"off by {delta:.4g} (allowed {limit:.4g})"

    return score


def fields(required: list[str] | None = None, tolerance: float = 0.0) -> Scorer:
    """Per-field comparison of two dicts, as in document extraction.

    Passes only when every required field matches. `tolerance` allows a partial
    score to be recorded while still failing the case, which keeps the pass rate
    honest but preserves the signal about how close it got.
    """

    def score(output: Any, expected: Any) -> Score:
        if not isinstance(output, dict) or not isinstance(expected, dict):
            return False, 0.0, "both output and expected must be objects"
        keys = required if required is not None else sorted(expected)
        if not keys:
            return False, 0.0, "no fields to compare"
        wrong = []
        for k in keys:
            want, got = expected.get(k), output.get(k)
            if isinstance(want, (int, float)) and isinstance(got, (int, float)) and tolerance:
                if abs(float(got) - float(want)) > abs(float(want)) * tolerance:
                    wrong.append(k)
            elif _normalize(str(got)) != _normalize(str(want)):
                wrong.append(k)
        frac = 1.0 - len(wrong) / len(keys)
        return not wrong, frac, "" if not wrong else f"wrong fields: {', '.join(wrong)}"

    return score


def set_f1(threshold: float = 1.0) -> Scorer:
    """F1 over two collections, for multi-label or tag extraction."""

    def score(output: Any, expected: Any) -> Score:
        got = {_normalize(str(x)) for x in (output or [])}
        want = {_normalize(str(x)) for x in (expected or [])}
        if not got and not want:
            return True, 1.0, ""
        overlap = len(got & want)
        precision = overlap / len(got) if got else 0.0
        recall = overlap / len(want) if want else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        ok = f1 >= threshold
        return ok, f1, "" if ok else f"F1 {f1:.2f} (precision {precision:.2f}, recall {recall:.2f})"

    return score


def judge(
    rubric: str,
    complete: Callable[[str], str],
    passing_score: int = 4,
    scale: int = 5,
) -> Scorer:
    """LLM-as-judge against a rubric.

    `complete` is any callable that takes a prompt and returns text, so this
    module stays free of provider SDKs and is trivially testable with a stub.

    An unvalidated judge is the most common silent failure in agent evaluation.
    Before trusting one, score a subset by hand and check agreement with
    `judge_agreement` -- a judge that disagrees with you is measuring something
    other than what you care about.
    """
    if not 1 <= passing_score <= scale:
        raise ValueError("passing_score must be within the scale")

    def score(output: Any, expected: Any) -> Score:
        prompt = (
            f"{rubric.strip()}\n\n"
            f"Expected answer:\n{expected}\n\n"
            f"Candidate answer:\n{output}\n\n"
            f"Score the candidate from 1 to {scale}. "
            f"Reply with the number first, then one sentence of justification."
        )
        reply = complete(prompt)
        match = re.search(r"\b([1-9][0-9]?)\b", reply or "")
        if not match:
            return False, 0.0, f"judge returned no score: {(reply or '')[:80]!r}"
        value = min(int(match.group(1)), scale)
        return value >= passing_score, value / scale, f"judge {value}/{scale}: {(reply or '').strip()[:120]}"

    return score


@dataclass
class JudgeValidation:
    """Whether a judge is a usable stand-in for the human who labelled the set."""

    n: int
    agreement: float
    kappa: float
    judge_lenient: int      # judge passed, human failed -- inflates your score
    judge_strict: int       # judge failed, human passed -- deflates it
    usable: bool
    reason: str

    def __str__(self) -> str:
        verdict = "USABLE" if self.usable else "NOT USABLE"
        return (f"judge {verdict}: {self.agreement:.0%} agreement, kappa {self.kappa:.2f} "
                f"on {self.n} hand-labelled cases ({self.reason})")


def validate_judge(
    judge_passes: list[bool],
    human_passes: list[bool],
    min_kappa: float = 0.6,
    min_n: int = 20,
) -> JudgeValidation:
    """Check a judge against hand labels before trusting anything it scored.

    Run this on a subset you labelled yourself. If it comes back not usable,
    every number the judge produced downstream is decoration, and the honest
    move is to fix the rubric rather than report the score.

    Kappa rather than raw agreement is the gate because agreement is inflated
    whenever one class dominates: a judge that passes everything scores 90%
    agreement on a set that is 90% passes, while carrying no information at all.
    """
    stats = judge_agreement(judge_passes, human_passes)
    n = int(stats["n"])
    lenient = sum(1 for j, h in zip(judge_passes, human_passes) if j and not h)
    strict = sum(1 for j, h in zip(judge_passes, human_passes) if h and not j)

    if n == 0:
        return JudgeValidation(0, float("nan"), float("nan"), 0, 0, False,
                               "no hand-labelled cases to check against")
    if n < min_n:
        return JudgeValidation(
            n, stats["agreement"], stats["kappa"], lenient, strict, False,
            f"only {n} hand-labelled cases; label at least {min_n} before trusting the judge",
        )

    kappa = stats["kappa"]
    if kappa != kappa:  # NaN
        return JudgeValidation(n, stats["agreement"], kappa, lenient, strict, False,
                               "kappa is undefined, usually because one label is constant")
    if kappa < min_kappa:
        skew = ("it passes cases you failed" if lenient > strict
                else "it fails cases you passed" if strict > lenient
                else "it disagrees in both directions")
        return JudgeValidation(
            n, stats["agreement"], kappa, lenient, strict, False,
            f"kappa {kappa:.2f} is below {min_kappa:.2f} and {skew}; fix the rubric",
        )
    return JudgeValidation(n, stats["agreement"], kappa, lenient, strict, True,
                           f"kappa {kappa:.2f} clears {min_kappa:.2f}")


def judge_agreement(judge_passes: list[bool], human_passes: list[bool]) -> dict[str, float]:
    """Agreement between a judge and human labels on the same cases.

    Report this alongside any judge-scored result. Raw agreement alone is
    misleading when classes are imbalanced, so Cohen's kappa is included:
    below about 0.6, the judge is not a usable stand-in for a human.
    """
    if len(judge_passes) != len(human_passes):
        raise ValueError("judge and human label lists must be the same length")
    n = len(judge_passes)
    if n == 0:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan")}

    agree = sum(1 for j, h in zip(judge_passes, human_passes) if j == h) / n
    pj, ph = sum(judge_passes) / n, sum(human_passes) / n
    chance = pj * ph + (1 - pj) * (1 - ph)
    kappa = 1.0 if chance == 1.0 else (agree - chance) / (1 - chance)
    return {"n": n, "agreement": agree, "kappa": kappa}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()
