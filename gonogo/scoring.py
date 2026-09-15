"""Scorers: did this prediction match the expected output?

A scorer is just `(prediction_output, expected) -> (passed, score, detail)`.
Bring your own if none of these fit -- that is the whole contract.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Protocol

Score = tuple[bool, float, str]


class Scorer(Protocol):
    def __call__(self, output: Any, expected: Any) -> Score: ...


class CaseScorer(Protocol):
    """A scorer that also sees the Case, for routing or reading metadata.

    `evaluate` accepts either form and passes the Case only to scorers that
    declare a third positional parameter.
    """
    def __call__(self, output: Any, expected: Any, case: Any) -> Score: ...


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
            if k not in output:
                # A missing field is wrong even if the expected value is falsy;
                # comparing two absences as equal strings would excuse it.
                wrong.append(k)
            elif _is_number(want) and _is_number(got):
                # Numbers compare numerically even at tolerance zero, so an
                # expected 1 is not failed against an output of 1.0.
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
        got = _label_set(output)
        want = _label_set(expected)
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
        value = int(match.group(1))
        if value > scale:
            # A score off the scale means the judge ignored the rubric format.
            # Failing it is safer than clamping it to the most lenient reading.
            return False, 0.0, f"judge returned {value}, outside the 1-{scale} scale: {(reply or '').strip()[:80]}"
        return value >= passing_score, value / scale, f"judge {value}/{scale}: {(reply or '').strip()[:120]}"

    return score


@dataclass
class JudgeValidation:
    """Whether a judge is a usable stand-in for whoever labelled the set.

    `label_source` records who that was, and the wording of every rendering
    follows it. Only `"human"` earns the word human anywhere in the output.
    `"model"` means a second model labelled the set, which measures agreement
    between two machines and nothing more. `"structural"` means deterministic
    checks (duplicates, dangling references, malformed output) that can see a
    judge waving through broken cases but cannot see meaning at all.
    """

    n: int
    agreement: float
    kappa: float
    judge_lenient: int      # judge passed, reference failed -- inflates your score
    judge_strict: int       # judge failed, reference passed -- deflates it
    usable: bool
    reason: str
    label_source: str = "human"
    label_source_note: str = ""

    @property
    def labels_are_human(self) -> bool:
        return self.label_source == "human"

    def describe_labels(self) -> str:
        """The labels, in words that are safe to put in front of a reader."""
        base = {
            "human": "hand-labelled cases",
            "model": "cases labelled by an independent model",
            "structural": "cases labelled by deterministic structural checks",
        }.get(self.label_source, f"cases labelled by {self.label_source}")
        return f"{base} ({self.label_source_note})" if self.label_source_note else base

    def __str__(self) -> str:
        verdict = "USABLE" if self.usable else "NOT USABLE"
        return (f"judge {verdict}: {self.agreement:.0%} agreement, kappa {self.kappa:.2f} "
                f"on {self.n} {self.describe_labels()} ({self.reason})")


def validate_judge(
    judge_passes: list[bool],
    reference_passes: list[bool] | None = None,
    min_kappa: float = 0.6,
    min_n: int = 20,
    *,
    label_source: str = "human",
    label_source_note: str = "",
    human_passes: list[bool] | None = None,
) -> JudgeValidation:
    """Check a judge against reference labels before trusting anything it scored.

    Run this on a subset labelled independently of the judge. If it comes back
    not usable, every number the judge produced downstream is decoration, and
    the honest move is to fix the rubric rather than report the score.

    Say where the labels came from. `label_source="human"` (the default) is the
    only source that supports the claim "validated against humans", and the
    result's wording will not make that claim for any other source. A second
    model's labels are `"model"`; deterministic checks are `"structural"`. The
    parameter used to be called `human_passes`; that name still works and
    warns, because a function that calls every label human is how model labels
    end up described as human validation.

    Kappa rather than raw agreement is the gate because agreement is inflated
    whenever one class dominates: a judge that passes everything scores 90%
    agreement on a set that is 90% passes, while carrying no information at all.

    Structural labels are the exception. They can only fail cases that are
    visibly broken, so a stricter judge is expected and kappa against them is
    bounded. There the gate is leniency alone: a judge that passes a case the
    structural check failed is rubber-stamping, whatever its kappa.
    """
    if human_passes is not None:
        if reference_passes is not None:
            raise TypeError("pass either reference_passes or human_passes, not both")
        warnings.warn(
            "validate_judge(human_passes=...) is deprecated; pass reference_passes and "
            "say where the labels came from with label_source",
            DeprecationWarning, stacklevel=2,
        )
        reference_passes = human_passes
    if reference_passes is None:
        raise TypeError("validate_judge() missing required argument: 'reference_passes'")
    if not label_source or not label_source.strip():
        raise ValueError("label_source must say where the labels came from")

    stats = judge_agreement(judge_passes, reference_passes)
    n = int(stats["n"])
    lenient = sum(1 for j, h in zip(judge_passes, reference_passes) if j and not h)
    strict = sum(1 for j, h in zip(judge_passes, reference_passes) if h and not j)
    human = label_source == "human"
    who = "you" if human else "the reference"

    def result(usable: bool, reason: str) -> JudgeValidation:
        return JudgeValidation(n, stats["agreement"], stats["kappa"], lenient, strict,
                               usable, reason, label_source=label_source,
                               label_source_note=label_source_note)

    if n == 0:
        return result(False, "no labelled cases to check against")
    if n < min_n:
        return result(False, f"only {n} labelled cases; label at least {min_n} before trusting the judge")

    if label_source == "structural":
        if lenient:
            return result(False, (
                f"the judge passed {lenient} case{'s' if lenient != 1 else ''} that a structural "
                f"check failed; that is rubber-stamping, and no kappa excuses it"))
        return result(True, (
            f"the judge passed nothing the structural checks failed; kappa {stats['kappa']:.2f} "
            f"is not the gate here, since structural labels cannot see meaning and a stricter "
            f"judge is expected -- this rules out rubber-stamping, it does not validate the judge"))

    kappa = stats["kappa"]
    if kappa != kappa:  # NaN
        return result(False, (
            "judge and reference labels are both constant and identical, so agreement is "
            "guaranteed by the base rate and carries no information; find harder cases"))
    if kappa < min_kappa:
        skew = (f"it passes cases {who} failed" if lenient > strict
                else f"it fails cases {who} passed" if strict > lenient
                else "it disagrees in both directions")
        return result(False, f"kappa {kappa:.2f} is below {min_kappa:.2f} and {skew}; fix the rubric")

    if human:
        return result(True, f"kappa {kappa:.2f} clears {min_kappa:.2f}")
    return result(True, (
        f"kappa {kappa:.2f} clears {min_kappa:.2f} against {label_source} labels; "
        f"this is agreement with {label_source} labels, not human validation"))


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
    # chance == 1 only when both raters are constant and identical. Agreement
    # there is guaranteed by the base rate alone, so kappa is 0/0: undefined,
    # not perfect. A judge that passed everything on a subset the human also
    # fully passed has demonstrated nothing.
    kappa = float("nan") if chance == 1.0 else (agree - chance) / (1 - chance)
    return {"n": n, "agreement": agree, "kappa": kappa}


def _is_number(value: Any) -> bool:
    # bool is an int subclass but True should compare as a label, not as 1.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _label_set(value: Any) -> set[str]:
    # A bare string is one label, not a collection of characters.
    if value is None:
        return set()
    if isinstance(value, str):
        value = [value]
    return {_normalize(str(x)) for x in value}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()
