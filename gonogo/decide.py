"""The verdict layer.

Every other eval tool stops at a number. This module turns the number into a
deployment decision, and refuses to when the evidence is too thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .stats import (
    Interval,
    OperatingPoint,
    best_operating_point,
    expected_calibration_error,
    required_n,
    wilson,
)


class Verdict(str, Enum):
    AUTOMATE = "AUTOMATE"
    AUTOMATE_WITH_REVIEW = "AUTOMATE WITH REVIEW"
    ASSIST_ONLY = "ASSIST ONLY"
    DO_NOT_AUTOMATE = "DO NOT AUTOMATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"


# Above this, miscalibrated confidence makes an abstention threshold meaningless.
ECE_UNUSABLE = 0.15
# An operating point covering less than this isn't worth the plumbing.
MIN_USEFUL_COVERAGE = 0.25
# Below this pass rate, tuning won't save it.
ASSIST_FLOOR = 0.50


@dataclass
class Decision:
    verdict: Verdict
    reason: str
    pass_rate: Interval
    target: float
    operating_point: OperatingPoint | None = None
    calibration_error: float = float("nan")
    needed_n: int | None = None
    notes: list[str] = field(default_factory=list)
    # Set when the trials were groups of cases rather than cases. Then
    # pass_rate.n and needed_n count groups, and the case count is context.
    n_groups: int | None = None

    @property
    def unit(self) -> str:
        return "groups" if self.n_groups is not None else "cases"

    @property
    def can_automate(self) -> bool:
        return self.verdict in (Verdict.AUTOMATE, Verdict.AUTOMATE_WITH_REVIEW)


def decide(
    results: list[tuple[float, bool]],
    target: float = 0.95,
    level: float = 0.95,
    min_coverage: float = MIN_USEFUL_COVERAGE,
    unit: str = "cases",
) -> Decision:
    """Turn per-trial (confidence, passed) results into a deployment decision.

    `unit` is the word for one trial in the wording -- "cases" normally,
    "groups" when `evaluate` folded correlated cases into one trial each.

    The ordering is deliberate. We ask "is the whole task good enough?" first,
    then "is some confident subset good enough?", and only then do we consider
    whether the sample is simply too small to say. A tool that reports
    INSUFFICIENT EVIDENCE when it should report DO NOT AUTOMATE is just as
    dishonest as one that reports a bare point estimate.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be in (0, 1), got {target}")

    n = len(results)
    if n == 0:
        return Decision(
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            reason="no cases were evaluated",
            pass_rate=wilson(0, 0, level),
            target=target,
        )

    passed = sum(1 for _, p in results if p)
    rate = wilson(passed, n, level)
    notes: list[str] = []

    has_confidence = len({c for c, _ in results}) > 1
    # With no confidence signal (every case defaulted to the same value), an
    # ECE would just measure |accuracy - 1| and read as miscalibration the
    # agent never claimed. Leave it NaN so reports omit the line entirely.
    ece = expected_calibration_error(results) if has_confidence else float("nan")
    if not has_confidence:
        notes.append(
            "Every case reported the same confidence, so no abstention threshold "
            "can be derived. Emit a real per-case confidence to unlock selective automation."
        )

    # 1. The whole task clears the bar on its own.
    if rate.low >= target:
        return Decision(
            verdict=Verdict.AUTOMATE,
            reason=(f"pass rate {rate} clears the {target:.0%} target across all {unit} "
                    f"at the {level:.0%} level"),
            pass_rate=rate, target=target, calibration_error=ece, notes=notes,
        )

    # 2. A confident subset clears the bar, with the remainder going to a human.
    #
    # Note that a high calibration error does NOT disqualify a threshold. The
    # precision at each cut point is measured directly from the results and
    # carries its own interval, so it stands whether or not the confidence
    # number is on a meaningful scale. Miscalibration only means the threshold
    # value is a cut point rather than a probability -- worth saying out loud,
    # not worth throwing away a working operating point over.
    if has_confidence:
        if ece > ECE_UNUSABLE:
            notes.append(
                f"Calibration error {ece:.2f} exceeds {ECE_UNUSABLE:.2f}: the confidence score "
                f"ranks cases usefully but its scale is not a probability. Treat any threshold "
                f"below as an opaque cut point, and recalibrate before reading it as a percentage."
            )
        point = best_operating_point(results, target, min_coverage, level)
        if point is not None:
            notes.append(
                f"The {point.threshold:.2f} threshold was chosen by searching this same case "
                f"set, so its precision is optimistically biased. Re-measure it on fresh cases "
                f"before relying on it."
            )
            return Decision(
                verdict=Verdict.AUTOMATE_WITH_REVIEW,
                reason=(f"overall pass rate {rate} misses the {target:.0%} target, but "
                        f"abstaining below confidence {point.threshold:.2f} reaches "
                        f"{point.precision.point:.1%} precision on {point.coverage:.0%} of {unit}"),
                pass_rate=rate, target=target, operating_point=point,
                calibration_error=ece, notes=notes,
            )
        near = best_operating_point(results, target, 0.0, level)
        if near is not None:
            notes.append(
                f"A threshold of {near.threshold:.2f} would hit the target but only covers "
                f"{near.coverage:.0%} of {unit}, below the {min_coverage:.0%} floor."
            )

    # 3. The point estimate looks good but the sample can't support the claim.
    if rate.point >= target:
        needed = required_n(rate.point, target, level)
        return Decision(
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            reason=(f"observed {rate.point:.1%} on {n} {unit} is above the {target:.0%} target, "
                    f"but the {level:.0%} interval reaches down to {rate.low:.1%}"),
            pass_rate=rate, target=target, calibration_error=ece, needed_n=needed,
            notes=notes + ([f"At this rate, about {needed} {unit} would be needed to claim "
                            f"{target:.0%}; you have {n}."] if needed else []),
        )

    # 4. Too weak to automate, but a human-in-the-loop assist may still pay off.
    if rate.point >= ASSIST_FLOOR:
        return Decision(
            verdict=Verdict.ASSIST_ONLY,
            reason=(f"pass rate {rate} is well short of the {target:.0%} target and no "
                    f"confident subset reaches it; useful as a draft-generator, not as "
                    f"an unattended step"),
            pass_rate=rate, target=target, calibration_error=ece, notes=notes,
        )

    return Decision(
        verdict=Verdict.DO_NOT_AUTOMATE,
        reason=(f"pass rate {rate} is below {ASSIST_FLOOR:.0%}; this workflow is not a fit "
                f"for automation as currently scoped"),
        pass_rate=rate, target=target, calibration_error=ece, notes=notes,
    )
