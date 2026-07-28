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

    @property
    def can_automate(self) -> bool:
        return self.verdict in (Verdict.AUTOMATE, Verdict.AUTOMATE_WITH_REVIEW)


def decide(
    results: list[tuple[float, bool]],
    target: float = 0.95,
    level: float = 0.95,
    min_coverage: float = MIN_USEFUL_COVERAGE,
) -> Decision:
    """Turn per-case (confidence, passed) results into a deployment decision.

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
    ece = expected_calibration_error(results)
    notes: list[str] = []

    has_confidence = len({c for c, _ in results}) > 1
    if not has_confidence:
        notes.append(
            "Every case reported the same confidence, so no abstention threshold "
            "can be derived. Emit a real per-case confidence to unlock selective automation."
        )

    # 1. The whole task clears the bar on its own.
    if rate.low >= target:
        return Decision(
            verdict=Verdict.AUTOMATE,
            reason=(f"pass rate {rate} clears the {target:.0%} target across all cases "
                    f"at the {level:.0%} level"),
            pass_rate=rate, target=target, calibration_error=ece, notes=notes,
        )

    # 2. A confident subset clears the bar, with the remainder going to a human.
    point = None
    if has_confidence:
        if ece > ECE_UNUSABLE:
            notes.append(
                f"Calibration error {ece:.2f} exceeds {ECE_UNUSABLE:.2f}: stated confidence "
                f"does not track accuracy, so thresholding on it is unreliable."
            )
        else:
            point = best_operating_point(results, target, min_coverage, level)
            if point is not None:
                return Decision(
                    verdict=Verdict.AUTOMATE_WITH_REVIEW,
                    reason=(f"overall pass rate {rate} misses the {target:.0%} target, but "
                            f"abstaining below confidence {point.threshold:.2f} reaches "
                            f"{point.precision.point:.1%} precision on {point.coverage:.0%} of cases"),
                    pass_rate=rate, target=target, operating_point=point,
                    calibration_error=ece, notes=notes,
                )
            near = best_operating_point(results, target, 0.0, level)
            if near is not None:
                notes.append(
                    f"A threshold of {near.threshold:.2f} would hit the target but only covers "
                    f"{near.coverage:.0%} of cases, below the {min_coverage:.0%} floor."
                )

    # 3. The point estimate looks good but the sample can't support the claim.
    if rate.point >= target:
        needed = required_n(rate.point, target, level)
        return Decision(
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            reason=(f"observed {rate.point:.1%} on {n} cases is above the {target:.0%} target, "
                    f"but the {level:.0%} interval reaches down to {rate.low:.1%}"),
            pass_rate=rate, target=target, calibration_error=ece, needed_n=needed,
            notes=notes + ([f"At this rate, about {needed} cases would be needed to claim "
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
