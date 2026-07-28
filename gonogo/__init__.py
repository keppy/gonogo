"""gonogo -- decide whether an agent is good enough to ship.

Not an eval framework. A decision harness for pilot-scale case sets, where the
honest answer is often "not enough evidence yet" and sometimes "don't automate
this at all".
"""

from .cases import Case, CaseResult, Prediction
from .decide import Decision, Verdict, decide
from .evaluate import evaluate
from .report import Report
from .scoring import (
    exact,
    fields,
    judge,
    judge_agreement,
    numeric,
    set_f1,
)
from .stats import (
    Interval,
    OperatingPoint,
    best_operating_point,
    expected_calibration_error,
    reliability_table,
    required_n,
    risk_coverage,
    wilson,
)

__version__ = "0.1.0"

__all__ = [
    "Case", "CaseResult", "Prediction",
    "Decision", "Verdict", "decide",
    "evaluate", "Report",
    "exact", "fields", "judge", "judge_agreement", "numeric", "set_f1",
    "Interval", "OperatingPoint", "best_operating_point",
    "expected_calibration_error", "reliability_table", "required_n",
    "risk_coverage", "wilson",
]
