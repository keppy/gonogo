"""gonogo -- decide whether an agent is good enough to ship.

Not an eval framework. A decision harness for pilot-scale case sets, where the
honest answer is often "not enough evidence yet" and sometimes "don't automate
this at all".
"""

from .cases import Case, CaseResult, Prediction
from .compare import Comparison, compare, outcomes
from .decide import Decision, Verdict, decide
from .evaluate import GROUP_RULES, evaluate, group_results
from .report import Report
from .scoring import (
    CaseScorer,
    JudgeValidation,
    Scorer,
    exact,
    fields,
    judge,
    judge_agreement,
    numeric,
    set_f1,
    validate_judge,
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

__version__ = "0.2.0"

__all__ = [
    "Case", "CaseResult", "Prediction",
    "Comparison", "compare", "outcomes",
    "Decision", "Verdict", "decide",
    "evaluate", "group_results", "GROUP_RULES", "Report",
    "exact", "fields", "judge", "judge_agreement", "numeric", "set_f1",
    "Scorer", "CaseScorer",
    "JudgeValidation", "validate_judge",
    "Interval", "OperatingPoint", "best_operating_point",
    "expected_calibration_error", "reliability_table", "required_n",
    "risk_coverage", "wilson",
]
