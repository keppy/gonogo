"""Regressions for trial units, selection bias, invalid values and serialization."""
import json
import math

import pytest

from gonogo import Case, Prediction, compare, evaluate, validate_judge
from gonogo.scoring import fields, judge


def test_fields_nonfinite_numbers_never_match():
    scorer = fields(["amount"])
    for got, want in [(float("nan"), 10), (float("inf"), float("inf")),
                      (float("-inf"), 10), (10, float("nan"))]:
        passed, score, detail = scorer({"amount": got}, {"amount": want})
        assert not passed and score == 0 and "amount" in detail


def test_judge_requires_number_first():
    score = judge("Check", lambda _: "Explanation of rubric 5; score 2")
    assert score("x", "y")[0] is False
    assert "no score" in score("x", "y")[2]
    assert judge("Check", lambda _: "  4 - good")("x", "y")[0] is True
    assert judge("Check", lambda _: "4.5 - maybe")("x", "y")[0] is False


def test_structural_positive_only_cannot_validate_judge():
    v = validate_judge([True] * 30, [True] * 30, label_source="structural")
    assert not v.usable and "no failing cases" in v.reason


def test_mixed_missing_confidence_rejected():
    cases = [Case(input=i, expected=i) for i in range(2)]
    with pytest.raises(ValueError, match="mixed missing and present confidence"):
        evaluate(lambda c: Prediction(c.expected, confidence=0.8) if c.input else c.expected, cases)


def grouped_report(fail_first=False):
    cases = [Case(input=(i, j), expected="ok" if not (fail_first and i == 0 and j == 1) else "bad",
                  id=f"check-{j}", group=f"g{i}") for i in range(40) for j in range(2)]
    return evaluate(lambda c: ("ok", 0.4 if c.group == "g0" and c.id == "check-1" else 0.9),
                    cases, target=0.9, level=0.90, group_rule="all")


def test_grouped_curve_uses_group_trials_requested_level_and_group_labels():
    report = grouped_report(fail_first=True)
    points = report.curve()
    assert [p.n_covered for p in points] == [40, 39]
    assert all(p.precision.level == 0.90 for p in points)
    assert points[0].precision.point == pytest.approx(39 / 40)
    md, html = report.markdown(), report.html()
    assert "40 groups" in md and "40 groups" in html
    assert "| Stated confidence | Groups |" in md
    assert "| Groups selected | Precision | Groups to review |" in md
    assert "<th scope=\"col\">Groups</th>" in html
    assert "<th scope=\"col\">Groups to review</th>" in html
    assert "39" in md


def test_grouped_comparison_pairs_groups_not_repeated_case_ids():
    a, b = grouped_report(fail_first=True), grouped_report()
    comparison = compare(a, b)
    assert comparison.unit == "groups" and comparison.n_shared == 40
    assert comparison.only_b == 1
    assert "shared groups" in str(comparison)
    assert compare(a.to_dict(), b.to_dict()).only_b == 1


def test_grouped_comparison_refuses_membership_mismatch_and_ungrouped():
    a, b = grouped_report(), grouped_report()
    payload = b.to_dict()
    payload["cases"][1]["id"] = "different-check"
    with pytest.raises(ValueError, match="different case membership"):
        compare(a, payload)
    with pytest.raises(ValueError, match="grouped trials with ungrouped"):
        compare(a, evaluate(lambda c: "ok", [Case(input=1, expected="ok", id="c")]))


def test_paired_interval_non_degenerate_at_boundaries():
    a = {"cases": [{"id": str(i), "passed": False} for i in range(10)]}
    b = {"cases": [{"id": str(i), "passed": True} for i in range(10)]}
    comparison = compare(a, b)
    assert comparison.difference == 1.0
    assert comparison.low < comparison.difference <= comparison.high
    assert comparison.low > 0.0


def test_report_dict_is_strict_json_when_calibration_missing_or_custom_score_nan():
    cases = [Case(input=i, expected=i) for i in range(5)]
    report = evaluate(lambda c: c.expected, cases, scorer=lambda out, expected: (True, math.nan, ""))
    payload = report.to_dict()
    assert payload["calibration_error"] is None
    assert all(c["score"] is None for c in payload["cases"])
    json.dumps(payload, allow_nan=False)


def test_numeric_nonfinite_operands_and_tolerance_rejected():
    from gonogo.scoring import numeric
    scorer = numeric()
    for got, want in ((5, float("inf")), (float("inf"), float("inf")),
                      (float("nan"), 5)):
        assert scorer(got, want)[0] is False
    with pytest.raises(ValueError, match="finite"):
        numeric(tolerance=float("inf"))
    from gonogo.scoring import fields
    with pytest.raises(ValueError, match="finite"):
        fields(tolerance=float("inf"))


def test_structural_all_fail_and_always_reject_does_not_validate():
    v = validate_judge([False] * 30, [False] * 30, label_source="structural")
    assert not v.usable
    assert "cannot validate semantic" in v.reason


def test_confidence_free_answers_with_explicit_abstention_are_valid():
    cases = [Case(input=i, expected=i) for i in range(2)]
    report = evaluate(lambda c: Prediction(None, abstained=True) if c.input else c.expected,
                      cases)
    assert report.n == 2 and report.n_passed == 1
    assert report.results[1].confidence == 0.0
    assert report.results[0].confidence == 1.0
