import json

import pytest

from gonogo import Case, compare, evaluate
from gonogo.compare import outcomes


def report_from(passes: dict[str, bool], task="t"):
    """Build a real Report whose per-case outcomes match `passes`."""
    cases = [Case(input=cid, expected="yes", id=cid) for cid in passes]
    return evaluate(lambda c: "yes" if passes[c.id] else "no", cases, task=task)


class TestPairing:
    def test_pairs_on_case_id(self):
        a = report_from({"1": True, "2": False, "3": True})
        b = report_from({"3": True, "1": False, "2": True})  # different order
        c = compare(a, b)
        assert c.n_shared == 3
        assert c.rate_a == pytest.approx(2 / 3)
        assert c.rate_b == pytest.approx(2 / 3)

    def test_ignores_cases_only_one_agent_ran(self):
        a = report_from({"1": True, "2": True, "extra": False})
        b = report_from({"1": True, "2": False, "other": True})
        c = compare(a, b)
        assert c.n_shared == 2

    def test_works_from_serialized_reports(self):
        # The realistic case: today's run against a JSON file from last week.
        a = json.loads(json.dumps(report_from({"1": True, "2": False}).to_dict()))
        b = json.loads(json.dumps(report_from({"1": True, "2": True}).to_dict()))
        c = compare(a, b)
        assert c.n_shared == 2 and c.only_b == 1

    def test_to_dict_carries_per_case_outcomes(self):
        d = report_from({"1": True, "2": False}).to_dict()
        assert {c["id"] for c in d["cases"]} == {"1", "2"}
        assert {c["id"]: c["passed"] for c in d["cases"]} == {"1": True, "2": False}

    def test_no_shared_ids_raises(self):
        with pytest.raises(ValueError, match="share no case ids"):
            compare(report_from({"a": True}), report_from({"b": True}))

    def test_duplicate_ids_raise(self):
        with pytest.raises(ValueError, match="duplicate case id"):
            outcomes({"cases": [{"id": "x", "passed": True}, {"id": "x", "passed": False}]})

    def test_rejects_payload_without_per_case_data(self):
        with pytest.raises(ValueError, match="per-case results"):
            outcomes({"task": "t", "verdict": "AUTOMATE"})


class TestMcNemar:
    def test_concordant_cases_carry_no_information(self):
        # Same disagreement pattern, wildly different numbers of agreements.
        # McNemar must ignore the agreements entirely.
        few = compare(report_from({**{str(i): True for i in range(4)}, "x": True, "y": False}),
                      report_from({**{str(i): True for i in range(4)}, "x": False, "y": True}))
        many = compare(report_from({**{str(i): True for i in range(400)}, "x": True, "y": False}),
                       report_from({**{str(i): True for i in range(400)}, "x": False, "y": True}))
        assert few.p_value == pytest.approx(many.p_value)

    def test_total_agreement_is_not_evidence(self):
        a = report_from({"1": True, "2": False})
        c = compare(a, a)
        assert c.discordant == 0 and c.p_value == 1.0 and not c.significant
        assert c.low == 0.0 and c.high == 0.0

    def test_lopsided_disagreement_is_significant(self):
        # B fixes 15 cases A failed and breaks none.
        base = {str(i): True for i in range(50)}
        a = report_from({**base, **{f"d{i}": False for i in range(15)}})
        b = report_from({**base, **{f"d{i}": True for i in range(15)}})
        c = compare(a, b)
        assert c.only_b == 15 and c.only_a == 0
        assert c.p_value < 0.001 and c.significant
        assert c.difference > 0

    def test_balanced_disagreement_is_not_significant(self):
        base = {str(i): True for i in range(50)}
        a = report_from({**base, **{"x": True, "y": False}})
        b = report_from({**base, **{"x": False, "y": True}})
        c = compare(a, b)
        assert c.only_a == 1 and c.only_b == 1
        assert c.p_value == 1.0 and not c.significant

    def test_exact_p_matches_hand_computed_value(self):
        # 5 disagreements all favoring B: two-sided p = 2 * (1/2)^5 = 0.0625
        base = {str(i): True for i in range(20)}
        a = report_from({**base, **{f"d{i}": False for i in range(5)}})
        b = report_from({**base, **{f"d{i}": True for i in range(5)}})
        assert compare(a, b).p_value == pytest.approx(0.0625)

    def test_direction_is_b_minus_a(self):
        a = report_from({"1": False, "2": True})
        b = report_from({"1": True, "2": True})
        c = compare(a, b)
        assert c.difference > 0 and c.rate_b > c.rate_a

    def test_exact_flag_and_summary(self):
        a = report_from({"1": True, "2": False})
        b = report_from({"1": True, "2": True})
        c = compare(a, b)
        assert c.exact
        assert "McNemar p" in c.summary()
        assert "shared cases" in str(c)

    def test_rejects_invalid_level(self):
        a = report_from({"1": True})
        with pytest.raises(ValueError):
            compare(a, a, level=1.5)
