"""Correlated cases: the independent unit is the group, not the case.

Ten runs of one scenario with four checks each is not forty trials. An
interval over forty says more than the evidence does, and can centre on an
average that hides a check failing in every single run.
"""

import json
import tempfile
from pathlib import Path

import pytest

from gonogo import Case, Verdict, evaluate

CHECKS = ["first nodes", "rename", "tangent", "recovery"]


def runs(n_runs: int, fail_check: str | None = None) -> list[Case]:
    """n_runs groups, one case per check in each. `fail_check` fails in every run."""
    out = []
    for r in range(n_runs):
        for chk in CHECKS:
            out.append(Case(input=chk, expected="ok" if chk != fail_check else "never",
                            id=chk, group=f"run-{r}"))
    return out


agent = lambda c: "ok"


class TestGroupField:
    def test_group_defaults_to_none(self):
        assert Case(input=1, expected=1).group is None

    def test_from_jsonl_reads_group_and_keeps_it_out_of_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.jsonl"
            p.write_text(json.dumps({"input": 1, "expected": 1, "group": "r1", "site": "x"}) + "\n"
                         + json.dumps({"input": 2, "expected": 2}) + "\n", encoding="utf-8")
            a, b = Case.from_jsonl(p)
        assert a.group == "r1" and a.metadata == {"site": "x"}
        assert b.group is None

    def test_group_round_trips_through_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.jsonl"
            Case.to_jsonl([Case(input=1, expected=1, id="a", group="g")], p)
            (back,) = Case.from_jsonl(p)
        assert back.group == "g"

    def test_numeric_group_is_stored_as_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cases.jsonl"
            p.write_text(json.dumps({"input": 1, "expected": 1, "group": 7}) + "\n", encoding="utf-8")
            (c,) = Case.from_jsonl(p)
        assert c.group == "7"


class TestGroupRuleNone:
    def test_default_ignores_groups_and_counts_cases(self):
        report = evaluate(agent, runs(10, fail_check="recovery"), target=0.80)
        assert not report.grouped
        assert report.decision.pass_rate.n == 40
        assert report.n_passed == 30

    def test_unknown_rule_is_rejected(self):
        with pytest.raises(ValueError):
            evaluate(agent, runs(2), group_rule="mean")


class TestGroupRuleAll:
    def test_one_check_failing_every_run_is_zero_of_ten_not_seventy_five_percent(self):
        report = evaluate(agent, runs(10, fail_check="recovery"), target=0.80, group_rule="all")
        d = report.decision
        assert report.grouped and d.n_groups == 10
        assert d.pass_rate.n == 10
        assert d.pass_rate.point == 0.0
        assert d.verdict is Verdict.DO_NOT_AUTOMATE
        # the per-case picture is still there, as context
        assert report.n == 40 and report.n_passed == 30
        assert report.n_groups_passed() == 0

    def test_all_passing_gives_interval_over_groups(self):
        report = evaluate(agent, runs(10), target=0.80, group_rule="all")
        d = report.decision
        assert d.pass_rate.n == 10 and d.pass_rate.point == 1.0
        # Wilson 10/10 at 95%: lower bound ~0.72, not the ~0.91 that 40/40 would give
        assert 0.70 < d.pass_rate.low < 0.75
        assert d.unit == "groups"

    def test_wording_says_groups(self):
        report = evaluate(agent, runs(10), target=0.80, group_rule="all")
        assert "groups" in report.decision.reason
        assert "10 groups" in report.summary()
        md = report.markdown()
        assert "Groups evaluated | 10" in md
        assert "not 40 cases" in md

    def test_group_confidence_is_the_least_confident_case(self):
        def agent_conf(c):
            return "ok", (0.4 if c.id == "recovery" else 0.95)
        report = evaluate(agent_conf, runs(5), target=0.80, group_rule="all")
        # every group's trial confidence should be 0.4, so no confident subset exists
        assert all(min(r.confidence for r in rs) == 0.4 for rs in report.groups().values())

    def test_needed_n_is_in_groups(self):
        # 9 of 10 groups pass against a 0.8 target: point estimate clears, interval does not
        cases = runs(10)
        for c in cases:
            if c.group == "run-0" and c.id == "recovery":
                c.expected = "never"
        report = evaluate(agent, cases, target=0.80, group_rule="all")
        d = report.decision
        assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert d.needed_n is not None and d.needed_n > 10
        assert any("groups would be needed" in n for n in d.notes)

    def test_recurring_ids_render_as_k_of_n(self):
        report = evaluate(agent, runs(10, fail_check="recovery"), target=0.80, group_rule="all")
        rec = dict((cid, (k, n)) for cid, k, n in report.recurring())
        assert rec["recovery"] == (0, 10)
        assert rec["first nodes"] == (10, 10)
        md = report.markdown()
        assert "`recovery` | 0 of 10 groups" in md
        html = report.html()
        assert "0 of 10 groups" in html and "Per case, across groups" in html

    def test_to_dict_carries_groups(self):
        report = evaluate(agent, runs(3), target=0.80, group_rule="all")
        d = report.to_dict()
        assert d["n_groups"] == 3 and d["n_groups_passed"] == 3
        assert {c["group"] for c in d["cases"]} == {"run-0", "run-1", "run-2"}


class TestGroupErrors:
    def test_rule_without_any_groups_is_an_error(self):
        cases = [Case(input=i, expected="ok", id=f"c{i}") for i in range(5)]
        with pytest.raises(ValueError, match="no case has a group"):
            evaluate(agent, cases, group_rule="all")

    def test_mixed_grouped_and_ungrouped_is_an_error(self):
        cases = runs(2) + [Case(input="x", expected="ok", id="loose")]
        with pytest.raises(ValueError, match="have no group"):
            evaluate(agent, cases, group_rule="all")
