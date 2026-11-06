"""0.4.6: four behaviours the TypeScript port (gonogo-ts) surfaced and copied faithfully.

A. A `--group-rule none` report of grouped rows can be compared (pairs on group and id).
B. A point exactly at the target says so, and why no sample size helps.
C. Cut points print so they reproduce themselves (0.625, not "0.62"; 62.5%, not "62%").
D. Abstentions in a confidence-free run keep their threshold but get no calibration error.
"""

import json
import math

import pytest

from gonogo import Verdict, compare, decide, report_from_results, results_from_jsonl
from gonogo.cli import main
from gonogo.stats import fmt_cut, fmt_share


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------- A

def _grouped_rows(fail_ids):
    # three groups, the same three case ids in each: ids repeat across groups by design
    return [{"id": f"check-{i}", "group": f"g{g}", "passed": (g, i) not in fail_ids}
            for g in range(3) for i in range(3)]


def test_a_group_rule_none_reports_of_grouped_rows_compare_row_by_row(tmp_path):
    a = report_from_results(results_from_jsonl(_write(tmp_path / "a.jsonl", _grouped_rows({(0, 0), (1, 1)}))),
                            target=0.8, group_rule="none")
    b = report_from_results(results_from_jsonl(_write(tmp_path / "b.jsonl", _grouped_rows({(1, 1)}))),
                            target=0.8, group_rule="none")
    for left, right in ((a.to_dict(), b.to_dict()), (a, b)):   # payloads and live reports
        c = compare(left, right)
        assert c.unit == "cases" and c.n_shared == 9
        assert (c.both_passed, c.both_failed, c.only_a, c.only_b) == (7, 1, 0, 1)


def test_a_folded_reports_still_compare_by_group(tmp_path):
    a = report_from_results(results_from_jsonl(_write(tmp_path / "a.jsonl", _grouped_rows({(0, 0)}))),
                            target=0.8, group_rule="all")
    b = report_from_results(results_from_jsonl(_write(tmp_path / "b.jsonl", _grouped_rows(set()))),
                            target=0.8, group_rule="all")
    c = compare(a.to_dict(), b.to_dict())
    assert c.unit == "groups" and c.n_shared == 3 and c.only_b == 1


def test_a_a_repeated_id_inside_one_group_is_still_refused(tmp_path):
    rows = _grouped_rows(set()) + [{"id": "check-0", "group": "g0", "passed": True}]
    payload = report_from_results(results_from_jsonl(_write(tmp_path / "a.jsonl", rows)),
                                  target=0.8, group_rule="none").to_dict()
    with pytest.raises(ValueError, match="duplicate case id 'check-0' within group 'g0'"):
        compare(payload, payload)


def test_a_through_the_shell(tmp_path, capsys):
    for name, fails in (("a", {(0, 0)}), ("b", set())):
        main(["decide", str(_write(tmp_path / f"{name}.jsonl", _grouped_rows(fails))),
              "--target", "0.8", "--json"])
        (tmp_path / f"{name}.json").write_text(capsys.readouterr().out, encoding="utf-8")
    assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["unit"] == "cases" and out["n_shared"] == 9 and out["only_b"] == 1


# ---------------------------------------------------------------- B

def test_b_a_point_exactly_at_the_target_says_so():
    d = decide([(1.0, True)] * 19 + [(1.0, False)], target=0.95)
    assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needed_n is None
    assert "is at the 95% target" in d.reason
    assert "cannot clear a target the point only equals" in d.reason
    assert "is above" not in d.reason


def test_b_a_point_above_the_target_is_unchanged():
    d = decide([(1.0, True)] * 49 + [(1.0, False)], target=0.95)
    assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE and d.needed_n
    assert d.reason.startswith("observed 98.0% on 50 cases is above the 95% target, but")


# ---------------------------------------------------------------- C

@pytest.mark.parametrize("x,want", [
    (0.625, "0.625"), (0.231, "0.231"), (0.8, "0.80"), (0.5, "0.50"), (0.15, "0.15"),
    (0.150375, "0.1504"), (0.30000000000000004, "0.30"), (0.0, "0.00"), (1.0, "1.00"),
])
def test_c_fmt_cut(x, want):
    assert fmt_cut(x) == want


@pytest.mark.parametrize("x,want", [
    (0.625, "62.5%"), (0.29, "29%"), (2 / 3, "66.7%"), (1.0, "100%"), (0.25, "25%"), (0.99999, "100%"),
])
def test_c_fmt_share(x, want):
    assert fmt_share(x) == want


def test_c_the_threshold_in_a_reason_is_the_real_cut_point():
    # 20 confident passes at 0.625, 10 failures at 0.3: the cut is exactly 0.625, on 2/3 of cases
    d = decide([(0.625, True)] * 20 + [(0.3, False)] * 10, target=0.8)
    assert d.verdict is Verdict.AUTOMATE_WITH_REVIEW
    assert d.operating_point.threshold == 0.625
    assert "confidence 0.625 reaches" in d.reason and "on 66.7% of cases" in d.reason
    assert any(n.startswith("The 0.625 threshold") for n in d.notes)
    md = report_from_results(results_from_jsonl_rows([(0.625, True)] * 20 + [(0.3, False)] * 10),
                             target=0.8).markdown()
    assert "**0.625**" in md and "(66.7%)" in md


def results_from_jsonl_rows(pairs, tmp=None):
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    return results_from_jsonl(_write(d / "r.jsonl", [{"id": f"c{i}", "passed": p, "confidence": c}
                                                      for i, (c, p) in enumerate(pairs)]))


# ---------------------------------------------------------------- D

def _abstaining_rows():
    # no confidence anywhere: 14 answers (12 right), 6 abstentions on cases that wanted an answer
    rows = [{"id": f"a{i}", "passed": i < 12} for i in range(14)]
    rows += [{"id": f"x{i}", "passed": False, "action": "abstain"} for i in range(6)]
    return rows


def test_d_abstentions_without_confidence_get_no_calibration_error(tmp_path, capsys):
    r = report_from_results(results_from_jsonl(_write(tmp_path / "r.jsonl", _abstaining_rows())), target=0.8)
    d = r.decision
    assert math.isnan(d.calibration_error)
    assert main(["decide", str(tmp_path / "r.jsonl"), "--target", "0.8", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["calibration_error"] is None
    assert not any(n.startswith("Calibration error") for n in d.notes)
    assert any(n.startswith("No answered case reported its own confidence") for n in d.notes)
    assert not any(n.startswith("Every case reported the same confidence") for n in d.notes)
    assert "Calibration error" not in r.markdown()


def test_d_the_abstain_versus_answer_threshold_survives(tmp_path):
    # 20 answers all right, 6 abstentions: "automate what it didn't abstain on" is the operating point
    rows = [{"id": f"a{i}", "passed": True} for i in range(20)]
    rows += [{"id": f"x{i}", "passed": False, "action": "abstain"} for i in range(6)]
    d = report_from_results(results_from_jsonl(_write(tmp_path / "r.jsonl", rows)), target=0.8).decision
    assert d.verdict is Verdict.AUTOMATE_WITH_REVIEW
    assert d.operating_point.threshold == 1.0 and d.operating_point.n_covered == 20
    assert math.isnan(d.calibration_error)


def test_d_supplied_confidence_still_measures_calibration(tmp_path):
    rows = [{"id": f"a{i}", "passed": i < 12, "confidence": 0.9} for i in range(14)]
    rows += [{"id": f"x{i}", "passed": False, "action": "abstain"} for i in range(6)]
    d = report_from_results(results_from_jsonl(_write(tmp_path / "r.jsonl", rows)), target=0.8).decision
    assert not math.isnan(d.calibration_error)
    assert not any(n.startswith("No answered case reported its own confidence") for n in d.notes)


def test_d_direct_callers_keep_the_old_reading():
    pairs = [(1.0, True)] * 12 + [(1.0, False)] * 2 + [(0.0, False)] * 6
    assert not math.isnan(decide(pairs, target=0.8).calibration_error)
    assert math.isnan(decide(pairs, target=0.8, calibrated=False).calibration_error)


def test_d_confidence_only_on_errored_rows_is_not_the_agents(tmp_path):
    # train 2026-11-G: found by replaying the port's fixture table. Errored rows may carry a
    # harness-given confidence; with none on any answer, calibration is not measured.
    rows = [{"id": f"c{i}", "passed": i != 2} for i in range(8)]
    rows += [{"id": "e1", "passed": False, "error": "boom", "confidence": 0.2},
             {"id": "e2", "passed": False, "action": "error", "confidence": 0.6}]
    d = report_from_results(results_from_jsonl(_write(tmp_path / "r.jsonl", rows))).decision
    assert math.isnan(d.calibration_error)
    assert any(n.startswith("No answered case reported its own confidence") for n in d.notes)
