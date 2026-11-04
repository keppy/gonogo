"""`gonogo decide RESULTS.jsonl` and `gonogo compare A.json B.json`: the shell surface for a
run scored somewhere else (another language, or a harness that scores as it goes)."""

import json

import pytest

from gonogo import Case, Verdict, compare, evaluate, report_from_results, results_from_jsonl
from gonogo.cli import main


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _agent(case):
    # Wrong on the last eight cases, which are also the least confident, so a
    # review threshold exists and the operating point is worth comparing.
    i = int(case.id.split("-")[1])
    return ("ok" if i <= 22 else "no"), 1.0 - i / 40


CASES = [Case(input=i, expected="ok", id=f"c-{i}") for i in range(1, 31)]


class TestDecide:
    def test_a_file_of_outcomes_gets_the_decision_evaluate_gives(self, tmp_path):
        ref = evaluate(_agent, CASES, target=0.85)
        rows = [{"id": r.case.id, "passed": r.passed, "confidence": r.confidence} for r in ref.results]
        assert main(["decide", _write(tmp_path / "r.jsonl", rows), "--target", "0.85", "--json"]) == 0
        got = report_from_results(results_from_jsonl(tmp_path / "r.jsonl"), target=0.85)
        assert ref.decision.verdict is Verdict.AUTOMATE_WITH_REVIEW
        assert got.decision.verdict is ref.decision.verdict
        assert got.decision.pass_rate == ref.decision.pass_rate
        assert got.decision.operating_point.threshold == ref.decision.operating_point.threshold
        assert got.decision.calibration_error == pytest.approx(ref.decision.calibration_error)

    def test_actions_are_graded_like_evaluate(self, tmp_path):
        rows = [
            {"id": "m1", "passed": True, "confidence": 0.9},
            {"id": "m2", "passed": True, "confidence": 0.8, "expected_action": "no_reply", "action": "no_reply"},
            {"id": "m3", "passed": False, "confidence": 0.7, "expected_action": "ask_human", "action": "answer"},
            {"id": "m4", "passed": False, "action": "wait", "confidence": 0.6},
            {"id": "m5", "passed": False, "action": "abstain"},
        ]
        results = results_from_jsonl(_write(tmp_path / "a.jsonl", rows))
        by_id = {r.case.id: r for r in results}
        assert by_id["m4"].confidence == 0.6        # a confident wait keeps its confidence
        assert by_id["m5"].confidence == 0.0        # a deferral sits at the bottom
        from gonogo import report_from_results
        acts = report_from_results(results).actions()
        assert acts["acted_when_should_hold_back"] == 1   # m3
        assert acts["held_back_when_should_act"] == 2     # m4 and m5

    @pytest.mark.parametrize("row,fragment", [
        ({"passed": True, "action": "wait"}, "cannot pass"),
        ({"passed": True, "expected_action": "no_reply", "action": "answer"}, "means false"),
        ({"passed": "yes"}, "passed must be true or false"),
        ({"passed": True, "confidence": 1.5}, "confidence"),
    ])
    def test_a_row_that_contradicts_the_rules_is_refused_with_its_line(self, tmp_path, row, fragment):
        path = _write(tmp_path / "bad.jsonl", [{"id": "ok", "passed": True}, row])
        with pytest.raises(ValueError, match=f"bad.jsonl:2: .*{fragment}"):
            results_from_jsonl(path)
        with pytest.raises(SystemExit) as exc:
            main(["decide", path])
        assert exc.value.code == 2

    def test_text_output_is_the_verdict_first(self, tmp_path, capsys):
        rows = [{"id": f"c{i}", "passed": i != 3} for i in range(50)]
        assert main(["decide", _write(tmp_path / "r.jsonl", rows), "--target", "0.9"]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == "INSUFFICIENT EVIDENCE"
        assert out[2].startswith("49/50 looks like 98%; you'd need about")

    def test_grouped_rows_decide_over_groups(self, tmp_path, capsys):
        rows = [{"id": f"check-{j}", "group": f"run-{i}", "passed": not (i == 2 and j == 1)}
                for i in range(10) for j in range(3)]
        assert main(["decide", _write(tmp_path / "g.jsonl", rows), "--group-rule", "all", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert (data["n"], data["n_groups"], data["n_groups_passed"]) == (30, 10, 9)
        assert data["cases"][0]["group"] == "run-0"


class TestCompare:
    def test_two_decide_files_pair_on_case_id(self, tmp_path, capsys):
        a_rows = [{"id": f"c{i}", "passed": i % 4 != 0} for i in range(40)]
        b_rows = [{"id": f"c{i}", "passed": i % 8 != 0} for i in range(40)]
        for name, rows in (("a", a_rows), ("b", b_rows)):
            assert main(["decide", _write(tmp_path / f"{name}.jsonl", rows), "--json"]) == 0
            (tmp_path / f"{name}.json").write_text(capsys.readouterr().out)
        assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        ref = compare(json.loads((tmp_path / "a.json").read_text()),
                      json.loads((tmp_path / "b.json").read_text()))
        assert data == ref.to_dict()
        assert (data["n_shared"], data["only_a"], data["only_b"]) == (40, 0, 5)
        assert data["verdict"] in ("distinguishable", "borderline", "not distinguishable")
        assert data["better"] == ("B" if data["significant"] else None)
        # The text form carries the same verdict line the Python summary prints.
        assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
        assert capsys.readouterr().out.strip() == ref.summary()

    def test_a_file_that_is_not_a_report_exits_2(self, tmp_path):
        (tmp_path / "x.json").write_text("{}")
        with pytest.raises(SystemExit) as exc:
            main(["compare", str(tmp_path / "x.json"), str(tmp_path / "x.json")])
        assert exc.value.code == 2
