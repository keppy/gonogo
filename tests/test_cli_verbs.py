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


def _crashing(on):
    """The confident agent above, raising on the case numbers in `on`."""
    def agent(case):
        if int(case.id.split("-")[1]) in on:
            raise RuntimeError("upstream timed out")
        return _agent(case)
    return agent


def _cases_back(report, tmp_path, name="back.jsonl"):
    """What `decide --json` wrote for each case, as the rows a harness would load."""
    return results_from_jsonl(_write(tmp_path / name, report.to_dict()["cases"]))


def _same_report(a, b):
    da, db = a.to_dict(), b.to_dict()
    da.pop("metadata"), db.pop("metadata")
    assert da == db


class TestErroredCases:
    def test_an_error_row_is_the_case_evaluate_records_for_a_crash(self, tmp_path):
        ref = evaluate(_crashing({3}), CASES[:5])
        rows = [{"id": "c-1", "passed": True, "confidence": 0.975},
                {"id": "c-2", "passed": True, "confidence": 0.95},
                {"id": "c-3", "passed": False, "error": "RuntimeError: upstream timed out"},
                {"id": "c-4", "passed": True, "confidence": 0.9},
                {"id": "c-5", "passed": True, "confidence": 0.875}]
        got = report_from_results(results_from_jsonl(_write(tmp_path / "e.jsonl", rows)))
        _same_report(got, ref)
        crash = got.errors[0]
        assert (crash.passed, crash.score, crash.detail) == (False, 0.0, "agent error: RuntimeError: upstream timed out")
        assert crash.confidence == 1.0             # supplied by nobody, so it sits at the top
        assert "error" not in crash.case.metadata  # a result field, not metadata
        assert got.actions()["held_back_when_should_act"] == 0  # an error neither acted nor held back

    def test_an_errored_row_stays_out_of_the_mixed_confidence_check(self, tmp_path):
        rows = [{"id": "a", "passed": True, "confidence": 0.9},
                {"id": "b", "passed": False, "error": "boom"},
                {"id": "c", "passed": False, "error": "boom", "confidence": 0.2},
                {"id": "d", "passed": False, "action": "error"}]
        by_id = {r.case.id: r for r in report_from_results(results_from_jsonl(_write(tmp_path / "c.jsonl", rows))).results}
        assert (by_id["b"].confidence, by_id["c"].confidence, by_id["d"].confidence) == (1.0, 0.2, 1.0)
        # The check still holds for the agent's own rows.
        rows.append({"id": "e", "passed": True})
        with pytest.raises(ValueError, match="mixed missing and present confidence"):
            report_from_results(results_from_jsonl(_write(tmp_path / "c.jsonl", rows)))

    def test_action_error_and_a_null_error(self, tmp_path):
        rows = [{"id": "a", "passed": False, "action": "error"},
                {"id": "b", "passed": False, "action": "error", "error": "boom", "detail": "kept"},
                {"id": "c", "passed": True, "error": None}]
        a, b, c = results_from_jsonl(_write(tmp_path / "x.jsonl", rows))
        assert (a.prediction.error, a.detail, a.prediction.action) == ("error", "agent error: error", "answer")
        assert (b.prediction.error, b.detail) == ("boom", "kept")
        assert c.prediction.error is None and c.passed

    @pytest.mark.parametrize("row,fragment", [
        ({"passed": True, "error": "boom"}, "an errored case cannot pass"),
        ({"passed": True, "action": "error"}, "an errored case cannot pass"),
        ({"passed": False, "error": "boom", "action": "wait"}, "error is set, but action is 'wait'"),
        ({"passed": False, "error": "boom", "action": "answer"}, "error is set, but action is 'answer'"),
        ({"passed": False, "error": ""}, "error must be a non-empty string"),
        ({"passed": False, "error": 3}, "error must be a non-empty string"),
    ])
    def test_a_contradictory_error_row_is_refused_with_its_line(self, tmp_path, row, fragment):
        path = _write(tmp_path / "bad.jsonl", [{"id": "ok", "passed": True}, {"id": "ok2", "passed": True}, row])
        with pytest.raises(ValueError, match=f"bad.jsonl:3: {fragment}"):
            results_from_jsonl(path)
        with pytest.raises(SystemExit) as exc:
            main(["decide", path])
        assert exc.value.code == 2

    @pytest.mark.parametrize("crashes", [set(), {4, 9, 27}, set(range(1, 31))])
    def test_decide_json_round_trips(self, tmp_path, capsys, crashes):
        ref = evaluate(_crashing(crashes), CASES, target=0.85)
        back = report_from_results(_cases_back(ref, tmp_path), target=0.85)
        _same_report(back, ref)
        assert [r.prediction.error for r in back.results] == [r.prediction.error for r in ref.results]
        # And through the shell: decide --json, its cases fed back to decide --json.
        first = _write(tmp_path / "first.jsonl", ref.to_dict()["cases"])
        assert main(["decide", first, "--target", "0.85", "--json"]) == 0
        once = json.loads(capsys.readouterr().out)
        second = _write(tmp_path / "second.jsonl", once["cases"])
        assert main(["decide", second, "--target", "0.85", "--json"]) == 0
        twice = json.loads(capsys.readouterr().out)
        once.pop("metadata"), twice.pop("metadata")
        assert once == twice
        assert once["verdict"] == ref.decision.verdict.value

    def test_round_trip_with_actions_keeps_the_error(self, tmp_path):
        cases = CASES[:10] + [Case(input=0, expected=None, id="c-99", expected_action="wait")]
        def agent(case):
            if case.id == "c-99":
                raise RuntimeError("tool unavailable")
            return _agent(case)
        ref = evaluate(agent, cases)
        payload = ref.to_dict()["cases"][-1]
        assert (payload["action"], payload["error"]) == ("error", "RuntimeError: tool unavailable")
        back = report_from_results(_cases_back(ref, tmp_path))
        _same_report(back, ref)
        assert back.actions() == ref.actions()

    def test_every_case_errored_is_insufficient_evidence(self, tmp_path, capsys):
        ref = evaluate(_crashing(set(range(1, 31))), CASES, target=0.85)
        rows = [{"id": f"c-{i}", "passed": False, "error": "harness down"} for i in range(1, 31)]
        loaded = report_from_results(results_from_jsonl(_write(tmp_path / "all.jsonl", rows)), target=0.85)
        for d in (ref.decision, loaded.decision):
            assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE
            assert d.needed_n is None
            assert d.reason.startswith("every case errored (30 of 30), so the run measured nothing")
            assert d.operating_point is None and d.notes == []
        assert main(["decide", str(tmp_path / "all.jsonl"), "--target", "0.85"]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == "INSUFFICIENT EVIDENCE" and len(out) == 2

    def test_every_case_errored_in_a_grouped_run(self, tmp_path):
        rows = [{"id": f"k{j}", "group": f"run-{i}", "passed": False, "error": "boom"}
                for i in range(4) for j in range(3)]
        d = report_from_results(results_from_jsonl(_write(tmp_path / "g.jsonl", rows)), group_rule="all").decision
        assert (d.verdict, d.n_groups, d.unit) == (Verdict.INSUFFICIENT_EVIDENCE, 4, "groups")

    def test_some_cases_errored_adds_one_note(self, tmp_path):
        ref = evaluate(_crashing({4, 9, 27}), CASES, target=0.85)
        loaded = report_from_results(_cases_back(ref, tmp_path), target=0.85)
        for d in (ref.decision, loaded.decision):
            assert [n for n in d.notes if "errored" in n] == [
                "3 of 30 cases errored and count as failures; "
                "read their detail before reading the pass rate as the agent's."]
        assert not any("errored" in n for n in evaluate(_agent, CASES).decision.notes)
