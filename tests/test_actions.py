"""Expected actions: sometimes the right behaviour is to not answer.

An agent's action space includes staying silent, asking a person, and
waiting. A case can say which of those is correct, and the harness grades
the action instead of scoring an output.
"""

import json

import pytest

from gonogo import Case, Prediction, evaluate
from gonogo.cases import CaseResult

# An inbox: some messages need a drafted reply, some need no reply at all,
# and some need a person.
INBOX = [
    Case(input="Where is my order #1001?", expected="reply", id="m1"),
    Case(input="Can I change my delivery address?", expected="reply", id="m2"),
    Case(input="Do you ship to Canada?", expected="reply", id="m3"),
    Case(input="Weekly newsletter", expected=None, id="m4", expected_action="abstain"),
    Case(input="Out-of-office auto-reply", expected=None, id="m5", expected_action="abstain"),
    Case(input="I am disputing this charge with my bank", expected=None, id="m6",
         expected_action="ask"),
    Case(input="Will confirm the meeting time tomorrow", expected=None, id="m7",
         expected_action="wait"),
]
ANSWER_IDS = {"m1", "m2", "m3"}
HOLD_IDS = {"m4", "m5", "m6", "m7"}


def oracle(case):
    if case.expected_action == "answer":
        return case.expected
    return Prediction(None, action=case.expected_action)


def always_answer(case):
    return "reply"


def always_abstain(case):
    return Prediction(None, abstained=True)


def by_id(report):
    return {r.case.id: r for r in report.results}


class TestScoring:
    def test_oracle_passes_everything(self):
        report = evaluate(oracle, INBOX)
        assert report.n_passed == report.n == 7

    def test_always_answer_fails_every_non_answer_case(self):
        results = by_id(evaluate(always_answer, INBOX))
        assert {i for i, r in results.items() if not r.passed} == HOLD_IDS
        assert "answered" in results["m6"].detail and "'ask'" in results["m6"].detail

    def test_always_abstain_fails_every_answer_case(self):
        results = by_id(evaluate(always_abstain, INBOX))
        assert {i for i, r in results.items() if r.passed} == {"m4", "m5"}
        for i in ANSWER_IDS:
            assert results[i].detail == "agent abstained"
            assert results[i].confidence == 0.0
        # Abstaining is not asking or waiting.
        assert not results["m6"].passed and not results["m7"].passed
        assert "'abstain'" in results["m6"].detail and "'ask'" in results["m6"].detail

    def test_scorer_not_called_on_non_answer_cases(self):
        seen = []

        def scorer(output, expected, case):
            seen.append(case.id)
            return output == expected

        evaluate(oracle, INBOX, scorer=scorer)
        assert set(seen) == ANSWER_IDS

    def test_non_answer_action_on_answer_case_names_the_action(self):
        r = evaluate(lambda c: Prediction(None, action="ask"), [Case(input=1, expected=1)]).results[0]
        assert not r.passed and r.confidence == 0.0
        assert r.detail == "agent took action 'ask' instead of answering"
        assert r.prediction.action == "ask" and r.prediction.abstained

    def test_correct_non_answer_keeps_agent_confidence(self):
        def agent(case):
            if case.expected_action == "answer":
                return case.expected, 0.9
            return Prediction(None, confidence=0.7, action=case.expected_action)

        report = evaluate(agent, INBOX)
        results = by_id(report)
        assert results["m6"].confidence == 0.7 and results["m6"].passed
        assert results["m1"].confidence == 0.9

    def test_missing_confidence_on_a_graded_action_is_mixed(self):
        def agent(case):
            if case.expected_action == "answer":
                return case.expected, 0.9
            return Prediction(None, action=case.expected_action)  # no confidence

        with pytest.raises(ValueError, match="mixed missing and present confidence"):
            evaluate(agent, INBOX)


class TestLegacy:
    def test_abstained_true_still_fails_an_answer_case_as_before(self):
        cases = [Case(input=i, expected=i) for i in range(2)]
        report = evaluate(lambda c: Prediction(None, abstained=True) if c.input else c.expected, cases)
        r = report.results[1]
        assert (r.passed, r.score, r.detail, r.confidence) == (False, 0.0, "agent abstained", 0.0)
        assert r.prediction.abstained and r.prediction.action == "abstain"
        assert not report.uses_actions
        assert "## Actions" not in report.markdown()
        assert "actions" not in report.to_dict()
        assert "expected_action" not in report.to_dict()["cases"][0]

    def test_prediction_normalization(self):
        assert Prediction(1).action == "answer" and not Prediction(1).abstained
        assert Prediction(1, abstained=True).action == "abstain"
        assert Prediction(1, action="abstain").abstained
        assert Prediction(1, action="wait").abstained
        assert Prediction(1, abstained=True, action="ask").action == "ask"
        with pytest.raises(ValueError):
            Prediction(1, action="")

    def test_case_rejects_empty_expected_action(self):
        with pytest.raises(ValueError, match="expected_action"):
            Case(input=1, expected=1, expected_action="")


class TestJsonl:
    def test_round_trip_and_omission_when_answer(self, tmp_path):
        p = tmp_path / "inbox.jsonl"
        Case.to_jsonl(INBOX, p)
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        assert "expected_action" not in rows[0]
        assert rows[5]["expected_action"] == "ask"
        back = Case.from_jsonl(p)
        assert [c.expected_action for c in back] == [c.expected_action for c in INBOX]
        assert all("expected_action" not in c.metadata for c in back)

    def test_legacy_file_is_byte_identical(self, tmp_path):
        src = tmp_path / "legacy.jsonl"
        src.write_text(json.dumps({"id": "a", "input": 1, "expected": 1, "site": "x"}) + "\n",
                       encoding="utf-8")
        out = tmp_path / "out.jsonl"
        Case.to_jsonl(Case.from_jsonl(src), out)
        assert out.read_bytes() == src.read_bytes()

    def test_non_string_expected_action_is_rejected_with_line(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps({"input": 1, "expected": 1, "expected_action": 3}) + "\n",
                     encoding="utf-8")
        with pytest.raises(ValueError, match="bad.jsonl:1"):
            Case.from_jsonl(p)


class TestReport:
    def test_confusion_counts(self):
        def agent(case):
            if case.id == "m1":
                return Prediction(None, action="ask")      # held back, should have acted
            if case.id == "m4":
                return "reply"                             # acted, should have held back
            if case.id == "m6":
                return Prediction(None, action="wait")     # wrong kind of holding back
            if case.id == "m7":
                raise RuntimeError("timeout")
            return oracle(case)

        report = evaluate(agent, INBOX)
        acts = report.actions()
        matrix = {(r["expected"], r["taken"]): r["n"] for r in acts["matrix"]}
        assert matrix == {
            ("answer", "answer"): 2, ("answer", "ask"): 1,
            ("abstain", "abstain"): 1, ("abstain", "answer"): 1,
            ("ask", "wait"): 1, ("wait", "error"): 1,
        }
        assert acts["acted_when_should_hold_back"] == 1
        assert acts["held_back_when_should_act"] == 1
        assert sum(matrix.values()) == report.n

        md = report.markdown()
        assert "## Actions" in md and "| ask | wait | 1 |" in md
        assert "Acted when it should have held back: 1" in md
        assert "Held back when it should have acted: 1" in md
        assert "<caption>Actions</caption>" in report.html()
        d = report.to_dict()
        assert d["actions"] == json.loads(json.dumps(acts))
        assert {c["id"]: c["action"] for c in d["cases"]}["m7"] == "error"

    def test_report_mixed_confidence_check_counts_graded_actions(self):
        # Rebuilding a Report from results applies the same rule as evaluate().
        from gonogo import Report
        report = evaluate(oracle, INBOX)
        results = [CaseResult(r.case, Prediction(r.prediction.output, confidence=0.5,
                                                 action=r.prediction.action), r.passed, r.score)
                   if r.case.expected_action == "answer" else r for r in report.results]
        with pytest.raises(ValueError, match="mixed"):
            Report(task="t", results=results, decision=report.decision)


class TestGrouped:
    def test_grouped_run(self):
        cases = []
        for run in range(3):
            cases.append(Case(input="q", expected="a", id="answer-it", group=f"thread-{run}"))
            cases.append(Case(input="fyi", expected=None, id="leave-it", group=f"thread-{run}",
                              expected_action="abstain"))

        def agent(case):
            # Answers the question every time; replies to the FYI in thread-2.
            if case.expected_action == "abstain" and case.group != "thread-2":
                return Prediction(None, action="abstain")
            return "a"

        report = evaluate(agent, cases, group_rule="all")
        assert report.n_groups == 3 and report.n_groups_passed() == 2
        assert report.decision.pass_rate.n == 3
        assert report.actions()["acted_when_should_hold_back"] == 1
        assert ("leave-it", 2, 3) in report.recurring()
