import json

import pytest

from gonogo import Case, Prediction, Verdict, evaluate
from gonogo.scoring import (
    exact, fields, judge, judge_agreement, numeric, set_f1, validate_judge,
)


def cases(n=10, expected="yes"):
    return [Case(input=f"q{i}", expected=expected, id=f"c{i}") for i in range(n)]


class TestRunner:
    def test_perfect_agent_ships(self):
        report = evaluate(lambda c: "yes", cases(200), target=0.95)
        assert report.decision.verdict is Verdict.AUTOMATE
        assert report.n_passed == 200

    def test_agent_may_return_bare_value_tuple_or_prediction(self):
        for agent in (lambda c: "yes",
                      lambda c: ("yes", 0.9),
                      lambda c: Prediction("yes", confidence=0.9)):
            report = evaluate(agent, cases(20))
            assert report.n_passed == 20

    def test_agent_exception_fails_only_that_case(self):
        def flaky(case):
            if case.id == "c3":
                raise RuntimeError("boom")
            return "yes"

        report = evaluate(flaky, cases(10))
        assert report.n_passed == 9
        assert len(report.errors) == 1
        assert "boom" in report.errors[0].detail

    def test_scorer_exception_is_reported_not_raised(self):
        def broken(output, expected):
            raise ValueError("bad scorer")

        report = evaluate(lambda c: "yes", cases(5), scorer=broken)
        assert report.n_passed == 0
        assert "scorer error" in report.results[0].detail

    def test_abstention_defers_first_and_does_not_count_as_correct(self):
        # 20 abstentions out of 100: the overall rate (80%) misses a 90% target,
        # but the 80 answered cases clear it, which is the whole point of abstaining.
        def cautious(case):
            if int(case.id[1:]) < 20:
                return Prediction(None, abstained=True)
            return Prediction("yes", confidence=0.99)

        report = evaluate(cautious, cases(100), target=0.90)
        assert report.n_passed == 80
        assert report.decision.operating_point is not None
        # Abstentions land at zero confidence, so they defer first.
        assert report.decision.operating_point.n_deferred == 20

    def test_parallel_matches_serial(self):
        agent = lambda c: ("yes", 0.9)
        a = evaluate(agent, cases(40), workers=1)
        b = evaluate(agent, cases(40), workers=4)
        assert a.n_passed == b.n_passed == 40

    def test_empty_case_list_raises(self):
        with pytest.raises(ValueError):
            evaluate(lambda c: "yes", [])


class TestScorers:
    def test_exact_normalizes_whitespace_and_case(self):
        assert exact()("  Yes ", "yes")[0]
        assert not exact()("no", "yes")[0]

    def test_numeric_tolerance(self):
        assert numeric(tolerance=0.01)(100.5, 100)[0]
        assert not numeric(tolerance=0.001)(100.5, 100)[0]
        assert not numeric()("abc", 100)[0]

    def test_fields_requires_every_field(self):
        scorer = fields()
        ok, score, _ = scorer({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert ok and score == 1.0
        ok, score, detail = scorer({"a": 1, "b": 9}, {"a": 1, "b": 2})
        assert not ok and score == 0.5 and "b" in detail

    def test_set_f1(self):
        assert set_f1()(["a", "b"], ["b", "a"])[0]
        ok, score, _ = set_f1(threshold=0.9)(["a"], ["a", "b"])
        assert not ok and 0 < score < 1

    def test_judge_parses_score_and_is_provider_free(self):
        scorer = judge(rubric="Is it right?", complete=lambda p: "5 - looks correct")
        assert scorer("x", "x")[0]
        scorer_low = judge(rubric="Is it right?", complete=lambda p: "2 - wrong")
        assert not scorer_low("x", "x")[0]

    def test_judge_handles_unparseable_reply(self):
        scorer = judge(rubric="r", complete=lambda p: "I cannot say")
        ok, score, detail = scorer("x", "x")
        assert not ok and score == 0.0 and "no score" in detail

    def test_judge_agreement_and_kappa(self):
        perfect = judge_agreement([True, False, True], [True, False, True])
        assert perfect["agreement"] == 1.0 and perfect["kappa"] == 1.0
        # Judge says pass on everything while humans disagree: kappa collapses.
        useless = judge_agreement([True] * 10, [True] * 5 + [False] * 5)
        assert useless["agreement"] == 0.5 and useless["kappa"] == pytest.approx(0.0)

    def test_judge_agreement_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            judge_agreement([True], [True, False])


class TestJudgeValidation:
    def test_high_agreement_with_zero_kappa_is_rejected(self):
        # The classic trap: a judge that passes everything looks 90% accurate
        # on a set that is 90% passes, while carrying no information at all.
        v = validate_judge([True] * 30, [True] * 27 + [False] * 3)
        assert v.agreement == pytest.approx(0.9)
        assert v.kappa == pytest.approx(0.0)
        assert not v.usable
        assert "below" in v.reason

    def test_a_judge_that_tracks_the_human_is_usable(self):
        judge_labels = [True] * 20 + [False] * 10
        human_labels = [True] * 18 + [False] * 2 + [False] * 8 + [True] * 2
        v = validate_judge(judge_labels, human_labels)
        assert v.usable and v.kappa > 0.6

    def test_too_few_labels_is_rejected_even_when_perfect(self):
        v = validate_judge([True] * 5, [True] * 5)
        assert not v.usable
        assert "at least 20" in v.reason

    def test_direction_of_disagreement_is_reported(self):
        # Judge passes 10 cases the human failed: it inflates the score.
        v = validate_judge([True] * 25, [True] * 15 + [False] * 10)
        assert v.judge_lenient == 10 and v.judge_strict == 0
        assert "passes cases you failed" in v.reason

    def test_empty_is_rejected(self):
        assert not validate_judge([], []).usable


class TestCaseLoading:
    def test_round_trip_jsonl(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        original = [Case(input="a", expected="b", id="x1", metadata={"kind": "invoice"})]
        Case.to_jsonl(original, path)
        loaded = Case.from_jsonl(path)
        assert loaded[0].input == "a" and loaded[0].expected == "b"
        assert loaded[0].id == "x1" and loaded[0].metadata["kind"] == "invoice"

    def test_missing_key_names_the_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"input": "a"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required key"):
            Case.from_jsonl(path)

    def test_invalid_json_names_the_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"input": "a", "expected": }\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            Case.from_jsonl(path)

    def test_blank_lines_and_comments_are_skipped(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text('// note\n\n{"input":"a","expected":"b"}\n', encoding="utf-8")
        assert len(Case.from_jsonl(path)) == 1

    def test_confidence_must_be_a_probability(self):
        with pytest.raises(ValueError):
            Prediction("x", confidence=1.4)


class TestReport:
    def test_markdown_leads_with_the_verdict(self):
        report = evaluate(lambda c: ("yes", 0.9), cases(200), task="Extract fields")
        md = report.markdown()
        assert md.startswith("# Score report: Extract fields")
        assert "**AUTOMATE**" in md
        assert "confidence interval" in md

    def test_markdown_reports_insufficient_evidence(self):
        def agent(case):
            return ("yes", 0.9) if case.id != "c0" else ("no", 0.9)

        report = evaluate(agent, cases(20), target=0.90)
        md = report.markdown()
        assert "INSUFFICIENT EVIDENCE" in md

    def test_failures_are_listed_worst_first(self):
        def agent(case):
            return "no" if case.id in ("c1", "c2") else "yes"

        report = evaluate(agent, cases(10))
        failures = report.failures()
        assert {f.case.id for f in failures} == {"c1", "c2"}
        assert "c1" in report.markdown()

    def test_to_dict_is_json_serializable(self):
        report = evaluate(lambda c: ("yes", 0.9), cases(50))
        json.dumps(report.to_dict())  # must not raise

    def test_summary_is_one_line(self):
        report = evaluate(lambda c: "yes", cases(100), task="Route ticket")
        assert "\n" not in report.summary()
        assert "Route ticket" in report.summary()


class TestHtmlReport:
    def test_standalone_is_a_full_document_with_inlined_css(self):
        report = evaluate(lambda c: ("yes", 0.9), cases(200), task="Route ticket")
        html = report.html()
        assert html.startswith("<!doctype html>")
        assert "<style>" in html and "http://" not in html and "https://" not in html
        assert "Route ticket" in html

    def test_fragment_omits_the_document_shell(self):
        report = evaluate(lambda c: ("yes", 0.9), cases(200))
        frag = report.html(standalone=False)
        assert not frag.startswith("<!doctype")
        assert frag.strip().startswith('<section class="gng"')
        assert frag.strip().endswith("</section>")

    def test_verdict_is_exposed_for_styling(self):
        report = evaluate(lambda c: ("yes", 0.9), cases(200), target=0.95)
        assert 'data-verdict="automate"' in report.html()
        bad = evaluate(lambda c: ("no", 0.9), cases(200), target=0.95)
        assert 'data-verdict="do-not-automate"' in bad.html()

    def test_escapes_untrusted_text(self):
        nasty = [Case(input="x", expected="<script>alert(1)</script>", id="<b>id</b>")]
        report = evaluate(lambda c: "wrong", nasty, task="<img src=x onerror=1>")
        html = report.html()
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
        assert "<img src=x onerror=1>" not in html

    def test_rows_are_flagged_against_the_target(self):
        report = evaluate(lambda c: ("yes", 0.9), cases(200), target=0.95)
        html = report.html()
        assert "gng-ok" in html

    def test_show_failures_zero_omits_the_section(self):
        report = evaluate(lambda c: "no", cases(50), target=0.95)
        assert report.failures(0) == []
        assert len(report.failures()) == 50
        assert "Failures worth reading" not in report.markdown(show_failures=0)
        assert "Failures worth reading" not in report.html(show_failures=0)
