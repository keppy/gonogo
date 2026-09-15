"""A scorer may take the Case as a third argument."""

import pytest

from gonogo import Case, evaluate
from gonogo.scoring import exact, numeric


def cases():
    return [
        Case(input="a", expected="4", id="num", metadata={"kind": "numeric"}),
        Case(input="b", expected="yes", id="txt", metadata={"kind": "text"}),
    ]


class TestScorerArity:
    def test_two_argument_scorers_are_unchanged(self):
        report = evaluate(lambda c: c.expected, cases(), scorer=exact())
        assert report.n_passed == 2

    def test_three_argument_scorer_receives_the_case(self):
        seen = []

        def scorer(output, expected, case):
            seen.append(case.id)
            return output == expected, 1.0, ""

        report = evaluate(lambda c: c.expected, cases(), scorer=scorer)
        assert report.n_passed == 2
        assert seen == ["num", "txt"]

    def test_routing_by_case_metadata(self):
        by_kind = {"numeric": numeric(tolerance=0.5, relative=False), "text": exact()}

        def route(output, expected, case):
            return by_kind[case.metadata["kind"]](output, expected)

        agent = lambda c: "4.3" if c.id == "num" else "yes"
        report = evaluate(agent, cases(), scorer=route)
        assert report.n_passed == 2  # 4.3 within 0.5 of 4; "yes" exact

    def test_three_argument_scorer_error_is_reported_not_raised(self):
        def bad(output, expected, case):
            raise KeyError(case.id)

        report = evaluate(lambda c: "x", cases(), scorer=bad)
        assert report.n_passed == 0
        assert all("scorer error" in r.detail for r in report.results)

    def test_scorer_with_defaulted_third_parameter_still_gets_the_case(self):
        got = {}

        def scorer(output, expected, case=None):
            got["case"] = case
            return True, 1.0, ""

        evaluate(lambda c: "x", cases()[:1], scorer=scorer)
        assert got["case"] is not None and got["case"].id == "num"

    def test_uninspectable_callable_falls_back_to_two_arguments(self):
        class Callable2:
            def __call__(self, output, expected):
                return True, 1.0, ""

        report = evaluate(lambda c: "x", cases(), scorer=Callable2())
        assert report.n_passed == 2
