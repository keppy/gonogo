"""Label provenance on validate_judge.

The failure this guards against is silent: labels from a second model, fed to
a function whose parameter was called human_passes, came back described as
human validation. Every assertion here is about the words the result uses.
"""

import pytest

from gonogo import validate_judge

# 30 cases, kappa well above 0.6: 26 agreements, 2 lenient, 2 strict.
JUDGE = [True] * 15 + [False] * 15
REF = [True] * 13 + [False] * 2 + [False] * 13 + [True] * 2


def words(v) -> str:
    return (str(v) + " " + v.reason).lower()


class TestDefaultIsHuman:
    def test_default_source_is_human_and_says_so(self):
        v = validate_judge(JUDGE, REF)
        assert v.usable
        assert v.label_source == "human"
        assert v.labels_are_human
        assert "hand-labelled" in str(v)

    def test_human_result_text_is_unchanged_from_before(self):
        v = validate_judge(JUDGE, REF)
        assert v.reason == "kappa 0.73 clears 0.60"


class TestModelLabels:
    def test_model_labels_never_described_as_human(self):
        v = validate_judge(JUDGE, REF, label_source="model",
                           label_source_note="nemotron-3-super-120b via OpenRouter")
        assert v.usable
        assert not v.labels_are_human
        assert "human validation" in v.reason  # "...not human validation"
        text = words(v).replace("not human validation", "")
        assert "human" not in text
        assert "hand-label" not in text
        assert "nemotron-3-super-120b" in str(v)

    def test_model_skew_wording_does_not_say_you(self):
        # kappa below the gate, judge lenient: the old wording said "cases you failed".
        judge = [True] * 20 + [False] * 5
        ref = [True] * 10 + [False] * 15
        v = validate_judge(judge, ref, label_source="model")
        assert not v.usable
        assert "the reference failed" in v.reason
        assert " you " not in f" {v.reason} "

    def test_human_skew_wording_still_says_you(self):
        judge = [True] * 20 + [False] * 5
        ref = [True] * 10 + [False] * 15
        v = validate_judge(judge, ref)
        assert "cases you failed" in v.reason


class TestStructuralLabels:
    def test_lenient_judge_is_rubber_stamping_whatever_the_kappa(self):
        judge = [True] * 20
        ref = [True] * 15 + [False] * 5
        v = validate_judge(judge, ref, label_source="structural")
        assert not v.usable
        assert v.judge_lenient == 5
        assert "rubber-stamping" in v.reason

    def test_stricter_judge_passes_the_structural_gate_without_claiming_validation(self):
        judge = [True] * 10 + [False] * 10
        ref = [True] * 15 + [False] * 5
        v = validate_judge(judge, ref, label_source="structural")
        assert v.usable
        assert v.judge_lenient == 0 and v.judge_strict == 5
        assert "not the gate" in v.reason
        assert "does not validate the judge" in v.reason
        assert "human" not in words(v)


class TestDeprecatedAlias:
    def test_human_passes_still_works_and_warns(self):
        with pytest.warns(DeprecationWarning):
            old = validate_judge(JUDGE, human_passes=REF)
        new = validate_judge(JUDGE, REF)
        assert (old.n, old.agreement, old.kappa, old.usable, old.reason) == \
               (new.n, new.agreement, new.kappa, new.usable, new.reason)
        assert old.label_source == "human"

    def test_both_names_is_an_error(self):
        with pytest.raises(TypeError):
            validate_judge(JUDGE, REF, human_passes=REF)

    def test_missing_reference_is_an_error(self):
        with pytest.raises(TypeError):
            validate_judge(JUDGE)


class TestEdges:
    def test_empty_source_is_rejected(self):
        with pytest.raises(ValueError):
            validate_judge(JUDGE, REF, label_source="")

    def test_small_n_wording_is_source_neutral(self):
        v = validate_judge([True] * 5, [True] * 4 + [False], label_source="model")
        assert not v.usable
        assert "hand-label" not in words(v)
        assert "human" not in words(v)

    def test_free_text_source_is_carried_through(self):
        v = validate_judge(JUDGE, REF, label_source="crowd")
        assert "labelled by crowd" in str(v)
        assert "not human validation" in v.reason
