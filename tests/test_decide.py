import pytest

from gonogo.decide import Verdict, decide


class TestVerdicts:
    def test_automate_when_lower_bound_clears_target(self):
        results = [(0.9, True)] * 200
        d = decide(results, target=0.95)
        assert d.verdict is Verdict.AUTOMATE
        assert d.can_automate

    def test_insufficient_evidence_when_point_estimate_flatters_a_small_sample(self):
        # 47/50 = 94% observed against a 90% target: looks like a pass, isn't one.
        results = [(0.9, True)] * 47 + [(0.9, False)] * 3
        d = decide(results, target=0.90)
        assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert d.needed_n is not None and d.needed_n > 50
        assert not d.can_automate

    def test_automate_with_review_when_a_confident_subset_qualifies(self):
        # Low-confidence cases fail, high-confidence ones pass.
        results = [(0.95, True)] * 60 + [(0.30, False)] * 20
        d = decide(results, target=0.90)
        assert d.verdict is Verdict.AUTOMATE_WITH_REVIEW
        assert d.operating_point is not None
        assert d.operating_point.n_deferred == 20
        assert d.can_automate

    def test_assist_only_when_mediocre_and_unsalvageable(self):
        results = [(0.5, True)] * 35 + [(0.5, False)] * 25
        d = decide(results, target=0.95)
        assert d.verdict is Verdict.ASSIST_ONLY

    def test_do_not_automate_when_clearly_bad(self):
        results = [(0.5, True)] * 10 + [(0.5, False)] * 50
        d = decide(results, target=0.95)
        assert d.verdict is Verdict.DO_NOT_AUTOMATE
        assert not d.can_automate

    def test_no_cases_is_insufficient_evidence(self):
        d = decide([], target=0.95)
        assert d.verdict is Verdict.INSUFFICIENT_EVIDENCE

    def test_rejects_invalid_target(self):
        with pytest.raises(ValueError):
            decide([(0.9, True)], target=1.5)


class TestHonesty:
    def test_a_bad_agent_is_never_rescued_by_thresholding(self):
        # Confidence is pure noise, so no threshold should produce a ship verdict.
        results = [(c / 100, False) for c in range(1, 61)]
        d = decide(results, target=0.95)
        assert not d.can_automate

    def test_flat_confidence_is_flagged_not_silently_ignored(self):
        results = [(0.8, True)] * 40 + [(0.8, False)] * 10
        d = decide(results, target=0.95)
        assert any("same confidence" in n for n in d.notes)
        assert d.operating_point is None

    def test_miscalibrated_confidence_blocks_the_operating_point(self):
        # Confidence is inverted: high confidence is wrong, low is right.
        results = [(0.99, False)] * 30 + [(0.01, True)] * 30
        d = decide(results, target=0.90)
        assert d.operating_point is None
        assert any("calibration" in n.lower() for n in d.notes)

    def test_near_miss_coverage_is_explained(self):
        # A qualifying threshold exists but covers too little to be useful.
        # 60 perfect cases clear a 90% lower bound (0.940); 30 would not (0.887).
        results = [(0.99, True)] * 60 + [(0.1, False)] * 40
        d = decide(results, target=0.90, min_coverage=0.75)
        assert d.operating_point is None
        assert any("below the" in n for n in d.notes)

    def test_reason_is_always_populated(self):
        for results in ([(0.9, True)] * 100, [(0.5, False)] * 100, []):
            assert decide(results).reason
