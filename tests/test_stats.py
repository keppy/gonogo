import math

import pytest

from gonogo.stats import (
    best_operating_point,
    expected_calibration_error,
    reliability_table,
    required_n,
    risk_coverage,
    wilson,
    z_for,
)


class TestWilson:
    def test_known_value(self):
        # 47/50 at 95%: Wilson interval is [0.8378, 0.9794], verified by hand.
        iv = wilson(47, 50)
        assert iv.point == pytest.approx(0.94)
        assert iv.low == pytest.approx(0.8378, abs=1e-3)
        assert iv.high == pytest.approx(0.9794, abs=1e-3)

    def test_the_premise_47_of_50_is_not_94_percent(self):
        # The whole reason this library exists.
        iv = wilson(47, 50)
        assert iv.low < 0.90, "a 94% point estimate on n=50 cannot support a 90% claim"

    def test_stays_in_unit_interval_at_extremes(self):
        assert wilson(0, 10).low == 0.0
        # Exactly 1.0 in exact arithmetic; floating point lands a hair under.
        assert wilson(10, 10).high == pytest.approx(1.0)
        assert wilson(10, 10).high <= 1.0
        assert 0.0 < wilson(10, 10).low < 1.0  # never claims certainty
        assert 0.0 < wilson(0, 10).high < 1.0

    def test_interval_narrows_as_n_grows(self):
        widths = [wilson(round(0.9 * n), n).width for n in (10, 50, 200, 1000)]
        assert widths == sorted(widths, reverse=True)

    def test_higher_confidence_is_wider(self):
        assert wilson(45, 50, 0.99).width > wilson(45, 50, 0.95).width > wilson(45, 50, 0.80).width

    def test_zero_cases_is_maximally_uncertain(self):
        iv = wilson(0, 0)
        assert math.isnan(iv.point) and iv.low == 0.0 and iv.high == 1.0

    def test_rejects_impossible_input(self):
        with pytest.raises(ValueError):
            wilson(11, 10)
        with pytest.raises(ValueError):
            wilson(-1, 10)

    def test_z_for_standard_levels(self):
        assert z_for(0.95) == pytest.approx(1.96, abs=1e-3)
        assert z_for(0.99) == pytest.approx(2.576, abs=1e-3)
        # Interpolated levels use the approximation and should stay monotonic.
        assert z_for(0.85) < z_for(0.95) < z_for(0.975)


class TestRequiredN:
    def test_returns_none_when_rate_is_at_or_below_target(self):
        assert required_n(0.95, 0.95) is None
        assert required_n(0.80, 0.95) is None

    def test_finds_a_sufficient_n(self):
        n = required_n(0.98, 0.95)
        assert n is not None
        assert wilson(round(0.98 * n), n).low >= 0.95

    def test_is_the_minimum_such_n(self):
        n = required_n(0.98, 0.95)
        assert wilson(round(0.98 * (n - 1)), n - 1).low < 0.95

    def test_closer_to_target_needs_more_cases(self):
        assert required_n(0.96, 0.95) > required_n(0.99, 0.95)


class TestRiskCoverage:
    def test_full_coverage_at_lowest_threshold(self):
        results = [(0.5, True), (0.7, True), (0.9, False)]
        curve = risk_coverage(results)
        assert curve[0].coverage == 1.0
        assert curve[0].n_deferred == 0

    def test_selective_thresholds_reduce_coverage(self):
        results = [(0.5, False), (0.7, True), (0.9, True)]
        curve = risk_coverage(results)
        coverages = [p.coverage for p in curve]
        assert coverages == sorted(coverages, reverse=True)

    def test_abstaining_on_low_confidence_raises_precision(self):
        # Confidence tracks correctness, so cutting the bottom should help.
        results = [(0.2, False), (0.3, False), (0.9, True), (0.95, True)]
        curve = risk_coverage(results)
        assert curve[0].precision.point == 0.5
        assert curve[-1].precision.point == 1.0

    def test_deferred_plus_covered_equals_n(self):
        results = [(0.1, True), (0.4, False), (0.6, True), (0.8, True), (1.0, False)]
        for p in risk_coverage(results):
            assert p.n_covered + p.n_deferred == len(results)

    def test_empty_and_invalid(self):
        assert risk_coverage([]) == []
        with pytest.raises(ValueError):
            risk_coverage([(1.5, True)])


class TestBestOperatingPoint:
    def test_prefers_highest_coverage_meeting_target(self):
        results = [(0.9, True)] * 30 + [(0.5, False)] * 5
        p = best_operating_point(results, target_precision=0.80)
        assert p is not None
        assert p.threshold == 0.9
        assert p.n_covered == 30

    def test_uses_lower_bound_not_point_estimate(self):
        # Three perfect cases at high confidence: 100% observed, but the lower
        # bound is far below 95%, so this must not be offered as an operating point.
        results = [(0.99, True)] * 3 + [(0.2, False)] * 20
        assert best_operating_point(results, target_precision=0.95) is None

    def test_returns_none_when_nothing_qualifies(self):
        results = [(0.5, False)] * 20
        assert best_operating_point(results, target_precision=0.95) is None

    def test_respects_min_coverage_floor(self):
        results = [(0.99, True)] * 40 + [(0.1, False)] * 60
        assert best_operating_point(results, 0.90, min_coverage=0.0) is not None
        assert best_operating_point(results, 0.90, min_coverage=0.75) is None


class TestCalibration:
    def test_perfectly_calibrated_is_near_zero(self):
        # In each bucket, the pass fraction matches the stated confidence.
        results = [(0.9, True)] * 9 + [(0.9, False)]
        assert expected_calibration_error(results) == pytest.approx(0.0, abs=0.01)

    def test_overconfident_model_has_high_error(self):
        results = [(0.99, False)] * 5 + [(0.99, True)] * 5
        assert expected_calibration_error(results) > 0.4

    def test_empty_is_nan(self):
        assert math.isnan(expected_calibration_error([]))

    def test_confidence_of_one_is_counted(self):
        # Right edge must land in the last bin rather than being dropped.
        rows = reliability_table([(1.0, True), (1.0, False)], bins=5)
        assert sum(r["n"] for r in rows) == 2

    def test_reliability_table_reports_accuracy_per_bin(self):
        results = [(0.1, False), (0.1, False), (0.9, True), (0.9, True)]
        rows = reliability_table(results, bins=2)
        assert len(rows) == 2
        assert rows[0]["accuracy"] == 0.0
        assert rows[1]["accuracy"] == 1.0
