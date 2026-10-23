import pytest

from gonogo import Verdict, decide, decide_counts
from gonogo.cli import format_counts, main


class TestDecideCounts:
    def test_matches_decide_on_the_same_trials(self):
        d = decide_counts(47, 50, target=0.90)
        ref = decide([(1.0, True)] * 47 + [(1.0, False)] * 3, target=0.90)
        assert d.verdict is ref.verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert d.needed_n == ref.needed_n
        assert d.pass_rate.low == pytest.approx(ref.pass_rate.low)

    def test_no_operating_point_without_confidence(self):
        d = decide_counts(188, 200, target=0.95)
        assert d.operating_point is None
        assert d.verdict is Verdict.ASSIST_ONLY

    def test_automate_when_the_count_clears_the_bar(self):
        assert decide_counts(400, 400, target=0.95).verdict is Verdict.AUTOMATE

    @pytest.mark.parametrize("passed,total", [(51, 50), (-1, 50), (0, 0)])
    def test_rejects_impossible_counts(self, passed, total):
        with pytest.raises(ValueError):
            decide_counts(passed, total)

    @pytest.mark.parametrize("passed,total", [(47.0, 50), (True, 50), ("47", 50)])
    def test_rejects_non_integers(self, passed, total):
        with pytest.raises(TypeError):
            decide_counts(passed, total)


class TestCli:
    def test_47_of_50_prints_the_verdict_interval_and_needed_n(self, capsys):
        assert main(["47/50", "--target", "0.9"]) == 0
        out = capsys.readouterr().out.splitlines()
        d = decide_counts(47, 50, target=0.90)
        assert out[0] == "INSUFFICIENT EVIDENCE"
        assert "83.8%" in out[1]
        assert out[2] == f"47/50 looks like 94%; you'd need about {d.needed_n} cases at this rate to claim 90%."

    def test_no_needed_n_line_when_the_claim_is_out_of_reach(self):
        text = format_counts(30, 50, target=0.9)
        assert text.splitlines()[0] == "ASSIST ONLY"
        assert "you'd need" not in text

    @pytest.mark.parametrize("arg", ["47", "51/50", "47/0", "a/b"])
    def test_bad_fraction_exits_2(self, arg, capsys):
        with pytest.raises(SystemExit) as exc:
            main([arg])
        assert exc.value.code == 2

    @pytest.mark.parametrize("target", ["0", "1", "1.5", "x"])
    def test_bad_target_exits_2(self, target):
        with pytest.raises(SystemExit) as exc:
            main(["47/50", "--target", target])
        assert exc.value.code == 2
