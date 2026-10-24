"""Terminal colors: green good, yellow hazard, red poor; plain unless it's a terminal."""

import io
import re

import pytest

from gonogo import Case, evaluate
from gonogo.cli import main
from gonogo.style import paint, use_color

G, Y, R = "\033[32;1m", "\033[33;1m", "\033[31;1m"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def test_plain_unless_a_terminal_or_forced_and_no_color_wins(monkeypatch):
    assert use_color(io.StringIO()) is False
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert use_color(io.StringIO()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert use_color(io.StringIO()) is False


@pytest.mark.parametrize("text,color", [
    ("93.4% [90.9%, 95.2%]", G),    # lower bound clears the target
    ("91.6% [88.1%, 94.2%]", Y),    # only the point does: looks fine, not proven
    ("78.1% [73.3%, 82.2%]", R),    # the point misses
])
def test_interval_colored_against_the_target(text, color):
    assert paint(text, target=0.9, enabled=True).startswith(color + text.split(" ")[0])


def test_do_not_automate_is_red_not_green():
    assert f"{R}DO NOT AUTOMATE" in paint("DO NOT AUTOMATE", enabled=True)
    assert f"{Y}AUTOMATE WITH REVIEW" in paint("AUTOMATE WITH REVIEW", enabled=True)


def test_paint_only_adds_color_codes():
    report = evaluate(lambda c: c.expected, [Case(input=i, expected=i) for i in range(30)], target=0.9)
    painted = paint(report.summary(), target=report.decision.target, enabled=True)
    assert painted != report.summary()
    assert re.sub(r"\033\[[\d;]*m", "", painted) == report.summary()


def test_cli_is_plain_in_a_pipe_and_colored_when_forced(capsys, monkeypatch):
    main(["47/50", "--target", "0.9"])
    assert "\033[" not in capsys.readouterr().out
    monkeypatch.setenv("FORCE_COLOR", "1")
    main(["47/50", "--target", "0.9"])
    out = capsys.readouterr().out
    assert out.startswith(f"{Y}INSUFFICIENT EVIDENCE") and f"{R}83.8%" in out
