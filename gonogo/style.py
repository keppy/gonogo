"""Terminal colors for gonogo's text output: green is good, yellow is a hazard, red is poor.

    from gonogo.style import paint
    print(paint(report.summary(), target=report.decision.target))

`paint` only adds color when it is writing to a terminal (or FORCE_COLOR is set), and never
when NO_COLOR is set (https://no-color.org). Otherwise it returns the text unchanged, so
logs, files and pipes stay plain. Nothing else in gonogo uses it: `summary()`, `markdown()`,
`html()` and `to_dict()` are the same as before.

What gets colored:

- an interval such as `78.1% [73.3%, 82.2%]`, when a target is given: green if the lower
  bound clears the target, yellow if only the point does (it looks fine but isn't proven),
  red if the point misses; the bracketed interval is dimmed;
- a verdict: AUTOMATE green; AUTOMATE WITH REVIEW, ASSIST ONLY and INSUFFICIENT EVIDENCE
  yellow; DO NOT AUTOMATE red;
- "does not hold", "misses the N% target" and similar in red; "clears the N% target" in green.

`color(text, "good" | "hazard" | "poor" | "dim")` colors one piece of text by meaning, for
scripts that print their own numbers.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, TextIO

__all__ = ["color", "paint", "use_color"]

_GREEN, _YELLOW, _RED, _DIM = "32;1", "33;1", "31;1", "2"

_INTERVAL = re.compile(r"(\d+(?:\.\d+)?)% \[(\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%\]")
_LOWER = re.compile(r"(reaches down to |lower bound )(\d+(?:\.\d+)?)%")
_VERDICTS = (("DO NOT AUTOMATE", _RED), ("AUTOMATE WITH REVIEW", _YELLOW),
             ("INSUFFICIENT EVIDENCE", _YELLOW), ("ASSIST ONLY", _YELLOW), ("AUTOMATE", _GREEN))
_VERDICT = re.compile(r"(?<![A-Za-z])(" + "|".join(v for v, _ in _VERDICTS) + r")(?![A-Za-z])")
_BAD = re.compile(r"(does not hold|misses the \d+(?:\.\d+)?% target|don't route by it)")
_GOOD = re.compile(r"(clears the \d+(?:\.\d+)?% target|the threshold holds on fresh cases)")
_TONES = {"good": _GREEN, "hazard": _YELLOW, "poor": _RED, "dim": _DIM}


def use_color(stream: TextIO | None = None) -> bool:
    """True when output to `stream` (default stdout) should be colored."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = sys.stdout if stream is None else stream
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _wrap(code: str) -> Callable[[str], str]:
    return lambda s: f"\033[{code}m{s}\033[0m"


def _by_target(point: float, low: float, target_pct: float) -> str:
    if point < target_pct:
        return _RED
    return _GREEN if low >= target_pct else _YELLOW


def color(text: str, tone: str, enabled: bool | None = None) -> str:
    """Color one piece of text by meaning: "good" (green), "hazard" (yellow), "poor" (red) or
    "dim". Plain unless colors are on (see `use_color`)."""
    if tone not in _TONES:
        raise ValueError(f"tone must be one of {sorted(_TONES)}, got {tone!r}")
    if enabled is None:
        enabled = use_color()
    return _wrap(_TONES[tone])(text) if enabled else text


def paint(text: str, target: float | None = None, enabled: bool | None = None) -> str:
    """Color `text` for a terminal. `target` is a rate in (0, 1), e.g. 0.9.

    `enabled=None` decides from the environment and stdout (see `use_color`)."""
    if enabled is None:
        enabled = use_color()
    if not enabled:
        return text
    dim = _wrap(_DIM)
    verdict_colors = dict(_VERDICTS)
    out = []
    for line in text.split("\n"):
        if target is not None:
            t = 100 * target
            line = _INTERVAL.sub(lambda m: _wrap(_by_target(float(m[1]), float(m[2]), t))(f"{m[1]}%")
                                 + dim(f" [{m[2]}%, {m[3]}%]"), line)
            line = _LOWER.sub(lambda m: m[1] + _wrap(_by_target(float(m[2]), float(m[2]), t))(f"{m[2]}%"), line)
        line = _VERDICT.sub(lambda m: _wrap(verdict_colors[m[1]])(m[1]), line, count=1)
        line = _BAD.sub(lambda m: _wrap(_RED)(m[1]), line)
        line = _GOOD.sub(lambda m: _wrap(_GREEN)(m[1]), line)
        out.append(line)
    return "\n".join(out)
