#!/usr/bin/env python3
"""Verdict-moment cast: question -> command -> verdict, at watchable speed.

Output lines are verbatim from the deterministic run of
examples/invoice_extraction.py (head of the report). Only timing is authored.
"""
import json, random, time

random.seed(7)
OUT = r"C:/Users/keppy/AppData/Local/hermes/cache/scratch/gonogo-verdict.cast"

QUESTION = "# 56 of 60 invoices right - good enough to ship?"
CMD = "python examples/invoice_extraction.py"
LINES = [
    "# Score report: Extract fields from invoice",
    "",
    "**ASSIST ONLY**: Use it to draft, keep a human on every case.",
    "",
    "Pass rate 93.3% [84.1%, 97.4%] is well short of the 95% target and no",
    "confident subset reaches it; useful as a draft-generator, not as an",
    "unattended step.",
    "",
    "| | |",
    "| --- | --- |",
    "| Cases evaluated | 60 |",
    "| Passed | 56 |",
    "| Pass rate | 93.3% [84.1%, 97.4%] |",
    "| Target | 95% |",
    "| Calibration error | 0.13 (well calibrated) |",
]

events = []
def emit(t, text):
    events.append([round(t, 3), "o", text])

t = 0.4
emit(t, QUESTION + "\r\n"); t += 1.0     # beat 1: the question
emit(t, "\r\n"); t += 0.3
for ch in CMD:                            # beat 2: the command, typed
    emit(t, ch)
    t += 0.045 + random.uniform(-0.015, 0.035)
emit(t, "\r\n"); t += 0.5
for line in LINES:                         # beat 3: verdict lands as a block
    emit(t, line + "\r\n")
    t += 0.035
t += 3.2                                   # hold the resting frame

header = {"version": 2, "width": 80, "height": 24, "timestamp": int(time.time()),
          "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"}}
with open(OUT, "w", encoding="utf-8") as f:
    f.write(json.dumps(header) + "\n")
    for e in events:
        f.write(json.dumps(e) + "\n")
print(f"{len(events)} events, {t:.1f}s")
