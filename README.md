# gonogo

**47 out of 50 is not 94%.** It lands somewhere between 84% and 98%, so if you were aiming at 90%, you can't yet say you got there.

Feed `gonogo` your agent and your real cases. Back comes a decision: ship it, ship it behind a human-review threshold, or walk away. When the honest answer is "you don't have enough cases to know," it says that instead of guessing.

This is deliberately **not** another eval framework — several good ones already exist. What none of them do is convert a score into a deployment decision you can defend at the sample sizes pilots actually run: forty to a hundred cases, not ten thousand.

```bash
pip install gonogo
```

## The idea

Evaluate an agent on a small set of real cases and three things go wrong.

**The point estimate flatters you.** 47/50 reads as 94%. The 95% interval is [83.8%, 97.9%]. Most tools print the 94% and stop there.

**Accuracy isn't what anyone's asking.** The real question is how much work you can hand over at 98% precision, and how many cases end up on someone's desk. Answering it means a risk–coverage curve over an abstention threshold. Almost nothing computes one.

**A dashboard isn't a decision.** Somebody still has to say ship or don't ship, and that call ought to fall out of the numbers rather than out of a meeting.

## Usage

```python
from gonogo import Case, evaluate
from gonogo.scoring import fields

cases = Case.from_jsonl("invoices.jsonl")     # your real cases

def agent(case):
    result = my_pipeline(case.input)
    return result.data, result.confidence      # confidence is optional but unlocks a lot

report = evaluate(agent, cases, scorer=fields(["vendor", "total"]), target=0.95)
print(report.markdown())
```

Your agent is any callable. No base class, no decorator, no framework to adopt. Return a bare value, a `(value, confidence)` tuple, or a `Prediction` when you want to signal abstention.

## What it tells you

Five verdicts, and only two of them mean ship:

| Verdict | Meaning |
| --- | --- |
| `AUTOMATE` | The pass rate's lower bound clears your target. Ship it. |
| `AUTOMATE WITH REVIEW` | Not good enough overall, but a confident subset is. Ship behind a threshold, route the rest to a person. |
| `ASSIST ONLY` | Useful as a draft generator, not as an unattended step. |
| `DO NOT AUTOMATE` | Not a fit for this workflow as scoped. |
| `INSUFFICIENT EVIDENCE` | The estimate looks good but your sample can't support the claim. Here's roughly how many cases you'd need. |

That last one is the whole reason this exists. It's the verdict an honest consultant gives and a dashboard never does.

## Why confidence matters

If your agent reports a per-case confidence, `gonogo` finds the abstention threshold that maximizes how much you can automate while keeping precision's *lower bound* above your target, and reports the whole curve so you can see the tradeoff:

| Confidence floor | Handled | Precision | To review |
| --- | --- | --- | --- |
| 0.36 | 100% | 88.3% [77.8%, 94.2%] | 0 |
| 0.70 | 87% | 94.2% [84.4%, 98.0%] | 8 |
| 0.85 | 82% | 98.0% [89.3%, 99.6%] | 11 |
| 0.89 | 57% | 100.0% [89.8%, 100.0%] | 26 |
| 0.99 | 2% | 100.0% [20.7%, 100.0%] | 59 |

That last row is why it uses the lower bound and not the point estimate. 100% precision on a single case is not an operating point, and the interval says so.

Note what the table above actually proves: against a 95% target, *no* threshold works here. The 0.85 row looks great at 98.0% until you read its lower bound of 89.3%. So the verdict is `ASSIST ONLY`, not a ship — which is the answer you want before you wire it into production, not after.

It also checks whether your confidence means anything. Expected calibration error above 0.15 and the harness refuses to recommend a threshold at all — thresholding on a number that doesn't track correctness is theater.

## A real measurement

`examples/banking77_routing.py` runs against [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) — 13,083 genuine retail-bank customer messages across 77 intents. The agent is an ordinary TF-IDF nearest-centroid baseline, stdlib only, no model API.

On a pilot-sized sample of 250 cases against a 95% target:

```
ASSIST ONLY: Use it to draft, keep a human on every case.
Pass rate 77.2% [71.6%, 82.0%]   (full 3,080-case split: 80.7% [79.3%, 82.1%])
Calibration error 0.42 (ranks cases, scale unreliable)
```

The calibration table is the interesting part:

| Stated confidence | Cases | Mean confidence | Actual accuracy |
| --- | --- | --- | --- |
| 0.0–0.2 | 83 | 0.11 | 61% |
| 0.2–0.4 | 74 | 0.30 | 78% |
| 0.4–0.6 | 50 | 0.51 | 84% |
| 0.6–0.8 | 30 | 0.70 | 97% |
| 0.8–1.0 | 13 | 0.87 | 100% |

That model is *badly* miscalibrated — it says 0.11 and is right 61% of the time — while still ranking cases almost perfectly. Those are two different properties, and conflating them is a common way to throw away a usable signal. `gonogo` reports the calibration error, says the number isn't a probability, and still measures precision at each cut point empirically, because that measurement doesn't depend on the scale being meaningful.

## On LLM-as-judge

`scoring.judge` takes any `complete(prompt) -> str` callable, so there's no provider SDK in the dependency tree.

An unvalidated judge is the most common silent failure in agent evaluation. Hand-label a subset and run `validate_judge(judge_labels, human_labels)`:

```
judge NOT USABLE: 90% agreement, kappa 0.00 on 30 hand-labelled cases
(kappa 0.00 is below 0.60 and it passes cases you failed; fix the rubric)
```

That's the trap. A judge that rubber-stamps everything scores 90% agreement on a set that's 90% passes, and carries no information whatsoever. Kappa is the gate, not agreement.

## Reports

`report.markdown()` for a terminal or a PR comment; `report.html()` for a self-contained styled page with no external assets, which you can hand to whoever signs off; `report.to_dict()` for JSON.

## Non-goals

This is a reference implementation, around 800 lines, readable in one sitting. It will not grow into:

- a hosted service or dashboard
- tracing / observability
- prompt management or versioning
- an agent framework
- a public leaderboard

If you want those, use one of the platforms. This does one thing.

## Run the example

```bash
python examples/invoice_extraction.py
```

Sixty simulated invoices, an agent that's good but not perfect, no API key required. Real output:

```
# Score report: Extract fields from invoice

**ASSIST ONLY**: Use it to draft, keep a human on every case.

Pass rate 88.3% [77.8%, 94.2%] is well short of the 95% target and no
confident subset reaches it; useful as a draft-generator, not as an
unattended step.

| Cases evaluated   | 60                   |
| Passed            | 53                   |
| Pass rate         | 88.3% [77.8%, 94.2%] |
| Target            | 95%                  |
| Calibration error | 0.11 (usable)        |
```

The calibration table in the full report is worth a look too: this agent is
well calibrated above 0.8 (stated 0.92, actual 98%) and badly calibrated in
the 0.6–0.8 band (stated 0.68, actual 33%). That's the kind of thing you want
to know before you pick a threshold.

## License

MIT
