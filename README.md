# gonogo

**47 out of 50 is not 94%.** It lands somewhere between 84% and 98%, so if you were aiming at 90%, you can't yet say you got there.

Feed `gonogo` your agent and your real cases. Back comes a decision: ship it, ship it behind a human-review threshold, or walk away. When the honest answer is "you don't have enough cases to know," it says that instead of guessing.

This is deliberately **not** another eval framework — several good ones already exist. What none of them do is convert a score into a deployment decision you can defend at the sample sizes pilots actually run: forty to a hundred cases, not ten thousand.

```bash
pip install gonogo-eval
```

Installs as `gonogo-eval` (the plain `gonogo` name on PyPI belongs to an unrelated project); imports as `gonogo`.

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

It also checks whether your confidence means anything. Expected calibration error above 0.15 and the harness flags the confidence scale as not-a-probability — but it keeps the operating point, because precision at a cut point is measured directly from held-out results and doesn't depend on the scale meaning anything. Calibration and discrimination are different properties; see the next section for a model that fails one and aces the other.

## A real measurement

`examples/banking77_routing.py` runs against [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) — 13,083 genuine retail-bank customer messages across 77 intents. The agent is an ordinary TF-IDF nearest-centroid baseline, stdlib only, no model API.

On a pilot-sized sample of 250 cases against a 95% target:

[![Fine-tuning an encoder and getting a go/no-go verdict — thomas + gonogo](https://i.ytimg.com/vi/ozWITnaJtf4/maxresdefault.jpg)](https://youtu.be/ozWITnaJtf4?t=2334)

**Video (40:33):** the Banking77 canary scored by a ModernBERT-small encoder fine-tuned through [thomas](https://github.com/keppy/thomas) on Modal — the verdict, the confidence intervals, and the operating point worked live, including a scorer bug that briefly reported 0.0% and the fix on camera. Result: 87.2% [82.5%, 90.8%] pass rate vs the 95% target → **AUTOMATE WITH REVIEW**; at confidence ≥ 0.91, 98.3% precision [95.1%, 99.4%] on 71% of cases with the rest routed to a human; calibration error 0.03. Live demo from 37:16. [Writeup](https://www.keppylab.com/blog/2026/09/21/banking77-canary-872-pass-two-dead-runs-one-false-alarm/).

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

### What kappa actually measures

Two raters grade the same cases blind and you count how often they agree. Cohen's kappa is that agreement minus whatever they would have hit by chance, rescaled so that 1.0 is perfect agreement and 0 is no better than luck:

```
kappa = (observed agreement − chance agreement) / (1 − chance agreement)
```

The obvious question is where "chance agreement" comes from. It comes from each rater's own base rate. If rater A passes 80% of cases and rater B passes 70%, then two raters with those habits who were otherwise flipping coins would both say *pass* on 0.80 × 0.70 = 56% of cases and both say *fail* on 0.20 × 0.30 = 6%. Add those and they agree 62% of the time without ever looking at a case. That 62% is the chance term; kappa only credits agreement above it. So 70% observed agreement against a 62% chance floor is kappa ≈ 0.21 — the two raters are barely doing better than their biases alone would produce.

Now the example above. The judge passes everything (100%), the human passes 90%. Chance agreement is 1.00 × 0.90 + 0.00 × 0.10 = 90% — exactly the observed agreement — so kappa is (0.90 − 0.90) / (1 − 0.90) = 0. The 90% was entirely purchased by the base rate, and kappa says so. Agreement tells you how often two raters said the same thing; kappa tells you whether that was because they were both looking at the case.

This is why the gate is 0.60 rather than "90% agreement." The threshold is conventional (Landis and Koch call 0.61–0.80 "substantial"), not derived, and reasonable people put it elsewhere; `validate_judge(..., min_kappa=...)` moves it. One edge case: if both raters are constant and identical — the judge passed everything on a subset the human also fully passed — chance agreement is 100%, kappa is 0/0, and the judge is rejected rather than reported as perfect. Agreeing with someone about a set where there was nothing to disagree on has demonstrated nothing.

**Say where the labels came from.** The second argument used to be called `human_passes`, and a function that calls every label human is how a second model's labels end up described as human validation. Pass `label_source`:

```python
validate_judge(judge_labels, model_labels, label_source="model",
               label_source_note="nemotron-3-super-120b via OpenRouter")
# judge USABLE: 77% agreement, kappa 0.42 on 20 cases labelled by an independent
# model (nemotron-3-super-120b via OpenRouter) (kappa 0.42 clears 0.60 against
# model labels; this is agreement with model labels, not human validation)
```

`"human"` is the default and the only source that earns the word *human* anywhere in the output. `"structural"` is for deterministic checks (duplicates, dangling references, malformed output): they can catch a judge waving broken cases through, but cannot see meaning, so there the gate is leniency alone and kappa is reported without being the verdict. The old keyword still works and warns.

## Correlated cases

Ten runs of one scenario with four checks each is not forty trials. The four checks share one draw of the system, and an interval over forty says more than the evidence does. Worse, it can centre on an average that hides a check failing in every single run:

```
ungrouped: 80.0% [67.0%, 88.8%] on 50 cases      INSUFFICIENT EVIDENCE
grouped  :  0.0% [ 0.0%, 27.8%] on 10 groups     DO NOT AUTOMATE
```

Same fifty cases. Put a `group` on each case (one per run, per document, per conversation) and ask for `group_rule="all"`: one trial per group, passing only if every case in it passed, interval over groups. The report keeps the case count as context and adds a row per recurring case id -- *`recovery`: 0 of 10 groups* -- which is usually the line you wanted.

```python
cases = [Case(input=..., expected=..., id="recovery", group="run-3"), ...]
report = evaluate(agent, cases, scorer=scorer, group_rule="all")
```

A mix of grouped and ungrouped cases is an error, not a guess. `required_n` answers in groups when the trials are groups.

Scorers may also take the case: `def scorer(output, expected, case)` is called with the `Case` as the third argument, so one run can route different cases to different scorers or read `case.metadata`. Two-argument scorers are unchanged.

## Comparing two agents

Swapping in a new model and watching the pass rate rise is the most common way a team convinces itself of an improvement that isn't there. Two overlapping intervals tell you very little — but when both agents ran the *same* cases, the results are paired, and the paired test is strictly more powerful. `compare()` runs McNemar's test on the shared case ids: cases both agents got right carry no information about which is better, so they're excluded rather than padding the denominator.

```python
from gonogo import compare

print(compare(baseline_report, claude_report).summary())
```

Real output, a Claude agent vs the TF-IDF baseline on the same 250 Banking77 cases (`examples/banking77_claude_agent.py`):

```
Shared cases        250
Agent A pass rate   77.2%
Agent B pass rate   81.6%
Difference          +4.4% [-0.8%, +9.6%] at 95%
Disagreements       45 (17 only-A, 28 only-B)
McNemar p           0.1352
Verdict             no detectable difference; the sample cannot separate them
```

That's a 4.4-point improvement that would headline a slide — and the paired test says this sample can't back it up. `compare()` accepts `Report` objects or `Report.to_dict()` payloads interchangeably, because the realistic comparison is today's run against a JSON file written last week — `to_dict()` carries per-case outcomes for exactly this reason.

## Reports

`report.markdown()` for a terminal or a PR comment; `report.html()` for a self-contained styled page with no external assets, which you can hand to whoever signs off; `report.to_dict()` for JSON, including per-case outcomes so runs can be paired and compared later.

## Non-goals

This is a reference implementation, around 1,400 lines, readable in one sitting. It will not grow into:

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
