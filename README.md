# gonogo

**47 out of 50 is not 94%.** It lands somewhere between 84% and 98%, so if you were aiming at 90%, you can't yet say you got there.

gonogo turns your agent's results on real cases into a deployment decision: ship it, test a candidate human-review threshold on fresh cases, or don't automate. When there are too few cases to know, it says so. It also grades the decisions not to act: no reply, wait, ask a human.

## Install

```bash
pip install gonogo-eval
```

The package is `gonogo-eval` on PyPI (the plain `gonogo` name belongs to an unrelated project). It imports as `gonogo`.

## Try it

You only need a pass count:

```
$ gonogo 47/50 --target 0.9
INSUFFICIENT EVIDENCE
observed 94.0% on 50 cases is above the 90% target, but the 95% interval reaches down to 83.8%
47/50 looks like 94%; you'd need about 204 cases at this rate to claim 90%.
```

From Python: `gonogo.decide_counts(47, 50, target=0.90)`. Per-case results add a review
threshold, calibration and the actions table.

On a terminal the output is colored: green is good, yellow is a hazard, red is poor. Here
the verdict is yellow and the 83.8% is red. `NO_COLOR=1` turns colors off. To color your
own output, use `gonogo.style.paint(report.summary(), target=report.decision.target)`.

## Grade what your agent didn't do

<a id="when-the-right-move-is-not-to-answer"></a>

Agent logs record what the agent did. A message the agent left alone leaves no row, so a
correct silence and a missed reply look the same. Give each case the action that was right,
have the agent return the action it took, and gonogo grades every decision, including the
ones where nothing was sent. A case with an `expected_action` passes only if the agent took
that action; the scorer is not called.

```python
from gonogo import Case, Prediction, evaluate

cases = [
    Case(input=msg1, expected=reference_reply, id="m1"),                    # needs an answer
    Case(input=msg2, expected=None, id="m2", expected_action="no_reply"),   # needs nothing
    Case(input=msg3, expected=None, id="m3", expected_action="ask_human"),  # needs the owner
]

def agent(case):
    action, draft, confidence = triage(case.input)
    if action == "reply":
        return draft, confidence
    return Prediction(None, confidence=confidence, action=action)

print(evaluate(agent, cases, scorer=reply_scorer, target=0.90).markdown())
```

The report gets an actions table (`report.actions()` returns the same counts as a dict).
This one is from [`examples/inbox_triage`](examples/inbox_triage): 329 held-out Enron
messages, with a fine-tuned encoder as the gate.

```
## Actions

| Expected  | Taken     | Cases |
| --------- | --------- | ----- |
| answer    | answer    |     7 |
| answer    | ask_human |     3 |
| ask_human | ask_human |   107 |
| ask_human | answer    |    10 |
| ask_human | no_reply  |    38 |
| ask_human | wait      |     4 |
| no_reply  | no_reply  |   137 |
| no_reply  | answer    |     2 |
| no_reply  | ask_human |     7 |
| wait      | wait      |     6 |
| wait      | answer    |     1 |
| wait      | ask_human |     3 |
| wait      | no_reply  |     4 |

- Acted when it should have held back: 13
- Held back when it should have acted: 3
```

The encoder leaves 42 messages the owner needed with no reply and no hand-off
(`ask_human` → `no_reply` or `wait`). On the same cases, Claude leaves 5 and TypeSafe's
Jev 3. Under a declared cost rule the three pass rates can't be told apart, and the
encoder's mistakes cost about twice as much per message. The pass rate alone does not show
this. The example replays offline with no API key: three gates, paired comparisons, a
frozen-threshold test and the cost rule.

### Action names and confidence

- In JSONL, `expected_action` is a top-level key. Leave it out and the case expects an
  answer, so existing files load and save unchanged.
- Action names are plain strings. Two have a special meaning: `"answer"`, and `"abstain"`,
  which means *I defer this one to a person* (the selective-prediction sense).
- The report names two mistakes: *acted when it should have held back* (answered a case that
  expected a non-answer) and *held back when it should have acted* (took a non-answer action
  on a case that expected an answer).
- Every action except `"abstain"` is a decision the agent made, so it keeps the agent's
  confidence. A confident `"wait"` on a message that needed a reply is a silent miss in
  production, and nobody reviews it. It counts against precision at every threshold instead
  of hiding below one.
- Only `"abstain"` on a case that expects an answer counts as a deferral: it fails and sits
  at confidence 0.0, as abstention always has. Name a "say nothing" decision `"no_reply"`,
  not `"abstain"`.
- `Prediction(abstained=True)` still works and means `action="abstain"`. Every non-answer
  action sets `abstained`.

## Why a score is not a decision

Several good eval frameworks exist. gonogo is not another one. It converts a score into a
deployment decision you can defend at the sample sizes pilots run: forty to a hundred cases,
not ten thousand. On a set that small, three things go wrong.

**The point estimate flatters you.** 47/50 reads as 94%. The 95% interval is [83.8%, 97.9%]. Most tools print the 94% and stop.

**Accuracy is the wrong question.** The useful question is how much work you can hand over at 98% precision, and how many cases end up on someone's desk. Answering it takes a risk–coverage curve over an abstention threshold. Few tools compute one.

**Someone still has to decide.** A dashboard does not say ship or don't ship. That call should follow from the numbers.

## Usage

```python
from gonogo import Case, evaluate
from gonogo.scoring import fields

cases = Case.from_jsonl("invoices.jsonl")     # your real cases

def agent(case):
    result = my_pipeline(case.input)
    return result.data, result.confidence      # confidence is optional; it enables review thresholds

report = evaluate(agent, cases, scorer=fields(["vendor", "total"]), target=0.95)
print(report.markdown())
```

Your agent is any callable. No base class, no decorator, no framework to adopt. Return a bare value, a `(value, confidence)` tuple, or a `Prediction` to signal abstention or another non-answer action (see [Grade what your agent didn't do](#grade-what-your-agent-didnt-do)).

### A deterministic example, no API key

```bash
python examples/invoice_extraction.py
```

Sixty simulated invoices and an agent that is good but not perfect. The example is
deterministic (stable per-case seeds). Excerpt from its report:

```
# Score report: Extract fields from invoice

**ASSIST ONLY**: Use it to draft, keep a human on every case.

Pass rate 93.3% [84.1%, 97.4%] is below the 95% target and no confident subset reaches it; useful as a draft-generator, not as an unattended step.

| | |
| --- | --- |
| Cases evaluated | 60 |
| Passed | 56 |
| Pass rate | 93.3% [84.1%, 97.4%] |
| Target | 95% |
| Calibration error | 0.13 (well calibrated) |
```

The full report also has a calibration table. This agent is well calibrated above 0.8
(stated 0.92, actual 96%) and has a gap in the 0.6–0.8 band (stated 0.65, actual 75%, but
only four cases). Those bins describe the set. They are not enough to set a threshold on.

## The five verdicts

A review threshold selected on the measured set is a candidate. It does not authorize shipping.

| Verdict | Meaning |
| --- | --- |
| `AUTOMATE` | The pass rate's lower bound clears your target. Ship it. |
| `AUTOMATE WITH REVIEW` | A candidate threshold found on this set; not permission to deploy. Test the fixed threshold on a fresh holdout before routing cases by it. `can_automate` stays false; holdout signoff is separate. |
| `ASSIST ONLY` | Useful as a draft generator, not as an unattended step. |
| `DO NOT AUTOMATE` | Not a fit for this workflow as scoped. |
| `INSUFFICIENT EVIDENCE` | The estimate looks good but the sample can't support the claim. The report says roughly how many cases you'd need. |

## Why confidence matters

If your agent reports a confidence for every case, `gonogo` finds a candidate abstention threshold: the one with the most coverage whose measured precision's *lower bound* clears your target. It reports the whole curve. It selected that threshold on this set, so re-measure the fixed cut point on fresh cases before deployment:

| Confidence floor | Handled | Precision | To review |
| --- | --- | --- | --- |
| 0.36 | 100% | 88.3% [77.8%, 94.2%] | 0 |
| 0.70 | 87% | 94.2% [84.4%, 98.0%] | 8 |
| 0.85 | 82% | 98.0% [89.3%, 99.6%] | 11 |
| 0.89 | 57% | 100.0% [89.8%, 100.0%] | 26 |
| 0.99 | 2% | 100.0% [20.7%, 100.0%] | 59 |

The last row is why gonogo uses the lower bound. 100% precision on a single case is not an operating point, and the interval says so.

Against a 95% target, no threshold in this table works. The 0.85 row shows 98.0% precision, but its lower bound is 89.3%. The verdict is `ASSIST ONLY`, and you learn that before the agent is in production.

gonogo also checks whether the confidence means anything. An expected calibration error above 0.15 flags the scale as not a probability. The candidate operating point stays, because precision at a cut point is measured directly and does not depend on the scale. That same-set estimate is optimistic until checked on a fresh holdout. Calibration and discrimination are different properties; the next section shows a model that fails one and aces the other.

## Banking77: a real measurement

[![Fine-tuning an encoder and getting a go/no-go verdict — thomas + gonogo](https://i.ytimg.com/vi/ozWITnaJtf4/maxresdefault.jpg)](https://youtu.be/ozWITnaJtf4?t=2334)

**Video (40:33):** the Banking77 canary scored by a ModernBERT-small encoder fine-tuned through [thomas](https://github.com/keppy/thomas) on Modal. The verdict, intervals and operating point were computed live, including a scorer bug that briefly reported 0.0% and its fix on camera. The live demo starts at 37:16; the link opens at the verdict, 38:54. [Writeup](https://www.keppylab.com/blog/2026/09/21/banking77-canary-872-pass-two-dead-runs-one-false-alarm/).

- 87.2% [82.5%, 90.8%] pass rate against a 95% target → **AUTOMATE WITH REVIEW** candidate (fresh holdout required before deployment)
- at confidence ≥ 0.91: 98.3% precision [95.1%, 99.4%] on 71% of cases, the rest routed to a human
- calibration error 0.03 after temperature scaling; +10.0 points over the TF-IDF baseline on the same 250 cases (p < 0.001)

`examples/banking77_routing.py` runs against [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets): 13,083 real retail-bank customer messages across 77 intents. The agent is an ordinary TF-IDF nearest-centroid baseline, stdlib only, no model API.

On a pilot-sized sample of 250 cases against a 95% target:

```
ASSIST ONLY: Use it to draft, keep a human on every case.
Pass rate 77.2% [71.6%, 82.0%]   (full 3,080-case split: 80.7% [79.3%, 82.1%])
Calibration error 0.42 (ranks cases, scale unreliable)
```

The calibration table:

| Stated confidence | Cases | Mean confidence | Actual accuracy |
| --- | --- | --- | --- |
| 0.0–0.2 | 83 | 0.11 | 61% |
| 0.2–0.4 | 74 | 0.30 | 78% |
| 0.4–0.6 | 50 | 0.51 | 84% |
| 0.6–0.8 | 30 | 0.70 | 97% |
| 0.8–1.0 | 13 | 0.87 | 100% |

The model is badly miscalibrated: it says 0.11 and is right 61% of the time. It still ranks cases almost perfectly; accuracy rises in every bin. Calibration and ranking are different properties, and treating them as one throws away a usable signal. `gonogo` reports the calibration error, says the number isn't a probability, and still measures precision at each cut point, because that measurement doesn't depend on the scale.

## On LLM-as-judge

`scoring.judge` takes any `complete(prompt) -> str` callable, so there's no provider SDK in the dependency tree.

An unvalidated judge fails silently. Hand-label a subset and run `validate_judge(judge_labels, human_labels)`:

```
judge NOT USABLE: 90% agreement, kappa 0.00 on 30 hand-labelled cases
(kappa 0.00 is below 0.60 and it passes cases you failed; fix the rubric)
```

A judge that passes everything scores 90% agreement on a set that is 90% passes, and carries no information. gonogo gates on kappa, not agreement.

### What kappa measures

Two raters grade the same cases blind and you count how often they agree. Cohen's kappa is that agreement minus what they would have hit by chance, rescaled so that 1.0 is perfect agreement and 0 is no better than luck:

```
kappa = (observed agreement − chance agreement) / (1 − chance agreement)
```

Chance agreement comes from each rater's base rate. If rater A passes 80% of cases and rater B passes 70%, two raters with those habits who were otherwise flipping coins would both say *pass* on 0.80 × 0.70 = 56% of cases and both say *fail* on 0.20 × 0.30 = 6%. They agree 62% of the time without looking at a case. That 62% is the chance term, and kappa only credits agreement above it. So 70% observed agreement against a 62% chance floor is kappa ≈ 0.21: the two raters barely beat their base rates.

In the judge example above, the judge passes everything (100%) and the human passes 90%. Chance agreement is 1.00 × 0.90 + 0.00 × 0.10 = 90%, exactly the observed agreement, so kappa is (0.90 − 0.90) / (1 − 0.90) = 0. All of the 90% came from the base rate. Agreement counts how often two raters said the same thing; kappa measures how much of that came from looking at the case.

The default gate is kappa 0.60, not 90% agreement. The threshold is conventional (Landis and Koch call 0.61–0.80 "substantial"), not derived, and reasonable people put it elsewhere; `validate_judge(..., min_kappa=...)` moves it. One edge case: if both raters are constant and identical (the judge passed everything on a subset the human also fully passed), chance agreement is 100%, kappa is 0/0, and the judge is rejected rather than reported as perfect. Agreement on a set with nothing to disagree about demonstrates nothing.

**Say where the labels came from.** The second argument used to be called `human_passes`. A function that calls every label human lets a second model's labels be described as human validation. Pass `label_source`:

```python
validate_judge(judge_labels, model_labels, label_source="model",
               label_source_note="nemotron-3-super-120b via OpenRouter")
```

A kappa of 0.42 would be **NOT USABLE** at the default 0.60 gate, even against model labels.

`"human"` is the default and the only source that earns the word *human* anywhere in the output. `"structural"` is for deterministic checks (duplicates, dangling references, malformed output): they can catch a judge waving broken cases through, but cannot establish semantic correctness. Structural-only validation never marks a judge usable; collect independent human positives and negatives before using it to score training or shipping. The old keyword still works and warns.

## Correlated cases

Ten runs of one scenario with five checks each is not fifty trials. The five checks in a run share one draw of the system, and an interval over fifty says more than the evidence does. It can also center on an average that hides a check failing in every run:

```
ungrouped: 80.0% [67.0%, 88.8%] on 50 cases      INSUFFICIENT EVIDENCE
grouped  :  0.0% [ 0.0%, 27.8%] on 10 groups     DO NOT AUTOMATE
```

Same fifty cases. Put a `group` on each case (one per run, per document, per conversation) and ask for `group_rule="all"`: one trial per group, passing only if every case in it passed, with the interval over groups. The report keeps the case count as context and adds a row per recurring case id, such as *`recovery`: 0 of 10 groups*. That row is usually the one you need.

```python
cases = [Case(input=..., expected=..., id="recovery", group="run-3"), ...]
report = evaluate(agent, cases, scorer=scorer, group_rule="all")
```

A mix of grouped and ungrouped cases is an error. `required_n` answers in groups when the trials are groups.

Scorers may also take the case: `def scorer(output, expected, case)` is called with the `Case` as the third argument, so one run can route different cases to different scorers or read `case.metadata`. Two-argument scorers are unchanged.

## Comparing two agents

A higher pass rate after swapping in a new model does not show the new model is better. Two overlapping intervals say little. When both agents ran the *same* cases, the results are paired. `compare()` runs McNemar's test on the shared case ids (or groups, when grouped); for that test, cases where the agents agree carry no information. The difference interval uses conservative simultaneous Wilson bounds on the two discordant rates and does not collapse at the boundaries. A winner is declared only when McNemar's test and that interval agree. When they disagree (the test rejects but the interval still includes zero), the verdict says *borderline* instead of putting "B is better" next to an interval that includes no difference. Grouped runs must pair the same case membership within each shared group.

```python
from gonogo import compare

print(compare(baseline_report, claude_report).summary())
```

A Claude agent vs the TF-IDF baseline on the same 250 Banking77 cases (`examples/banking77_claude_agent.py`). The difference interval below is recomputed from the paired counts with the current Wilson method:

```
Shared cases        250
Agent A pass rate   77.2%
Agent B pass rate   81.6%
Difference          +4.4% [-3.8%, +12.4%] at 95%
Disagreements       45 (17 only-A, 28 only-B)
McNemar p           0.1352
Verdict             no detectable difference; the sample cannot separate them
```

A 4.4-point gain would headline a slide. The paired test says this sample can't support it. `compare()` accepts `Report` objects or `Report.to_dict()` payloads, because the usual comparison is today's run against a JSON file written last week. `to_dict()` carries per-case outcomes for that reason.

## Reports

`report.markdown()` for a terminal or a PR comment; `report.html()` for a self-contained styled page with no external assets, to hand to whoever signs off; `report.to_dict()` for JSON, including per-case outcomes so runs can be paired and compared later.

## One loop, three libraries

<a id="works-with"></a>

`gonogo` is one of three libraries that take a task from *which model?* to *ship it or not*:

```
your task ──► evalroute ─ choose a (model, effort) arm;
                │         measured where available, priors marked
                ▼
your cases ──► thomas ── train on labels (encoder) or score_text (RL);
                │         evaluate on untouched cases with gonogo
                ▼
              gonogo ── ship it, test a threshold on fresh cases, or walk away
```

- **[gonogo](https://github.com/keppy/gonogo)** (this repo): the decision layer. Any
  agent, your real cases, a target; the verdict comes with the interval behind it.
- **[thomas](https://github.com/keppy/thomas)**: the training harness. RL shares
  `score_text` across training and held-out evaluation; encoder SFT instead learns labels
  and needs a separate held-out evaluation.
- **[evalroute](https://github.com/keppy/evalroute)**: the routing layer. Before any of it
  starts, it classifies the task and hands back the arm, with measured costs where available
  and priors otherwise. It collects ratings to prioritize the next controlled batch.
- **The Hermes plugins**: thin adapters over the same three, inside your agent's session:
  [gonogo](https://github.com/keppy/hermes-plugin-gonogo) for decisions,
  [thomas](https://github.com/keppy/hermes-plugin-thomas) with a human approval hook on its
  training tool, and [evalroute](https://github.com/keppy/hermes-plugin-evalroute) for
  `/route` before the first turn. All three are in the Hermes plugin catalog.

What thomas and gonogo agree on is written down and versioned in thomas's
[docs/CONTRACT.md](https://github.com/keppy/thomas/blob/main/docs/CONTRACT.md) (contract
version 1): case ids, reward → `passed` / `score`, the confidence definition, the artifact a
training run leaves on disk, eval splits, and the gonogo calls thomas depends on.

## Non-goals

This is a reference implementation, about 2,000 lines, readable in one sitting. It will not grow into:

- a hosted service or dashboard
- tracing / observability
- prompt management or versioning
- an agent framework
- a public leaderboard

For those, use one of the platforms.

## License

MIT
