# What Live Diagrammer taught gonogo

> **Status: implemented in 0.2.0** (2026-09-15). Items 1, 2 and 3 below landed
> as described, with two deviations: `group_rule` offers `"none"` and `"all"`
> only -- `"mean"` needs an interval for fractional successes that Wilson does
> not give, so it waits for a real need; and `Decision` gained `n_groups` and a
> `unit` property rather than a separate `n_cases`, since the report already
> holds the case count. The "also observed" items are still just observed.

Two features, both built locally around gonogo 0.1.1 at the Sep 12 hackathon
because the library had no seam for them. Both are generic. Both should move
up. A third, smaller API observation follows.

Local implementations to port from:

- `fullstackiest/live-diagrammer/report-card/score.py` (provenance banner, lines 25-29 and 237-243)
- `live-diagrammer/eval/demo_card.py` (run-level unit, lines 287-299; k-of-N rows, 301-315)

---

## 1. Label provenance on `validate_judge`

### The problem

`validate_judge(judge_passes, human_passes, ...)` names its second argument
`human_passes`, and `JudgeValidation` records nothing about where those labels
came from. Feed it labels from a second model and the result is
indistinguishable from human validation: same fields, same `usable`, same
`reason` text. The library will describe a model-validated judge in human
terms, and nothing downstream can tell.

This is not hypothetical. We had three label sources for the same 20 cases:

| tier | labels from | kappa vs judge |
| --- | --- | --- |
| bronze | deterministic structural checks (orphans, duplicates, rejections) | 0.36 |
| silver | an unrelated model family (nemotron-3-super-120b) | 0.42 |
| gold | two humans | 0.09 (the humans agreed with each other at 0.40) |

Silver's 0.42 would have been reported exactly as gold's would. The only thing
that stopped the card from claiming "human-validated" on model labels was a
banner in `score.py` that the library knew nothing about.

### The proposal

```python
@dataclass
class JudgeValidation:
    n: int
    agreement: float
    kappa: float
    judge_lenient: int
    judge_strict: int
    usable: bool
    reason: str
    label_source: str = "human"          # NEW: "human" | "model" | "structural" | free text
    label_source_note: str = ""          # NEW: e.g. "nemotron-3-super-120b via OpenRouter"

def validate_judge(
    judge_passes: list[bool],
    reference_passes: list[bool],        # RENAMED from human_passes (keep the old name as a deprecated alias)
    min_kappa: float = 0.6,
    min_n: int = 20,
    label_source: str = "human",
    label_source_note: str = "",
) -> JudgeValidation:
```

Behaviour:

- `reason` and every rendering (`markdown()`, `html()`, `to_dict()`) name the
  source: *"kappa 0.42 clears 0.60 against labels from an independent model
  (nemotron-3-super-120b via OpenRouter)"*.
- The word **human** appears in output only when `label_source == "human"`.
  This is the whole point: make the honest sentence the default sentence.
- `label_source="structural"` gets one extra line of interpretation, because
  structural labels can only detect a lenient judge: report `judge_lenient`
  as the headline (*"the judge passed N cases that fail a structural check"*)
  and note that kappa against structural labels is bounded and not the gate.

Tiers (bronze / silver / gold) are Live Diagrammer's names. Upstream should
carry the *source kind*, not the metal.

### Why upstream and not a wrapper

The wrapper works, but it has to be re-written by every user, and the failure
it prevents is silent. A library whose README says "hand-label a subset" and
whose parameter is called `human_passes` is inviting the shortcut of feeding it
model labels. The type should make the shortcut visible.

---

## 2. Correlated cases: the independent unit is not always the case

### The problem

`evaluate` treats every `Case` as an independent Bernoulli trial and puts the
Wilson interval over all of them. When several cases come from the same draw
of the system under test, that overstates precision.

Ours: ten replays of one recording, four segment checks each. Forty checks,
interval `[60%, 86%]`. But the four checks within a run share one model
sample. The honest unit was the run: *did every segment match in this run?*
Over ten runs that read `0 of 10, [0%, 28%]` before the dismiss fix and
`10 of 10, [72%, 100%]` after it. The 40-check interval was wrong in both
directions - too narrow, and centred on a number that hid a deterministic
failure (recovery failed in every single run) inside a 75% average.

### The proposal

A grouping key on the case, and an aggregation rule in `evaluate`:

```python
@dataclass
class Case:
    input: Any
    expected: Any
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    group: str | None = None             # NEW: cases sharing a group share one draw

def evaluate(
    agent, cases, scorer=None, task="task", target=0.95, level=0.95,
    min_coverage=MIN_USEFUL_COVERAGE, workers=1,
    group_rule: Literal["all", "mean", "none"] = "none",   # NEW
) -> Report:
```

- `group_rule="none"` (default): today's behaviour; every case independent.
- `group_rule="all"`: one trial per group, passing iff every case in the group
  passed. The interval and the verdict are over groups. This is the right rule
  when the group is "one run of the whole scenario" and the question is "does
  the scenario work".
- `group_rule="mean"`: one trial per group with the group's mean pass rate as a
  fractional success; interval over groups. Right when cases within a group
  are exchangeable checks of the same thing.

`Decision` gains `n_groups` and `n_cases`, and the report shows both: the
headline over groups, the per-case count as context. Per-case rows render as
*"passed in k of N groups"* when a case id recurs across groups (the
`demo_card.py` k-of-N collapse, generalised: strip the group prefix, count).

`required_n` should answer in groups when a rule is set, because that is the
unit the user can add more of.

### Why upstream

Anyone evaluating an agent by re-running the same scenario N times - which is
the natural thing to do with a non-deterministic model - hits this on day one
and either does not notice (and over-claims) or rebuilds the aggregation by
hand. The library already refuses to over-claim on small n; it should refuse
to over-claim on correlated n too.

---

## 3. Smaller: the scorer should be able to see the case

`Scorer` is `(output, expected) -> Score`. Two segments in the same run needed
different scorers (a board-match F1 for ordinary segments, a recall-only "was
the bait drawn" check for the board at a dismiss), and the only way to route
was to match `expected` by object identity back to its case
(`demo_card.py`, `scorer_with_case`). Ugly and fragile.

Proposal: accept either signature. If the scorer takes three parameters, pass
the `Case` as the third. No breaking change; existing two-argument scorers
keep working.

```python
Scorer = Callable[[Any, Any], Score] | Callable[[Any, Any, Case], Score]
```

---

## Also observed, not proposed as code

- The report card is gonogo's human-facing output, and the one people asked
  to see. `report.html()` is a page; the card is one screen: grade, headline,
  interval, a few rows, a footer that says what it was graded against.
  Worth a `report.card()` renderer eventually. `live-diagrammer/eval/charles01-screen.html`
  is the shape.
- Framing moved the number 70 points before the model changed (20% -> 90% ->
  33% -> 57%, same system). gonogo cannot fix that, but the docs could say
  plainly that the case definition is where most evaluation error lives, and
  that a small committed regression set (ours: ten runs of one recording)
  is what catches a grader defect before it becomes a claim.
