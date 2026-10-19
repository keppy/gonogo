# A docs site for gonogo, shaped like Instructor's

> Status: proposed** (2026-10-17). Nothing here is built. Section 9 is the
> order to build it in.

The README is 280 lines and it is five documents interleaved: the pitch, the
five-line moment, a concepts reference, a cookbook, and an ecosystem map. Each
is decent on its own. Together they mean a reader looking for "how do I compare
two agents" walks through the kappa derivation to get there, and a reader who
wants the kappa derivation can't link to it.

Instructor's docs at 1.0 (March 2024, the takeoff point) separated those same
five jobs into five layers, and the benefit of the library was visible on the
first page without adopting anything. This proposal maps gonogo onto that
shape. It is mostly splitting, not writing: roughly 70% of the pages below
already exist as README sections.

Reference: `jxnl/instructor` at tag `1.0.0`, `mkdocs.yml` nav and
`docs/index.md`.

---

## 1. What Instructor's docs did, layer by layer

| Layer | Job | Instructor 1.0 |
| --- | --- | --- |
| index | The five-line moment, no argument yet | `client = instructor.from_openai(OpenAI())`; `response_model=UserDetail`; typed object back |
| why | The argument, kept off the index page | Short, bulleted |
| Concepts | One page per noun, one screen each, runnable snippet | Models, Fields, Validators, Retrying, Patching, Partial, Union, Enums... ~20 pages |
| Cookbook | One page per situation the reader already has | Classification, Citations, PII, Knowledge Graphs, Moderation, Query Planning... titled by the reader's problem |
| Blog | Distribution, hosted inside the docs site | "Good LLM validation is just good validation", "RAG is more than embeddings", `course.md` |
| Hub | Copy-paste by provider | Ollama, Together, Groq, Anyscale... |
| Tutorials | Five notebooks, a free mini-course | 1-introduction through 4-validation |
| API reference | Free via mkdocstrings on a small typed library | |

The property that mattered: the library wrapped something the reader already
had (an OpenAI client) and the payoff was visible in the return type. No new
data to supply, no base class to adopt, before/after in one diff.

## 2. gonogo's equivalent of the patch moment

The README's five-liner needs `invoices.jsonl`, which the reader doesn't have.
`python examples/invoice_extraction.py` needs a clone. Neither is a ten-second
"oh".

The ten-second version is the README's own first line, and it needs no cases:

```python
>>> from gonogo import decide
>>> decide([(0.5, True)] * 47 + [(0.5, False)] * 3, target=0.90)
INSUFFICIENT EVIDENCE
observed 94.0% on 50 cases is above the 90% target, but the 95% interval reaches down to 83.8%
```

That runs on 0.3.0 today (verified), but the signature hides it behind a list
comprehension. Two small additions make it the index page:

- **`decide.from_counts(passed, n, target=...)`** (or `Decision.from_counts`):
  a count form of `decide`, no confidence, no cases. ~20 lines.
- **`gonogo 47/50 --target 0.9`** as a console script: same thing from the
  shell, prints the verdict, the interval, and `required_n`. ~30 lines, stdlib
  `argparse`, no new dependency.

The count form must print `Decision.needed_n` on the same line as the
verdict. For the input above it is 204, and "47/50 looks like 94%; you'd need
~204 cases to claim 90%" is the whole pitch of the INSUFFICIENT EVIDENCE
verdict in one sentence. Today it's a field you have to know to look for.

With the count form in place, page one makes the argument with a number the
reader already has in their head. `Case` arrives on page two, when they want
per-case results.

## 3. Nav tree

```yaml
nav:
  - Introduction:
    - gonogo: index.md                      # count form, verdict, interval, 3 links
    - Why not a point estimate?: why.md     # README "The idea", "What it tells you"
    - Install: install.md                   # gonogo-eval vs gonogo on PyPI
    - Philosophy: philosophy.md             # README "Non-goals" + __init__ docstring
  - Concepts:
    - Cases: concepts/cases.md
    - Predictions and abstention: concepts/predictions.md
    - Expected actions: concepts/actions.md            # from feat/expected-action
    - Scorers: concepts/scorers.md
    - The five verdicts: concepts/verdicts.md
    - Wilson intervals: concepts/intervals.md
    - Insufficient evidence and required_n: concepts/required-n.md
    - Operating points: concepts/operating-points.md
    - Calibration vs discrimination: concepts/calibration.md
    - Groups (correlated cases): concepts/groups.md
    - Pairing and McNemar: concepts/pairing.md
    - Kappa: concepts/kappa.md
    - Label provenance: concepts/provenance.md
    - Reports: concepts/reports.md
  - Cookbook:
    - Cookbook: cookbook/index.md
    - I have 47 out of 50: cookbook/counts.md
    - My agent sometimes shouldn't answer: cookbook/actions.md
    - I'm using an LLM as the judge: cookbook/judge.md
    - I ran 10 scenarios with 4 checks each: cookbook/groups.md
    - I swapped the model and the number went up: cookbook/compare.md
    - My agent reports confidence: cookbook/confidence.md
    - I only have 30 cases: cookbook/small-n.md
    - Three routers, one table: cookbook/three-routers.md
    - Is the vendor's confidence calibrated?: cookbook/vendor-confidence.md
    - Invoice extraction (deterministic, no API key): cookbook/invoices.md
    - Banking77 (real data, stdlib baseline): cookbook/banking77.md
  - Integrations:                            # gonogo's Hub: not providers, consumers
    - pytest: integrations/pytest.md
    - GitHub Actions PR comment: integrations/github-actions.md
    - Hermes plugin: integrations/hermes.md
    - thomas (training harness, CONTRACT v1): integrations/thomas.md
    - verifiers / Prime: integrations/verifiers.md
    - evalroute: integrations/evalroute.md
  - Tutorials:                               # workshop A, modules 1-3, free cut
    - 1. Make every decision visible: tutorials/1-decisions.md
    - 2. One case shape, three consumers: tutorials/2-case-shape.md
    - 3. Honest verdicts: tutorials/3-verdicts.md
  - Blog:
    - blog/index.md
  - API reference: reference/               # mkdocstrings, gonogo.*
```

## 4. README section → page mapping

| README section (lines, current main) | Destination | Work |
| --- | --- | --- |
| 3-7 hook | `index.md` top | rewrite around the count form |
| 9-15 video + results | `cookbook/banking77.md`, linked from index | move |
| 17-23 Install | `install.md`, one line stays on index | move |
| 25-53 Try it (invoice) | `cookbook/invoices.md` | move |
| 55-85 One loop, three libraries | `integrations/` index + `philosophy.md` | move, split |
| 87-95 The idea | `why.md` | move |
| 97-113 Usage | `index.md` second block, `concepts/cases.md` | move |
| 115-127 What it tells you | `concepts/verdicts.md` | move |
| 129-145 Why confidence matters | `concepts/operating-points.md`, `cookbook/confidence.md` | split: curve table → cookbook, lower-bound argument → concept |
| 147-169 A real measurement | `cookbook/banking77.md`, `concepts/calibration.md` | split: run → cookbook, calibration-vs-discrimination paragraph → concept |
| 171-207 On LLM-as-judge | `cookbook/judge.md`, `concepts/kappa.md`, `concepts/provenance.md` | split three ways; kappa derivation verbatim |
| 209-227 Correlated cases | `concepts/groups.md`, `cookbook/groups.md` | split |
| 229-251 Comparing two agents | `concepts/pairing.md`, `cookbook/compare.md` | split |
| 253-255 Reports | `concepts/reports.md` | move, expand with one rendered example each |
| 257-265 Works with | `integrations/thomas.md` | move |
| 267-277 Non-goals | `philosophy.md` | move |
| PR #2 "When the right move is not to answer" | `concepts/actions.md`, `cookbook/actions.md` | split when merged |

Pages with no README source, to write new: `cookbook/counts.md`,
`cookbook/small-n.md`, `cookbook/three-routers.md` (two of three columns
exist), `cookbook/vendor-confidence.md`, every `integrations/*` except thomas, all three
tutorials.

The README after the split: hook with the count form, install, the five-line
`evaluate` call, the verdict table, the ecosystem diagram, and a "docs →" link.
Under 80 lines.

## 5. gonogo is the referee, not a contestant

Three ways to build the System 1 gate an agent harness needs (does this
message need a reply / a human / nothing yet): a frontier LLM with structured
output (the Instructor `Literal[...]` pattern), a hosted zero-shot decision
model that takes the choice array at runtime (TypeSafe's Jev, `jev-1.14.0`,
$0.042/Mtok input as of Oct 2026), or an owned encoder fine-tuned through
thomas for about $1. gonogo does not take a side. It is the layer that tells
you which one to ship, at what threshold, on your cases.

That positioning goes on `why.md` in one paragraph, and it earns two cookbook
pages that no other eval library's docs can show, because none of them do
paired comparison at pilot sizes:

- **`cookbook/three-routers.md`: three routers, one table.** Same 250
  cases, three agents, `compare()` pairwise, McNemar. The Banking77 TF-IDF vs
  Claude run is already this page with two columns; the encoder report
  (`examples/banking77_encoder.json`) is the third. A Jev column is one
  script.
- **`cookbook/vendor-confidence.md`: is the vendor's confidence calibrated?**
  Jev's docs say "calibrated probabilities." Softmax over logits is a
  distribution; calibration is a measured property of that distribution
  against outcomes, and the two are routinely conflated (the pasted
  analysis that prompted this section called softmax output "perfectly
  calibrated"). `reliability_table` and ECE on the vendor's stated
  probabilities, before and after temperature scaling. Banking77 is the
  existing proof: TF-IDF ranks almost perfectly at ECE 0.42; the ModernBERT
  encoder reaches ECE 0.03 only *after* scaling. This is also a blog post and
  a talk slide.

`concepts/calibration.md` gets the one-sentence version: a softmax is not a
calibration; an ECE on held-out outcomes is.

The **graduation ladder** (course module 4) is the same idea as a sequence:
stage 0, LLM + structured output, labels the first traces (those labels are
`label_source="model"` and `validate_judge` already handles that); stage 1, a
zero-shot decision model while the schema is still moving; stage 2, the owned
encoder once `compare(stage1, stage2)` shows no detectable regression and the
operating point holds on a fresh holdout. Every rung is a gonogo verdict. It is
a tutorial page once the inbox-triage demo exists; until then it is a
paragraph on `cookbook/three-routers.md`.

Out of scope for gonogo: building the gate. A dynamic-schema local router
(bi-encoder prune, cross-encoder rerank, ModernBERT for the 8k context) is a
thomas recipe, and an Instructor-DX `from_encoder()` wrapper that returns a
typed choice with a probability is thomas's index page someday, not this
site's.

## 6. Blog

"Your agent eval is lying to you" (Jul 2026, yokozunasan), the Banking77 canary
writeup (Sep 21, keppylab), and the insufficient-evidence draft all currently
send their traffic somewhere other than the library. Instructor's blog lived
at `python.useinstructor.com/blog/`. Cross-post all three into `docs/blog/posts/`
with the docs site as canonical, or at minimum link back from the originals.

A cookbook page is a blog post with the narrative stripped. Writing them in
that order (cookbook first, post second) makes the next post cheap.

## 7. What doesn't transfer

- **Hub-as-providers.** gonogo has no provider surface. Integrations are the
  equivalent: the places a `Report` is produced or consumed.
- **Download velocity as the proof.** Instructor rode a pain that every LLM
  team hit in the same quarter. "Is my agent making the right decision,
  including not acting" is narrower. Docs can't manufacture that; the
  expected-action PR and traces → cases are what make gonogo the obvious
  answer when someone goes looking.
- **Volume.** Instructor shipped ~20 blog posts in its takeoff year. There are
  three here. The structure should make the fourth cheap, not pretend the
  twenty exist.
- **Notebooks.** The tutorials are markdown with runnable blocks, not `.ipynb`.
  Less to maintain and they render in the repo.

## 8. Tooling

- `mkdocs-material` + `mkdocstrings[python]` + the material blog plugin.
  Dev-only extras in `pyproject.toml` (`[project.optional-dependencies] docs`).
- API reference is close to free: 1,772 typed lines, `py.typed`, docstrings
  already on the public surface.
- GitHub Pages from a `docs.yml` workflow on push to main.
- `[project.urls] Documentation` in `pyproject.toml` so PyPI links to it.

## 9. Order

1. Merge `feat/expected-action` (PR #2). The course doc lists it first and
   `concepts/actions.md` depends on it.
2. `decide.from_counts` + the `gonogo` console script, both printing
   `needed_n`. Bump to 0.4.0; this is the only library change.
3. mkdocs scaffold, Pages workflow, API reference. README shrinks.
4. Split the README into Concepts and Cookbook per the table in section 4.
5. Cross-post the three blog posts.
6. Tutorials 1-3, after the Oct 24-25 inbox-triage demo exists, since that
   demo is the worked example for all three. Cut at exactly the free/paid line
   in `lifetracker/course/evals-course-oct12.md`: the tutorials show what the
   gate found, never how to build the case set from traces.

Steps 2-5 are a weekend. None of them wait on the Nov 3 answer or the Ergoly
week.
