# Inbox triage: three gates on the same cases

The worked example behind the Nov 3, 2026 AI Tinkerers Seattle talk, "Measuring what
your agent didn't do". An inbox-triage agent makes one decision per incoming message:
`reply`, `ask_human`, `wait` or `no_reply`. Every decision is a tool call, so every
message leaves a row, and gonogo grades each one as an action (`Case.expected_action`),
including the ones where the agent sends nothing.

Three gates make that decision on the same 870 real messages from four Enron inboxes (two
lawyers, a government-affairs lead and an operations lead):

- **Claude** (`claude-opus-5-5`, effort low) choosing one of four tools;
- **Jev** (TypeSafe `jev-1.14.0`) with the four options passed at runtime;
- **an encoder** (ModernBERT-base fine-tuned with [thomas](https://github.com/keppy/thomas)),
  run locally. The model is public:
  [keppy/inbox-triage-encoder](https://huggingface.co/keppy/inbox-triage-encoder).

Everything here replays offline with no API key and no email text:

```bash
pip install gonogo-eval
cd examples/inbox_triage
python run_gate.py --gate jev --split holdout --mix enriched --offline   # re-render a report
python holdout_check.py --gate claude-opus-5-5                           # the frozen-threshold test
python cost_rule.py --ratio 4 --split holdout                            # decide on cost, not pass rate
```

## Read this before quoting a number

The labels are a **reference policy written by two model families**, not decisions made
by a person: DeepSeek (`deepseek-v4-pro`) and Z.ai's GLM (`GLM-5.3`), neither of which is
a gate under test. Both read `guidelines.md`; a message is a case only where they agree.
Their per-message labels are in `data/labels/`. A person's blind labels on a held-back
sample (43 eval messages, not shipped until they are read back) will give the agreement
above chance that says how far to trust the reference.

Each gate's review threshold is chosen on eval, frozen, and tested once on the holdout.
A threshold that misses there is reported as a miss, never re-picked.

## What it found

Pass rate (95% interval), holdout: 329 messages (ask_human 159, no_reply 146, wait 14,
reply 10). Eval: 541 enriched (rare actions added from the same inboxes) and 344 natural
(the inbox as it arrives, mostly no_reply).

| gate | eval enriched | eval natural | holdout | calibration error (holdout) |
| --- | --- | --- | --- | --- |
| Claude | 91.7% [89.1, 93.7] | 91.6% [88.2, 94.1] | 90.6% [86.9, 93.3] | 0.12 |
| Jev | 85.4% [82.2, 88.1] | 82.6% [78.2, 86.2] | 86.0% [81.9, 89.4] | 0.10 |
| encoder | 80.4% [76.9, 83.5] | 86.0% [82.0, 89.3] | 78.1% [73.3, 82.2] | 0.06 |

**The actions table shows what the pass rate hides.** Holdout:

| gate | owner needed, left silent (ask_human → no_reply/wait) | replied where the owner had to answer | sent to the owner unneeded (no_reply/wait → ask_human) |
| --- | --- | --- | --- |
| Claude | 5 | 1 | 18 |
| Jev | 3 | 2 | 32 |
| encoder | 42 | 10 | 10 |

On the natural eval mix the encoder and Jev tie (86.0% vs 82.6%, no detectable
difference) with opposite failures: the encoder leaves owner messages silent, Jev
over-escalates.

**No gate's eval threshold held on the holdout** at the 90% precision target: Claude's
lower bound lands at 88.1%, Jev's at 86.9%, the encoder's at 85.5% (`holdout_check.py`).

**Decide on cost.** Say a message the owner needed and didn't get costs 4, and an
unneeded escalation costs 1. Then escalate whenever P(ask_human) ≥ 1/(1+4) = 0.2: the
threshold comes from the costs, and nothing is tuned on the data. Holdout, r = 4:

| gate | pass rate | owner missed | cost per 100 messages |
| --- | --- | --- | --- |
| Claude | 76.9% [72.1, 81.1] | 1 / 159 | 24.0 |
| Jev | 76.9% [72.1, 81.1] | 0 / 159 | 23.1 |
| encoder | 79.9% [75.3, 83.9] | 33 / 159 | 50.2 |

Pass rate separates no pair (McNemar p 0.34–1.0). Cost does: the encoder costs 26 more
per 100 messages than Claude [+12.8, +40.7] (paired bootstrap), and Claude and Jev tie
(−0.9 [−5.5, +3.3]), at $6.71 vs $0.05 for the whole run. The gate with the best pass
rate is the most expensive one, because its mistakes are the costly kind. (Claude reports
only its chosen option's confidence, so its P(ask_human) is an upper bound.)

Other measurements in `data/reports/`: Jev's own `confidence` field is a spread
statistic, not a probability (calibration error 0.16 against 0.08 for the probability it
puts on its chosen option), and earlier encoder versions (`encoder-inbox-triage-v1..v3`,
`-la` = logit-adjusted) are there for comparison.

## Files

- `agent.py`: the harness. The "before" log records only what it sent; the "after" trace
  makes `wait`, `ask_human` and `no_reply` tool calls, so every message leaves a row. After
  `fetch_text.py`, `agent.py --offline --legacy` vs `agent.py --offline` replays both from
  Claude's cache for free: 20 messages in, 1 row out, against 20 in, 20 out. (Reasons are
  blank in the shipped cache.)
- `gates.py`: the three gates and their cache; `guidelines.md`: the policy every gate
  and labeler reads (the cache file names carry its hash).
- `run_gate.py`: grade a gate with gonogo and save the report; `holdout_check.py`,
  `cost_rule.py`: the two decision rules above.
- `data/cases/`: ids, expected action, sampling and the Enron maildir path; no text.
  `data/cache/`: each gate's action, confidence and (Jev, encoder) per-option
  probabilities; reasons and drafts are left out, since they quote the emails.
- `fetch_text.py`: rebuilds the email text from CMU's public Enron release
  (`enron_mail_20150507.tar.gz`, https://www.cs.cmu.edu/~enron/) into
  `data/messages.jsonl`, which live runs (paid) and `agent.py` read. It is the FERC public
  release; CMU asks users to respect the privacy of the people in it.
