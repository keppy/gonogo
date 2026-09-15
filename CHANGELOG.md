# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0.post1] - 2026-09-15

Documentation only; no code changes.

### Fixed

- README described the pre-0.1.1 calibration behavior ("refuses to recommend a
  threshold"); it now matches the code, which keeps the operating point and
  flags the scale.
- README documents `compare()`/`outcomes()` and per-case outcomes in
  `to_dict()`, which shipped in 0.2.0 without a README section or changelog
  entry; the 0.2.0 entry below now includes them.

## [0.2.0] - 2026-09-15

Three changes from using gonogo as the harness for a live meeting-diagramming
agent at a hackathon — each a case where the library had no seam for something
the evaluation needed, and the workaround lived in the caller — plus paired
comparison, from running a Claude agent head-to-head against the Banking77
baseline.

### Added

- **Label provenance.** `validate_judge` takes `label_source` (`"human"`,
  `"model"`, `"structural"`, or free text) and `label_source_note`;
  `JudgeValidation` carries both and every rendering names the source. The
  word *human* appears in output only for `label_source="human"`. Structural
  labels are gated on leniency alone -- a judge that passes a case a
  structural check failed is rubber-stamping whatever its kappa -- and the
  result says it rules out rubber-stamping rather than validating the judge.
  Motivation: labels from a second model, passed as `human_passes`, were
  reported in exactly the words a human validation would use.
- **Correlated cases.** `Case.group` marks cases that share one draw of the
  system under test; `evaluate(group_rule="all")` makes one trial per group,
  passing only if every case in it passed, with the interval over groups.
  `Decision.n_groups` and `.unit` say which unit the numbers are in, and
  `decide` takes `unit` for its wording. Reports show group counts with the
  case count as context, and a "per case, across groups" table for case ids
  that recur across groups (*recovery: 0 of 10 groups*). `Case.from_jsonl`
  reads and `to_jsonl` writes a `group` key. A mix of grouped and ungrouped
  cases is a `ValueError`. Motivation: ten replays x four checks read as
  forty trials, `[60%, 86%]`, and hid a check that failed in every run
  inside a 75% average; over runs it was 0 of 10.
- **Scorers may take the case.** A scorer declaring a third positional
  parameter is called as `(output, expected, case)`; two-argument scorers are
  unchanged. `scoring.CaseScorer` names the protocol. Motivation: routing two
  scorers within one run required matching `expected` back to its case by
  object identity.
- **Paired comparison.** `compare(report_a, report_b)` runs McNemar's test on
  the case ids two runs share: cases both agents got right carry no
  information about which is better, so they are excluded rather than padding
  the denominator. Exact binomial up to 1,000 disagreements, chi-square with
  continuity correction above. `Comparison.significant`, `.summary()`, and an
  interval on the paired difference. `compare` accepts `Report` objects or
  `Report.to_dict()` payloads — `to_dict()` now carries per-case
  id/passed/score/confidence/detail so today's run can be paired against a
  JSON file from last week. `outcomes()` extracts `{case_id: passed}` from
  either form. New example: `examples/banking77_claude_agent.py`, a real
  Claude agent against the TF-IDF baseline on the same case set.

### Changed

- `validate_judge`'s second parameter is now `reference_passes`. `human_passes`
  still works as a keyword and raises `DeprecationWarning`; passing both is a
  `TypeError`.
- Wording in `JudgeValidation.reason` for sizes below `min_n` no longer says
  "hand-labelled", since the labels may not be.

## [0.1.1] - 2026-07-28

### Fixed

- `judge_agreement` returned kappa 1.0 when judge and human labels were both
  constant and identical (e.g. a pass-everything judge validated on a subset
  the human also fully passed). Kappa is undefined there, and `validate_judge`
  now rejects the judge instead of reporting it USABLE with kappa 1.00.
- An agent returning an out-of-range confidence (e.g. `("answer", 1.2)`) raised
  and aborted the entire evaluation. It is now recorded as that case's failure,
  matching the documented contract for agent errors.
- `set_f1` iterated a bare string character by character, which could produce
  false passes (`"abc"` vs `"cab"`). A string now counts as a single label.
- The `judge` scorer clamped scores above the scale to the top of the scale,
  turning a malformed reply like "10" on a 1-5 scale into a pass. Out-of-scale
  scores now fail with a detail message.
- `decide` reported a calibration error even when the agent never stated a
  confidence, reading as miscalibration the agent never claimed. It is now NaN
  (and omitted from reports) unless a real confidence signal exists.
- `fields` compared numbers as strings when `tolerance` was 0, failing an
  expected `1` against an output of `1.0`; numbers now always compare
  numerically. A key missing from the output is now wrong even when the
  expected value is falsy.
- `Report.markdown()` lowercased the rest of the decision reason via
  `str.capitalize`; `Report.to_dict()` now includes `precision_high` alongside
  `precision_low`.
- An agent returning `("answer", True)` was silently read as confidence 1.0;
  bool no longer counts as a confidence value.

### Added

- `py.typed` marker so type checkers pick up the package's annotations.
- This changelog.

### Changed

- The package version is now sourced from `gonogo/__init__.py` at build time
  instead of being duplicated in `pyproject.toml`.

## [0.1.0] - 2026-07-28

### Added

- Initial release: Wilson intervals, risk-coverage curves, calibration
  measurement, judge validation, the five-verdict `decide()` layer,
  `evaluate()` runner, and Markdown/HTML score reports.

[0.2.0.post1]: https://github.com/keppy/gonogo/compare/v0.2.0...v0.2.0.post1
[0.2.0]: https://github.com/keppy/gonogo/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/keppy/gonogo/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/keppy/gonogo/releases/tag/v0.1.0
