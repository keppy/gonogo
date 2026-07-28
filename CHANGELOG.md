# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/keppy/gonogo/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/keppy/gonogo/releases/tag/v0.1.0
