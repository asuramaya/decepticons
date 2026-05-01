# Changelog

All notable changes to `decepticons` will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(major.minor.patch). Pre-1.0 minor bumps may include breaking changes.

## [Unreleased]

### Added
- `decepticons.__version__` derived from package metadata.
- `scripts/release.sh` — one-command version bump + tag + push.
- Release workflow now creates a GitHub Release with notes from `CHANGELOG.md`
  and attaches the wheel + sdist as release assets.
- Release workflow smoke-installs the built wheel and verifies the installed
  `__version__` matches the pushed tag before publishing.

### Changed
- Aligned canonical phrasing across README, docs, site, and package metadata:
  unified five-pillar list (`substrates, memory, gating, routing, readouts`),
  unified ecosystem labels (`kernel · runtime · forensics`), unified promotion
  rule wording, unified causality-test pitch.
- `pyproject.toml` description now matches the README/site tagline voice.
- `decepticons.__init__` module docstring rewritten to match the canonical
  one-line description.
- `decepticons --help` description now describes what the CLI does instead of
  the generic "Decepticons CLI" placeholder.

## [0.1.0] — 2026-04-30

First public release. Alpha-stage research kernel.

### Kernel

- Reusable substrate primitives: `EchoStateSubstrate`, `DelayLineSubstrate`,
  `LinearMemorySubstrate`, `OscillatoryMemorySubstrate`, `MixedMemorySubstrate`,
  `HierarchicalSubstrate`, plus the `factories` config-driven dispatch.
- Memory primitives: `ExactContextMemory`, `NgramMemory`,
  `StatisticalBackoffMemory`, `OnlineCausalMemory`, plus `ExactContextCache`
  and `StatisticalBackoffCache` view layers.
- Control surfaces: `ControllerSummary`, `PredictiveController`,
  `LearnedSegmenter`, `PathwayGateController`, `SummaryRouter`,
  `HormoneModulator`, `PredictiveSurpriseController`.
- Feature views: `ByteLatentFeatureView`, `HierarchicalFeatureView`,
  `LinearMemoryFeatureView`, `SampledMultiscaleReadout`,
  `ProbabilityDiagnostics`, `bridge_feature_arrays`,
  `BidirectionalContextProbe`.
- Readouts and experts: `RidgeReadout`, `FrozenReadoutExpert`.
- Adapters: `CausalPredictiveAdapter`, `NoncausalReconstructiveAdapter`,
  `OracleAnalysisAdapter`, `BridgeExportAdapter`, `TeacherExportAdapter`,
  `ByteLatentPredictiveCoder`.
- Causal-bank family: `CausalBankConfig`, `build_linear_bank`,
  `validate_config`, `scale_config`, `learnable_substrate_keys`. Substrate
  modes `frozen`, `learnable_decays`, `learnable_mixing`, `learned_recurrence`,
  `gated_retention`. Optional memory attachment, stacked blocks, selective
  scan, banded readouts, byte-to-patch encoding, fast/slow hemispheres,
  polynomial expansion, training noise, adaptive regularization.
- Runtime: `CausalTrace`, `FitReport`, `evaluate_rollout`, `score_next_step`,
  `evaluate_dataset`, `evaluate_rollout_curve`, `evaluate_transfer_probe`,
  `TrainModeConfig`, `ArtifactAccounting`, artifact audits.

### Backends

- `decepticons.models.causal_bank_torch.CausalBankModel` — PyTorch
  implementation with frozen substrate, selective scan augment, banded readout.
  Optional installs: `pip install -e ".[torch]"`.
- `decepticons.models.causal_bank_mlx.CausalBankModel` — MLX equivalent.
  Optional installs: `pip install -e ".[metal]"`.

### CLI

- `decepticons fit` — single-command fit + sample over a UTF-8 corpus.

### Verification

- `tests/test_causality.py` verifies every substrate mode is causal under
  perturbation. CI fails if any future-leak is detected.
- `tests/test_dependency_firewall.py` AST-scans the kernel to ensure
  decepticons never imports its descendants.

[Unreleased]: https://github.com/asuramaya/decepticons/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/asuramaya/decepticons/releases/tag/v0.1.0
