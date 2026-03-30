# open predictive coder

`open-predictive-coder` is a Python library for extracting a reusable predictive substrate kernel from a broader
upstream experiment family. That family evolves into causal, noncausal, bridge, oracle, and byte-latent descendants,
so the point of this repo is to draw the boundary once: keep the shared substrate, control, memory, view, readout,
and runtime primitives in one place, and let downstream systems specialize from there.

It is designed as a kernel extraction scaffold, not a benchmark claim. The current implementation includes
multiscale substrates, predictive and exact-context memory primitives, control-side summaries, gates and routing,
compressed latent commits, learned boundary and local patch-latent blocks, and simple trained readouts exposed as
composable pieces.

The repo is explicitly anchored in both literature and the current library landscape rather than vague inspiration.
See [`docs/related_work.md`](./docs/related_work.md) for canonical references,
[`docs/landscape.md`](./docs/landscape.md) for nearby libraries and ecosystem gaps, and
[`docs/downstream_patterns.md`](./docs/downstream_patterns.md) for the generalized causal/noncausal/oracle/bridge/byte
pattern language extracted from related descendants. Use [`docs/lineage.md`](./docs/lineage.md) for source lineage,
attribution, and the rule this repo follows for non-vendored references. The current kernel extraction roadmap lives in
[`docs/kernel_matrix.md`](./docs/kernel_matrix.md), the package/code map is in
[`docs/architecture.md`](./docs/architecture.md), the concrete next implementation pass is in
[`docs/next_pass.md`](./docs/next_pass.md), the current live sibling-frontier extraction read is in
[`docs/frontier_pass.md`](./docs/frontier_pass.md), and the next controller extraction boundary is outlined in
[`docs/control_surface.md`](./docs/control_surface.md).

## Start Here

If you want to understand the repo quickly:

1. read [`docs/architecture.md`](./docs/architecture.md)
2. skim [`docs/kernel_matrix.md`](./docs/kernel_matrix.md)
3. read [`docs/frontier_pass.md`](./docs/frontier_pass.md)
4. read [`examples/README.md`](./examples/README.md)
5. use [`tests/README.md`](./tests/README.md) to find the verification surface for the area you care about

## Why This Exists

There are already predictive-coding libraries and there are already reservoir-computing libraries. What is missing is
the reusable kernel behind this workspace's project family:

- a substrate-first experimental core that can evolve into several downstream systems
- predictive coding for byte or sequence data rather than mostly image or tabular PCN demos
- fixed, delayed, mixed-memory, and hierarchical substrates instead of one model family
- control, routing, and memory primitives that can be reused across causal, noncausal, and bridge work
- compression-aware sequence experiments without dragging in a giant training stack

So the goal here is not "another general predictive coding framework" and not "a toy byte model." The goal is a
readable extraction of a reusable predictive substrate core, positioned so causal, noncausal, oracle, bridge, and
byte-latent systems can all grow from the same primitive layer.

## Line Drawing

The intended line is:

- `open-predictive-coder`: the extracted kernel of reusable primitives
- upstream workspace experiments: source lineage and attribution live in [`docs/lineage.md`](./docs/lineage.md)
- downstream descendants and future descendants: systems that add policy, runtime contracts, and
  task-specific claims

This repo is standalone. The example descendants under [`examples/projects`](./examples/projects) are reconstructions
built in this repo from the extracted primitives; they are not imported or copied from sibling repositories.
The first causal adapter should be treated as the shared contract that causal descendants thin around, not as a
single-descendant branch.

What belongs in the kernel:

- substrate dynamics
- predictive and exact-context memory primitives
- views and sampled summaries
- control-side summaries, gates, routing, and later modulation
- readout interfaces
- runtime and evaluation surfaces

What does not belong in the kernel:

- one project's legality policy
- one project's artifact format
- one project's benchmark story
- one project's noncausal privileges
- one project's bridge contract

That is the reason to build hierarchical and family-shaped example projects on top of the kernel: they let the repo test
whether the line is drawn correctly before more primitives are promoted into `src/`.

## Downstream Pattern Types

These are the kinds of downstream work this library is meant to support. They are not source adaptations for this
repo, and the public framing is by idea rather than by codename.

- `causal predictive/compressive`:
  strict prefix-only runtime systems with legality audits, artifact-boundary accounting, and memory-first sequence
  correction. See the attribution notes in [`docs/lineage.md`](./docs/lineage.md).
- `noncausal reconstructive`:
  whole-document removal and replay systems that treat the document as a field and make side-data economics central.
  See the attribution notes in [`docs/lineage.md`](./docs/lineage.md).
- `oracle analysis`:
  bidirectional analysis passes that estimate structural determinism, small candidate-set size, and future-context
  uplift, but do not make direct runtime codec claims. See the sibling descendant notes in
  [`docs/lineage.md`](./docs/lineage.md).
- `bridge export`:
  boundary layers that turn offline oracle findings into strictly causal exported features or replay artifacts.
  See the attribution notes in [`docs/lineage.md`](./docs/lineage.md).
- `byte-latent`:
  byte-visible systems with learned or heuristic patches, shorter internal latent streams, and recurrent latent
  refinement.
  This repo is the first reference implementation for that pattern, and public lineage notes are tracked in
  [`docs/lineage.md`](./docs/lineage.md).

Those patterns illustrate the problem family:

- causal runtime and artifact-boundary discipline
- noncausal field reconstruction and side-data-aware replay
- oracle analysis separated from legal runtime claims
- offline bridge features exported into causal consumers
- byte-patch latent modeling over a shorter internal sequence

The extracted adapters in `src/` today are the byte-latent reference path plus shared `causal_predictive`,
`oracle_analysis`, and `bridge_export` contracts. The kernel also carries example-project surfaces for an
ancestor-style hierarchical path and the more specific causal, bridge, and oracle descendants that sit on top of
those shared layers.

## Design Thesis

The core bet is simple:

- sequence models do not always need to reason at a single flat resolution
- a latent summary should earn its keep by compressing local activity into a shorter internal stream
- predictive coding should surface local residuals rather than only raw hidden states
- a frozen recurrent substrate plus a lightweight trained readout is a useful open reference point

This implementation synthesizes ideas from:

- predictive coding and error-carrying hierarchies
- reservoir computing and echo-state style fixed recurrent dynamics
- rate-distortion and information bottleneck thinking
- patch- or span-based byte modeling
- recurrent refinement over a shorter latent stream

## What The Library Includes

- `ByteLatentPredictiveCoder`: the extracted byte-latent adapter
- `CausalPredictiveAdapter`: the first extracted causal exact-context plus auxiliary-expert adapter
- `OpenPredictiveCoder`: compatibility alias for `ByteLatentPredictiveCoder`
- `EchoStateSubstrate`: frozen recurrent substrate
- `DelayLineSubstrate`: deterministic delay-memory substrate
- `LinearMemorySubstrate`: frozen linear decay-bank memory substrate
- `OscillatoryMemorySubstrate`: frozen exponential plus damped-oscillatory mode-bank substrate
- `MixedMemorySubstrate`: concatenated recurrent plus delay-memory substrate
- `HierarchicalSubstrate`: multiscale fast/mid/slow recurrent substrate
- `LinearMemoryFeatureView`: reusable view over linear decay-bank state
- adaptive or fixed byte patching
- `LatentCommitter`: patch commit and recurrent global-memory update primitive
- `LearnedBoundaryScorer` and `LearnedSegmenter`: reusable learned boundary probability and target-rate patching primitives
- `LocalByteEncoder`: reusable local byte encoder with fittable output projection
- `PatchPooler`: mean/last/mix patch pooling primitive
- `GlobalLocalBridge`: learned bridge from global and latent state back to local features
- `PredictiveController`: idea-based alias for the same controller surface
- `PredictiveSurpriseController`: reusable prediction/residual/surprise primitive
- `HierarchicalFeatureView`: pooled, predictive, and surprise-style views over fast/mid/slow state
- `ByteLatentFeatureView`: feature view over residuals, patch summaries, and latents
- `ControllerSummary` and `ControllerSummaryBuilder`: generic controller-summary contract for control-side primitives
- `PathwayGateController`: reusable fast-to-mid and mid-to-slow gate primitive
- `SummaryRouter`: equal/static/projection routing over branch summaries
- `HormoneModulator`: reusable hormone projection and bounded modulation primitive
- `SampledMultiscaleReadout`: deterministic banded sampling over multiscale state
- `ExactContextMemory`: causal exact-history count memory over 1/2/3-step contexts
- `NgramMemory`: smoothed unigram/bigram/trigram statistical memory primitive
- `StatisticalBackoffMemory`: fitted global mixture over order-wise n-gram priors with prefix-time fallback semantics
- `BridgeExportAdapter`: generic export/report surface over paired probability streams
- `bridge_feature_arrays`: causal proxy features derived from probability arrays
- `BidirectionalContextProbe`: noncausal context determinism and leave-one-out probe
- `select_scored_spans` and `replay_spans_from_scores`: reusable score-to-span selection helpers
- `SupportWeightedMixer`: support-biased blending over base and exact-context experts
- `ArtifactMetadata`, `ReplaySpan`, and `ArtifactAccounting`: causal artifact/replay/accounting primitives
- `CausalTrace`, `CausalSequenceReport`, and `CausalFitReport`: causal reporting wrappers over runtime/accounting surfaces
- `RidgeReadout`: closed-form readout fitting
- `FrozenReadoutExpert`: frozen substrate plus feature-function expert primitive
- sparse reservoir builders with Erdos-Renyi and small-world topologies
- substrate factories and `substrate_kind`-based adapter selection
- `SequenceTrace`, `SequenceReport`, and `FitReport`: runtime-level sequence accounting surfaces
- `score_next_step` and `evaluate_rollout`: lightweight scoring and rollout evaluation helpers
- `evaluate_dataset`, `evaluate_rollout_curve`, and `evaluate_transfer_probe`: weighted dataset eval, checkpointed rollout curves, and transfer probes
- `TrainModeConfig`: detached vs through-state semantics, sparse slow updates, and rollout checkpoint policy
- bits-per-byte scoring and greedy/sampled generation
- a small CLI for fitting on a text file and sampling completions

The package is intentionally small and readable. It is meant to be extended.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 examples/quickstart.py
```

Or from Python:

```python
from open_predictive_coder import ByteCodec, ByteLatentPredictiveCoder

text = "predictive coding likes repeated structure.\n" * 64
model = ByteLatentPredictiveCoder()
fit_report = model.fit(text)

prompt = ByteCodec.encode_text("predictive ")
sample = model.generate(prompt, steps=40, greedy=True)

print(fit_report.train_bits_per_byte)
print(ByteCodec.decode_text(sample))
```

## CLI

Train on a text file and print a sample:

```bash
opc fit --input ./corpus.txt --prompt "predictive " --generate 80
```

## Repo Layout

- `src/open_predictive_coder/`: reusable kernel primitives plus extracted causal, oracle, bridge, and byte-latent adapters
- `docs/`: architecture, roadmap, literature, landscape, and extraction notes
- `examples/`: quickstart, descendant-shaped example projects, and development tools
- `tests/`: kernel, runtime, boundary, and project-descendant test suites

For the detailed map, use:

- [`docs/architecture.md`](./docs/architecture.md)
- [`examples/README.md`](./examples/README.md)
- [`tests/README.md`](./tests/README.md)

## Scope

This is a reference implementation for research and prototyping. It is not a frontier language model,
not a production compression stack, and not yet a faithful reproduction of the full upstream experiment family. It
also does not yet implement the full causal/noncausal/oracle/bridge surface described in the docs, and its runtime
layer still stops short of the full upstream training/runtime harness and legality tooling. The point is to expose
the shared kernel cleanly enough that multiple descendants can be built on top of it without forcing their policies
back into the primitive layer. In the current replication round, the repeated promotions from project code into the
kernel were `LinearMemorySubstrate`, `LinearMemoryFeatureView`, `FrozenReadoutExpert`, `PredictiveSurpriseController`,
`HormoneModulator`, `SampledMultiscaleReadout`, `TrainModeConfig`, the first `ArtifactMetadata` / `ReplaySpan` /
`ArtifactAccounting` runtime slice, the first shared `CausalPredictiveAdapter` plus `OracleAnalysisAdapter` plus
`BridgeExportAdapter`, the first learned patch-latent kernel blocks (`LearnedSegmenter`, `LocalByteEncoder`,
`PatchPooler`, `GlobalLocalBridge`), and the next statistical/kernel additions (`OscillatoryMemorySubstrate`,
`NgramMemory`, `StatisticalBackoffMemory`, `bridge_feature_arrays`, `ProbabilityDiagnostics`, `BidirectionalContextProbe`,
`select_scored_spans`, `replay_spans_from_scores`); descendant mixer policies, noncausal replay economics,
rate-distortion objectives, and quantization/export policy remain deliberately project-local.
