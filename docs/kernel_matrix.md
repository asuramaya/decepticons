# Kernel Matrix

This matrix lays out the kernel-first build order for `open-predictive-coder` using three sources of guidance:

- the `carving_machine` thesis that the reusable center is the literal upstream core from which the downstream systems
  evolve
- the generalized downstream pattern language in [`downstream_patterns.md`](./downstream_patterns.md)
- the research anchors in [`related_work.md`](./related_work.md)

The point is not to preserve every historical branch name. The point is to extract the smallest reusable kernel from
`carving_machine` that can still support causal, noncausal, oracle, bridge, and byte-latent downstream systems.

Use [`lineage.md`](./lineage.md) for the attribution rule behind the upstream workspace paths named below.

## Source Anchors

### `carving_machine`

- `README.md#L8` in the upstream `carving_machine` workspace:
  the working thesis is a rich substrate plus a learned side-channel and readout
- `README.md#L21` in the upstream `carving_machine` workspace:
  the live fixed-substrate thread is hierarchical and multi-timescale
- `README.md#L29` in the upstream `carving_machine` workspace:
  the harness is meant to compare substrate designs, not just masks
- `ORIGIN.md#L15` in the upstream `carving_machine` workspace:
  the original design language was subtraction, predictive coding, and a reservoir plus controller plus readout stack

### Generalized downstream framing

- [`downstream_patterns.md`](./downstream_patterns.md):
  the kernel should be organized as substrate dynamics, side-channels, views, readouts, and runtime
- [`downstream_patterns.md`](./downstream_patterns.md):
  causal systems need legality, replay, and packed-artifact discipline
- [`downstream_patterns.md`](./downstream_patterns.md):
  noncausal systems need field selectors, replay, and side-data accounting
- [`downstream_patterns.md`](./downstream_patterns.md):
  oracle systems need analysis-only context and attack accounting
- [`downstream_patterns.md`](./downstream_patterns.md):
  bridge systems need explicit runtime contracts

### Research anchors

- [`related_work.md`](./related_work.md):
  predictive coding implies local residuals and error-carrying views
- [`related_work.md`](./related_work.md):
  reservoir computing anchors the fixed-substrate side of the kernel
- [`related_work.md`](./related_work.md):
  rate-distortion and information bottleneck thinking justify explicit latent commits
- [`related_work.md`](./related_work.md):
  sequence predictive coding and continual modeling motivate the runtime and rollout surfaces
- [`related_work.md`](./related_work.md):
  recurrent refinement and byte-level latent compression motivate the byte-latent adapter layer

## Matrix

| Kernel Area | Why It Exists | `carving_machine` source | Research anchor | Current status |
| --- | --- | --- | --- | --- |
| `substrates.echo_state` | fixed recurrent baseline substrate | `carving_machine/reservoir.py#L85` | Jaeger 2001, Maass et al. 2002 | Implemented |
| `substrates.delay` | deterministic fading-memory control line | `carving_machine/models.py#L384` | fading-memory / liquid-state intuition | Implemented |
| `substrates.linear_memory` | frozen linear multiscale decay-bank memory | `Conker-2/3` reconstruction from primitives | short/medium-horizon linear memory scaffold | Implemented |
| `substrates.mixed_memory` | recurrent plus delay hybrid | `carving_machine/models.py#L484`, `carving_machine/models.py#L609` | sequence memory and continual modeling | Implemented |
| `substrates.hierarchical` | fast/mid/slow multi-timescale substrate | `carving_machine/models.py#L224` | predictive coding hierarchies, multi-timescale memory | Implemented |
| `factories.substrates` | config-driven substrate construction and adapter dispatch | `carving_machine` harness principle: compare substrate designs | engineering bridge from research kernel to adapters | Implemented |
| `controllers.summary` | generic summary contract shared by gates and routing | `carving_machine/models.py#L224`, `carving_machine/models.py#L1129` | control-side summary views over substrate state | Implemented |
| `controllers.predictive` | latent commit, prediction, surprise, residual paths | `carving_machine/models.py#L224`, `carving_machine/models.py#L609` | Rao and Ballard 1999, Friston 2005 | Partial: generic predictive/surprise primitive implemented |
| `memory.exact_context` | causal exact-history experts over exact1/exact2/exact3 style contexts | `conker` archive docs and early exact-count branches | count-based language modeling, causal support-aware correction | Implemented |
| `controllers.gating` | reusable fast-to-mid and mid-to-slow pathway gates | `carving_machine/models.py#L224` | adaptive control over multiscale substrate paths | Implemented |
| `controllers.routing` | causal substrate/path selection over branch summaries | `carving_machine/models.py#L1129` | adaptive control over substrate views | Implemented |
| `controllers.modulation` | hormone/modulation paths | `carving_machine/models.py#L1354` | side-channel modulation over substrate | Implemented as primitive |
| `views.hierarchical` | pooled and predictive views over fast/mid/slow banks | `carving_machine/models.py#L224` | predictive coding hierarchies and surprise-style residuals | Implemented |
| `views.sampled_readout` | deterministic sampled bands over multiscale state | `ablations.py`, `v6.py` | sampled multiscale readout over fixed state | Implemented |
| `views.byte_latent` | residual + patch-summary + latent feature construction | current library | Rao and Ballard 1999, BLT 2025 | Implemented |
| `views.linear_memory` | reusable features over decay-bank memory state | `Conker-2/3` reconstruction from primitives | short/medium-horizon linear summaries | Implemented |
| `readouts.closed_form` | simple trained readout over frozen state | current library | Jaeger 2001 | Implemented |
| `experts.frozen_readout` | frozen substrate plus feature-function expert wrapper | `Conker-1/2/3` reconstruction from primitives | mixture-of-experts over fixed dynamical state | Implemented |
| `runtime.trace_reporting` | sequence traces, reports, accounting | current library | Shannon 1948 | Implemented |
| `runtime.eval_light` | next-step and single-rollout scoring over current adapters | `carving_machine/training.py#L64` | sequence predictive coding and continual modeling | Implemented |
| `runtime.train_eval` | weighted dataset eval, checkpointed rollout curves, and transfer probes | `carving_machine/training.py#L54`, `carving_machine/training.py#L111`, `carving_machine/experiments.py#L493` | sequence predictive coding and continual modeling | Implemented |
| `runtime.train_modes` | detached vs through-state semantics, sparse slow updates, rollout checkpoints | `bptt_test.py` | BPTT versus detached-state runtime regimes | Implemented |
| `runtime.artifacts_audits` | legality, replay, artifact-boundary surfaces | downstream docs, especially Conker family | compression/accounting discipline | Partial: metadata, replay spans, and accounting primitives implemented |
| `adapters.byte_latent` | first concrete downstream adapter | current library | BLT 2025, UT 2018 | Implemented |
| `adapters.causal_predictive` | causal predictive/compressive runtime systems | `conker`-family docs | sequence predictive coding | Not started |
| `adapters.noncausal_reconstructive` | document-field replay systems | `blinx`-family docs | reconstruction and side-data economics | Not started |
| `adapters.oracle_analysis` | bidirectional structure analysis | `blinx oracle` docs | predictive coding as analysis, not runtime | Not started |
| `adapters.bridge_export` | offline teacher to causal export layer | `giddy-up` docs | explicit boundary discipline | Not started |
| `presets` | reproducible named bundles over primitives | `carving_machine/catalog.py#L1` | engineering convenience, not theory | Not started |
| `examples.reference_projects` | thin project-shaped models over the kernel, used for smoke/dev loops | `carving_machine`, `conker`, `blinx oracle`, and `brelt` docs as downstream shapes | engineering bridge from primitives to dev/test targets | Implemented |

## Priority

### P0: finish the remaining kernel spine

- integrate the new modulation and predictive-surprise primitives into more than one descendant before widening them further
- richer exact-context expert families beyond the current exact-history memory core
- full training harness and optimizer-facing loop extraction beyond the current eval/transfer scaffold
- tighter hierarchy/controller integration beyond pooled predictive views and the compact predictive-surprise primitive

### P1: finish the causal kernel contract

- rollout evaluation
- replay-safe reporting
- packed-artifact and legality hooks
- memory-first expert/cache interfaces

### P2: grow sideways into noncausal and bridge work

- noncausal field selectors and replay accounting
- oracle analysis helpers
- bridge/export schemas

### P3: only then stabilize presets

- named preset bundles copied from `carving_machine`
- compatibility aliases for historical branch names if still useful

## Current Read

The kernel is no longer just an echo-state toy. It now has:

- fixed recurrent substrate
- delay substrate
- linear memory substrate
- mixed-memory substrate
- hierarchical multiscale substrate
- substrate factory and `substrate_kind` dispatch
- controller-summary contract
- predictive controller primitive plus compact predictive/surprise primitive
- pathway-gate primitive
- summary router primitive
- hormone modulation primitive
- exact-context memory and support-weighted blending primitives
- feature-view primitives for both byte-latent and hierarchical state
- deterministic sampled multiscale readout
- reusable linear-memory feature view
- reporting runtime surface
- lightweight rollout and next-step evaluation
- weighted dataset eval, rollout curves, and transfer probes
- frozen readout expert primitive
- byte-latent adapter
- example-project smoke surfaces for `carving_machine`-like and early `conker`-like builds
- `Conker-1/2/3` style replica projects built from primitives

The line to preserve is:

- if a mechanism is required by multiple descendants, it belongs here
- if a choice is really a downstream task policy, legality rule, artifact boundary, or evaluation claim, it does not
  belong here

That is why `carving_machine` remains the main source anchor even when the first concrete adapter in `src/` is still
`byte-latent`.

That is enough to justify continuing kernel-first rather than inventing downstream codenames too early.
