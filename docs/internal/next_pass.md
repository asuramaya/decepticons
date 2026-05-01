# Next Pass

This document turns the current state into the next concrete implementation pass.

## Goal

Use the new noncausal, bridge, and teacher-export descendants to pressure-test the shared contracts that just landed
in `src/`, while keeping the kernel readable and generic.

The live sibling-frontier read that informs this pass is tracked in [`frontier_pass.md`](./frontier_pass.md).

## Why This Is The Next Pass

The repo now has:

- enough substrate families
- enough controller primitives
- enough memory and view surfaces
- enough runtime scaffolding
- enough project descendants to test the boundary
- a first `src/`-level `causal_predictive` adapter
- a first `src/`-level `oracle_analysis` adapter
- a first `src/`-level `bridge_export` adapter
- a first `src/`-level `noncausal_reconstructive` adapter
- a first `src/`-level `teacher_export` contract over paired probability sources
- a first generic `artifacts_audits` helper layer over artifact accounting
- a first family-neutral probability diagnostics seam over scored distributions
- a first learned patch-latent kernel slice for segmentation, local encoding, pooling, and bridging
- the next statistical/kernel slice for oscillatory memory, n-gram memory, bridge features, and bidirectional context
- a first reusable scored-span selection primitive
- a first causal packed-memory controller descendant built from current primitives
- a first noncausal field-reconstruction descendant built from current primitives
- a second and third bridge-shaped descendants beyond the first bridge export surface

That changes the problem.

The next step is not to invent another abstract substrate mechanism immediately. The next step is to pressure-test the
new shared adapters and learned patch-latent blocks across descendants and keep the kernel readable while doing it.

That now also applies to the new statistical/kernel additions: they need consumers before they justify wider
abstraction or backend work.

## Workstreams

### 1. Thin causal descendants around the causal contract

Make the causal descendants consume the shared causal layer wherever the behavior is actually common.

What should stay local:

- mixer policy
- residual-repair policy
- legality framing
- benchmark and frontier claims

Acceptance criteria:

- less duplicate causal boilerplate in `examples/projects/causal_*`
- descendant policy still clearly lives in project code

### 2. Harden causal runtime and accounting

The causal adapter now exists, so the next reusable layer is better reporting and accounting around it.

Focus on:

- replay-safe accounting
- metadata tagging
- lightweight causal report wrappers
- evaluation hooks that do not import descendant policy
- artifact-boundary helpers that stay generic and composable instead of encoding legality policy

Acceptance criteria:

- causal reports are useful outside one example
- artifact/replay helpers stay policy-free

### 3. Decide the first noncausal adapter boundary

The shared adapters are more credible now, but the next missing family is still a real noncausal adapter in `src/`.

Use the current descendants:

- `examples/projects/noncausal/field_reconstruction`
- `examples/projects/bridge/proxy_features`
- `examples/projects/bridge/feature_export`
- `examples/projects/bridge/agreement_export`
- the new bridge-side teacher-export contract, now that it exists as a shared paired-export surface

Acceptance criteria:

- the noncausal adapter boundary is specific enough to name without descendant policy leaking into it
- any new promotion into `src/` is justified by repeated use across at least two descendants

### 4. Keep refining `hierarchical_predictive`, but only in project space

The ancestor example remains a boundary test, not a reason to widen `src/` recklessly.

Likely next work there:

- stronger predictor behavior
- more faithful routed/modulated variants
- better checkpoint reporting

Acceptance criteria:

- ancestor-specific policy stays outside the kernel unless it repeats elsewhere

### 5. Keep tightening repo readability

Every pass should also improve orientation:

- architecture docs stay current
- examples index stays current
- tests remain grouped by purpose
- root exports stay legible by category

## Recommended Order

1. thin causal and oracle examples around the shared adapters
2. harden runtime and accounting
3. pressure-test `noncausal_reconstructive`, `teacher_export`, and `artifacts_audits` across more than one consumer
4. only then consider wider family abstractions
5. keep the current live `Conker` / `BLINX` / `Giddy-Up` frontier read synced into the descendant rebuild plan

## Non-Goals For The Next Pass

These should wait:

- full optimizer/training harness extraction
- full legality framework
- noncausal replay economics beyond the minimal shared contract
- patch-latent rate-distortion, QAT, or second-stage downsampling in `src/` before a second consumer asks for them
- preset stabilization
- higher-order causal program/controller policy in `src/`

## Definition Of Done

The pass is done when:

- the shared causal, oracle, and bridge adapters clearly read as `src/` contracts
- the next noncausal contract is either clearly extracted or clearly deferred
- the corresponding examples look thinner around those contracts
- at least one more descendant pushes on one of the same shared surfaces
- docs explain the boundary without requiring project history to decode the codebase
