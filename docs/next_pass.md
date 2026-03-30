# Next Pass

This document turns the current state into the next concrete implementation pass.

## Goal

Build the first real `src/`-level causal adapter while continuing to improve readability and keep project policy out of
the kernel.

## Why This Is The Next Pass

The repo already has:

- enough substrate families
- enough controller primitives
- enough memory and view surfaces
- enough runtime scaffolding
- enough project descendants to test the boundary

What it does not yet have is a first-class causal adapter in `src/` that uses those pieces together.

That is the most important missing layer.

## Workstreams

### 1. `causal_predictive` adapter in `src/`

Build the first reusable causal adapter from existing pieces:

- exact-context memory
- frozen experts
- train/eval surfaces
- artifact accounting

The first version can be modest. It does not need to be the final Conker.

Acceptance criteria:

- lives in `src/`
- has `fit`, `score`, `predict_proba`, and basic accounting
- uses `ArtifactAccounting` or adjacent runtime hooks
- is simpler than the current Conker project replicas

### 2. Wire artifact accounting into causal paths

Use the new artifact/runtime slice for:

- replay span reporting
- artifact byte accounting
- metadata tagging for evaluation mode

Acceptance criteria:

- at least one causal path reports meaningful artifact accounting
- the API remains policy-free

### 3. Keep refining `carving_machine_like`, but only in project space

The ancestor example is now good enough to act as a boundary test.

The next work there is:

- stronger predictor behavior
- more faithful routed/modulated variants
- better checkpoint reporting

None of that should widen `src/` unless another descendant needs the same mechanism.

### 4. Add one bridge-shaped or noncausal descendant after the causal adapter lands

The current oracle example is enough for now.

After the causal adapter exists, the next descendant should probably be one of:

- a bridge/export example
- a noncausal reconstructive example

The purpose is to keep checking whether new kernel abstractions are actually shared.

### 5. Keep tightening repo readability

Every pass should also improve orientation:

- architecture docs stay current
- examples index stays current
- tests remain grouped by purpose
- root exports stay legible by category

## Recommended Order

1. create `src`-level causal adapter
2. use artifact accounting in that adapter
3. thin the Conker replicas around it
4. only then add another downstream family

## Non-Goals For The Next Pass

These should wait:

- full optimizer/training harness extraction
- full legality framework
- bridge/export finalization
- noncausal replay economics in `src`
- preset stabilization

## Definition Of Done

The pass is done when:

- the repo has one real causal adapter in `src/`
- the Conker examples clearly read as descendants of that adapter or as stress tests around it
- docs explain the boundary without requiring project history to decode the codebase
