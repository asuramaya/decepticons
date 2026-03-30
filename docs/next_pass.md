# Next Pass

This document turns the current state into the next concrete implementation pass.

## Goal

Harden the new `src/`-level causal adapter and prove that it is a real shared contract rather than a renamed Conker
branch.

## Why This Is The Next Pass

The repo now has:

- enough substrate families
- enough controller primitives
- enough memory and view surfaces
- enough runtime scaffolding
- enough project descendants to test the boundary
- a first `src/`-level `causal_predictive` adapter

That changes the problem.

The next step is not to invent another core abstraction immediately. The next step is to pressure-test the new causal
contract across descendants and keep the kernel readable while doing it.

## Workstreams

### 1. Thin Conker around the causal contract

Make the Conker replicas consume the shared causal layer wherever the behavior is actually common.

What should stay local:

- mixer policy
- residual-repair policy
- legality framing
- benchmark and frontier claims

Acceptance criteria:

- less duplicate causal boilerplate in `examples/projects/conker_*`
- Conker policy still clearly lives in project code

### 2. Harden causal runtime and accounting

The causal adapter now exists, so the next reusable layer is better reporting and accounting around it.

Focus on:

- replay-safe accounting
- metadata tagging
- lightweight causal report wrappers
- evaluation hooks that do not import descendant policy

Acceptance criteria:

- causal reports are useful outside one example
- artifact/replay helpers stay policy-free

### 3. Add a second consumer of the causal layer

The causal adapter is not stable until something besides the Conker line pushes on it.

Best candidates:

- a bridge/export-shaped descendant
- a noncausal analysis or replay descendant that still reuses part of the causal surface

Acceptance criteria:

- at least one more descendant reuses some of the same causal contract
- any new promotion into `src/` is justified by repeated use

### 4. Keep refining `carving_machine_like`, but only in project space

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

1. thin Conker around the shared causal adapter
2. harden causal runtime and accounting
3. add a second consumer of the causal layer
4. only then consider wider causal abstractions

## Non-Goals For The Next Pass

These should wait:

- full optimizer/training harness extraction
- full legality framework
- bridge/export finalization
- noncausal replay economics in `src/`
- preset stabilization

## Definition Of Done

The pass is done when:

- the causal adapter clearly reads as a shared `src/` contract
- Conker descendants look thinner around that contract
- at least one more descendant pushes on the same causal surface
- docs explain the boundary without requiring project history to decode the codebase
