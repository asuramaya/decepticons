# Chronohorn Kernel Boundary

This document defines the boundary between `open-predictive-coder` as the shared Python kernel and `Chronohorn` as the main runtime descendant system.

The goal is to keep the kernel mechanism-first and keep Chronohorn responsible for training, replay, packing, evaluation, and fleet execution.

## Repository Roles

### `open-predictive-coder`

`open-predictive-coder` remains the shared Python kernel and core.

It owns:

- reusable substrate and memory primitives
- routing, gating, and controller mechanisms
- readout interfaces and shared feature views
- export helpers for descendant systems
- family-neutral contracts that can be reused by more than one descendant

It does not own:

- project-specific runtime policy
- model-family-specific training recipes
- packed artifact formats
- legality or validity bundle policy
- benchmark claims or frontier reporting

### `Chronohorn`

`Chronohorn` is the main runtime descendant system.

It owns:

- training and mutation search
- Python-side model family implementations
- export from kernel objects into Chronohorn artifacts
- Rust replay and causal runtime
- offline table compilation and packing
- full-val evaluation
- fleet and orchestration surfaces

Chronohorn may use OPC mechanisms, but it should not re-declare them as a second shared kernel.

## Proposed Export ABI

The integration point should be a stable export ABI, not a loose checkpoint convention.

The export ABI should carry:

- `model_family_id`
- `model_variant`
- `kernel_version`
- `tokenizer_id`
- `data_root_id`
- deterministic substrate config
- learned tensor payloads
- optional packed-memory payloads
- artifact provenance
- replay metadata
- exporter version

The export ABI should separate:

- deterministic code-derived state
- learned trainable state
- optional packed residual state
- audit metadata

If a field can be regenerated exactly from code and config, it should not be treated as learned payload.

## Mechanism vs Policy

The kernel should only carry mechanisms.

### Mechanisms that belong in OPC

- substrate generation
- controller summaries
- gates and routing primitives
- memory views and scoring helpers
- readout blocks that are reusable across descendants
- export-friendly shared tensor layout helpers

### Policy that belongs in Chronohorn

- model-family choice
- training schedule
- readout selection for a specific family
- packed-table selection and byte budget
- held-out evaluation policy
- frontier reporting
- fleet placement and job scheduling

### Policy that does not belong in OPC

- audit trust levels
- validity bundle packaging
- submission/report formatting
- legality claim promotion
- runtime claim arbitration

## Explicit Non-Goals

This boundary is not trying to make OPC into a full training system.

It is not trying to:

- move Chronohorn training into the kernel
- make Rust a training framework
- collapse Chronohorn and OPC into one repo
- fold audit/ledger tooling into the kernel
- encode leaderboard or submission policy in the kernel
- make export ABI changes without versioning

The point is to keep the kernel reusable and the descendant system operationally complete.

## Practical Migration Rule

When a mechanism appears in both OPC and Chronohorn, ask:

1. can it be described without mentioning a specific model family?
2. can another descendant reuse it unchanged?
3. is the generalized API simpler than the duplicate descendant code?

Only if the answer is yes should the mechanism move into OPC.

Everything else should stay in Chronohorn as descendant policy.

## Working Rule For The Promoted Causal-Bank Line

After the promoted causal-bank line is expressed through Chronohorn:

- `opc` defines the reusable Python kernel
- `chronohorn` defines the concrete causal-bank implementations plus Rust runtime/compiler
- historical codenames stop being repo boundaries and become archive lineage only

That keeps the system legible:

- kernel first
- descendant second
- export ABI between them
- no duplicated authority
