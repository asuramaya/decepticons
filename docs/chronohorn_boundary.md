# Chronohorn Boundary

The boundary between `decepticons` (the shared Python kernel) and `chronohorn`
(the main runtime descendant). The kernel stays mechanism-first; chronohorn
owns training, replay, packing, evaluation, and fleet execution.

## Repository Roles

### `decepticons`

Owns:

- reusable substrate and memory primitives
- routing, gating, and controller mechanisms
- readout interfaces and shared feature views
- export helpers for descendant systems
- family-neutral contracts reusable by more than one descendant
- causal-bank config validation and mechanism-level knobs (`substrate_mode`,
  `state_impl`, `num_heads`, `readout_bands`, `fast_lr_mult`, …)

Does not own:

- project-specific runtime policy
- model-family-specific training recipes
- packed artifact formats
- legality or validity bundle policy
- benchmark claims or frontier reporting

### `chronohorn`

Owns:

- training and mutation search
- Python-side model family implementations
- export from kernel objects into chronohorn artifacts
- Rust replay and causal runtime
- offline table compilation and packing
- full-val evaluation
- fleet and orchestration surfaces
- family-owned scan regimes and promotion policy
- VRAM-tier placement and scheduler hints

Chronohorn may use decepticons mechanisms, but should not re-declare them as a
second shared kernel.

## Mechanism vs Policy

### Mechanisms (belong in decepticons)

- substrate generation
- controller summaries
- gates and routing primitives
- memory views and scoring helpers
- readout blocks reusable across descendants
- export-friendly shared tensor layout helpers
- learned-state substrates: head-factored `scan`, `retention`, `gated_retention`

### Policy (belongs in chronohorn)

- model-family choice
- training schedule
- readout selection for a specific family
- packed-table selection and byte budget
- held-out evaluation policy
- frontier reporting
- fleet placement and job scheduling
- cheap-lane ablation ordering, scale/context-survival promotion, replication

### Policy that does not belong in decepticons

- audit trust levels
- validity bundle packaging
- submission/report formatting
- legality claim promotion
- runtime claim arbitration

## Dependency Direction (Hard Rule)

```
decepticons ──→ numpy (only)
chronohorn  ──→ decepticons
chronohorn  ──→ numpy, torch/mlx, sentencepiece, …
```

**decepticons never imports chronohorn.** Not in source, not in tests, not in
examples. This firewall is enforced by an AST scan in
[`tests/test_dependency_firewall.py`](../tests/test_dependency_firewall.py).

If a decepticons test needs to verify a primitive works in a training loop, it
builds its own 10-line loop. It does not import a chronohorn trainer.

## Practical Migration Rule

When a mechanism appears in both repos, ask:

1. Can it be described without mentioning a specific model family?
2. Can another descendant reuse it unchanged?
3. Is the generalized API simpler than the duplicate descendant code?

Only if all three answer yes should the mechanism move into decepticons.

Concrete example:

- adding `gated_retention` as a reusable causal-bank substrate belongs in `decepticons`
- emitting `cb-substrate-s8-gret-h4-10k` / `cb-substrate-s12-gret-h4-10k` and
  deciding when they promote belongs in `chronohorn`

## Non-Goals

This boundary is not trying to:

- move chronohorn training into the kernel
- make Rust a training framework
- collapse chronohorn and decepticons into one repo
- fold audit/ledger tooling into the kernel
- encode leaderboard or submission policy in the kernel
- make export ABI changes without versioning

The kernel stays reusable. The descendant stays operationally complete.
