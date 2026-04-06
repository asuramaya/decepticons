# Decepticons — Developer Guide

Non-transformer primitives kernel. O(1) attention is deception.

## What This Is

Shared kernel for predictive descendants. Provides backend-neutral model mechanisms — substrates, memory, gating, routing, readouts — that downstream systems (chronohorn, heinrich) combine into trained models.

## What This Is NOT

Not a training framework. Not an experiment tracker. Not a fleet manager. Those belong in chronohorn. Not a transformer forensics tool. That's heinrich.

## Ecosystem

```
decepticons  → kernel (this repo)
chronohorn   → runtime (training, tracking, fleet, MCP)
heinrich     → forensics (model geometry, activation traces)
```

## Dependency Firewall

**decepticons never imports chronohorn.** Not in source, not in tests, not in examples.

Enforced by:
- `tests/test_dependency_firewall.py` — AST scan, catches any chronohorn import
- `tests/test_dependency_firewall.py::test_no_stale_opc_references` — catches leftover `opc`/`open-predictive-coder` references

```
decepticons → numpy (only hard dep)
chronohorn  → decepticons
```

If a decepticons test needs to verify something works in a training loop, it builds its own minimal loop. It does not import a chronohorn trainer.

## Key Modules

**Config & Substrate:**
- **`causal_bank.py`** — `CausalBankConfig` (43+ fields), `build_linear_bank()`, `validate_config()`, `scale_config()`, `learnable_substrate_keys()`. Backend-neutral deterministic substrate construction.

**Model Backends (require torch or mlx):**
- **`models/causal_bank_torch.py`** — PyTorch `CausalBankModel`. Frozen substrate, selective scan augment, banded readout.
- **`models/causal_bank_mlx.py`** — MLX backend equivalent.
- **`models/readouts_torch.py`** — MLP, GRU, `RoutedSquaredReLUReadout` (4 experts, pointwise, causal).
- **`models/common.py`** — Weight init primitives (`_xavier_uniform`, `_embedding_uniform`).
- **`models/diagnostics.py`** — Model introspection: mode liveness, timescale utilization, readout selectivity, interpreted findings.

**Substrate Primitives:**
- **`substrates.py`** — Recurrent, delay, linear-memory, oscillatory, mixed, hierarchical.
- **`online_memory.py`** — `OnlineCausalMemory` runtime n-gram accumulator.

**Control & Routing:**
- **`control.py`, `gating.py`, `routing.py`** — Controller summaries, pathway gates, summary routing.

## Causality

All substrate modes verified causal by `tests/test_causality.py`. The test feeds identical sequences up to position t, different after t. If logits at position t differ, causality is violated.

Verified modes: `frozen`, `learnable_mixing`, `learnable_decays`, selective scan augment (`state_dim > 0`), `readout_bands`, routed experts.

**History:** The conker research line lost months to an accidentally-unfrozen `causal_mask` that broke causality silently. This test prevents recurrence. Run it after any architecture change.

## How to Run Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -v
```

For model backend tests: `pip install -e ".[torch]"` or `pip install -e ".[metal]"`.

## Code Conventions

- `numpy>=1.26` is the only hard dependency for the kernel. `models/` requires torch or mlx.
- Backend-specific code goes in `models/`. Backend-neutral code goes in the top-level package.
- Config dataclasses are frozen (`@dataclass(frozen=True)`).
- Deterministic substrate construction uses `CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED = 42`.
- No hardcoded downstream names — if you need downstream behavior, use dependency injection (e.g., `model.set_ngram_table()`).
- Causality test must pass for all substrate modes.
- Diagnostics interpret their own measurements — findings, not just numbers.

## Substrate Modes

```
frozen             nothing learns — pure reservoir, fixed dynamics
learnable_decays   decay rates learn — how fast modes forget
learnable_mixing   input projection learns — what the bank sees
learned_recurrence everything learns — full learnable state machine
```

`readout_bands` (N>1) splits modes by timescale with separate gradient per band.
`state_dim` (>0) adds selective scan augment — content-dependent signal on top of the frozen bank.

## Adding a New Primitive

1. Create `src/decepticons/your_primitive.py`
2. Add tests in `tests/test_your_primitive.py`
3. If it's backend-specific (torch/mlx), put it in `models/`
4. If it involves sequence processing, add a causality test
5. If it could be described without mentioning a specific model family, it belongs here
6. If it can't, it belongs in chronohorn

## Current Families

- **causal-bank** — Uses decepticons. Frozen linear substrate, local conv, MLP/expert readout.
- **polyhash** — Does NOT use decepticons. Independent O(1) hash tables + gated scan.
