# Decepticons — Developer Guide

Non-transformer primitives kernel. O(1) attention is deception.

## What This Is

Shared kernel for predictive descendants. Provides backend-neutral model mechanisms — substrates, memory, gating, routing, readouts — that downstream systems (chronohorn, heinrich) combine into trained models.

## What This Is NOT

Not a training framework. Not an experiment tracker. Not a fleet manager. Those belong in chronohorn.

## Dependency Firewall

**decepticons never imports chronohorn.** Not in source, not in tests, not in examples.

This is enforced by `tests/test_dependency_firewall.py` (AST scan). If you add an import from chronohorn, the test fails. No exceptions.

```
decepticons → numpy (only hard dep)
chronohorn  → decepticons
```

If a decepticons test needs to verify something works in a training loop, it builds its own minimal loop. It does not import a chronohorn trainer.

## Key Modules

- **`causal_bank.py`** — `CausalBankConfig` (43 fields), `build_linear_bank()`, `validate_config()`, `scale_config()`. Backend-neutral deterministic substrate construction.
- **`models/`** — PyTorch and MLX backend implementations of `CausalBankModel`, readout layers, weight init primitives.
- **`substrates.py`** — Recurrent, delay, linear-memory, oscillatory, mixed, hierarchical substrate primitives.
- **`online_memory.py`** — `OnlineCausalMemory` runtime n-gram accumulator.
- **`control.py`, `gating.py`, `routing.py`** — Controller summaries, pathway gates, summary routing.

## How to Run Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -v
```

The workstream integration tests (`test_workstream_integration.py`) require `torch`. They skip gracefully without it.

## Code Conventions

- `numpy>=1.26` is the only hard dependency. torch and mlx are optional.
- Backend-specific code goes in `models/`. Backend-neutral code goes in the top-level package.
- Config dataclasses are frozen (`@dataclass(frozen=True)`).
- Deterministic substrate construction uses `CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED = 42`.
- No hardcoded family names from downstream — if you need downstream behavior, use dependency injection (e.g., `set_ngram_table()`).

## Adding a New Primitive

1. Create `src/decepticons/your_primitive.py`
2. Add tests in `tests/test_your_primitive.py`
3. If it's backend-specific (torch/mlx), put it in `models/`
4. If it could be described without mentioning a specific model family, it belongs here
5. If it can't, it belongs in chronohorn

## Current Families Using This Kernel

- **causal-bank** — SSM substrates, oscillatory modes, memory attachment, patch encoding
- **polyhash** — Does NOT use decepticons (independent O(1) hash tables + gated scan)
