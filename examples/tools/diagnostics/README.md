# Diagnostics

This package contains project-only analysis helpers for the old `look.py`, `look2.py`, and `silence_test.py` lineage,
plus thin example-flow adapters that can run the existing `carving_machine_like` and Conker replicas without touching
`src/`.

It stays out of `src/` by design. Use it for:

- mask, gate, and surprise snapshots
- ablation comparisons and two-factor decompositions
- simple info-flow summaries over numeric arrays
- example-flow reports for `carving_machine_like`, `conker_early_like`, and `conker_1/2/3_like`

The surface is intentionally lightweight and text-first.

## Import

```python
from examples.tools.diagnostics import (
    capture_snapshot,
    summarize_binary_mask,
    compare_ablation,
    diagnose_carving_machine_like,
)
```

## Typical use

Capture a snapshot:

```python
snapshot = capture_snapshot(
    1000,
    mask=mask_array,
    gate=gate_array,
    surprise=surprise_array,
)
```

Summarize a mask:

```python
summary = summarize_binary_mask(mask_array, name="mask")
```

Compare an ablation:

```python
result = compare_ablation("baseline", 1.23, "no_mask", 1.31)
```

Run an example-flow report:

```python
report = diagnose_carving_machine_like()
print(report.format_lines()[0])
```
