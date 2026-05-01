# Teacher Export

This example is a bridge descendant for export labels and attack-aware reporting.

It stays project-local on purpose:

- it synthesizes teacher and student probability streams locally
- it exports teacher labels through the shared bridge adapter
- it measures attack drift with a local token mutation pass and bidirectional-context probing
- it does not widen `src/` with descendant-specific policy

Kernel pieces reused here:

- [`TeacherExportAdapter`](../../../../src/decepticons/teacher_export.py)
- [`BridgeExportAdapter`](../../../../src/decepticons/bridge_export.py)
- [`probability_diagnostics`](../../../../src/decepticons/probability_diagnostics.py)
- [`BidirectionalContextProbe`](../../../../src/decepticons/bidirectional_context.py)

Entry points:

- [`probe.py`](./probe.py)
- [`smoke.py`](./smoke.py)

Run from the repo root:

```bash
PYTHONPATH=src python3 examples/projects/bridge/teacher_export/probe.py
PYTHONPATH=src python3 examples/projects/bridge/teacher_export/smoke.py
```
