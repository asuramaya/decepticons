# support_export

`support_export` is a bridge-style consumer over the shared cache and teacher/export surfaces.

It uses:

- [`ExactContextCache`](../../../../src/decepticons/memory_cache.py)
- [`StatisticalBackoffCache`](../../../../src/decepticons/memory_cache.py)
- [`TeacherExportAdapter`](../../../../src/decepticons/teacher_export.py)

The local policy is simple:

- the teacher stream is the active exact-context cache prediction
- the student stream is the mixed statistical-backoff prediction
- support and active-order summaries are exported alongside the paired probability streams

That keeps the bridge contract generic while proving it can consume shared cache records directly.
