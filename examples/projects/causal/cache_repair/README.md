# cache_repair

`cache_repair` is a thin causal descendant built directly on the shared memory-cache layer.

It uses:

- [`StatisticalBackoffCache`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/src/open_predictive_coder/memory_cache.py)
- [`ExactContextCache`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/src/open_predictive_coder/memory_cache.py)
- [`probability_diagnostics`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/src/open_predictive_coder/probability_diagnostics.py)

What stays local:

- the repair gate
- the feature vector
- the final trust policy over prior and exact-context repair

This is the intended boundary test for the new cache abstraction: shared prediction records and active/highest-order
semantics come from the kernel, while the descendant-specific repair choice stays project-local.
