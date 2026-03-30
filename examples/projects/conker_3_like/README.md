# conker_3_like

This folder is a from-scratch `Conker-3` style replica built on top of extracted primitives.

What moved into the kernel:

- the frozen linear decay-bank substrate, because it is a reusable substrate mechanism

What stays here:

- treating the second path as a local residual repair path
- using a selector to decide when local repair should modify the base distribution
- the choice to express correction as `base + weighted(local - unigram)` instead of as a general-purpose library API
- rollout mode switches already live in the kernel, while sampled readout remains irrelevant to this replica because its real local choice is residual-repair policy
