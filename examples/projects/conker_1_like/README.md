# conker_1_like

This folder is a from-scratch `Conker-1` style replica built on top of extracted primitives.

What stays in the kernel:

- frozen substrates
- feature/readout primitives
- basic evaluation surfaces

What stays here:

- the decision to use two causal experts
- the choice of a memory path plus a stability path
- the token-wise mixer policy used to combine them
- even though the kernel now has sampled readout, this Conker replica does not need it; the mixer policy is still the real project-local seam

That keeps the Conker-family compressor shape out of `src/` until it proves useful beyond this family.
