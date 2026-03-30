# replay_fields

`replay_fields` is a second noncausal descendant over the shared noncausal reconstruction adapter.

It keeps the kernel contract narrow:

- bidirectional replay and reconstruction come from [`NoncausalReconstructiveAdapter`](/Users/asuramaya/Code/carving_machine_v3/open-predictive-coder/src/open_predictive_coder/noncausal_reconstructive.py)
- local field slicing stays in the example

The only local policy is field-oriented overlap accounting: how much of the adapter's replay surface lands inside
field-shaped spans.
