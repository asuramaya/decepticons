# brelt_like

This folder is a runnable project-layer example anchored to the actual `brelt` repository in `/tmp/brelt_repo`.

The point is to mirror the real shape, not to reimplement the full training system:

- bytes in
- patch segmentation
- patch commit into a shorter latent stream
- recurrent global mixing
- bridge back to local byte prediction

The example uses current kernel primitives where they fit:

- `SegmenterConfig` and `AdaptiveSegmenter`
- `LatentConfig` and `LatentCommitter`
- `ByteLatentFeatureView`
- `RidgeReadout`
- `ByteCodec`

## What Is Faithful

- the causal byte interface
- the patch/commit boundary
- the shorter latent stream and recurrent global state
- the bridge from latent state back to byte prediction features
- bits-per-byte scoring

## What Is Simplified

- no large transformer stack
- no learned segmentation head
- no quantized export pipeline
- no distributed training or benchmark harness
- no full rate-distortion controller

The example is intended as a structural/dev target: enough to test the architecture from scratch, but small enough to run quickly.

## Entry Points

- `probe.py`: prints the model and patching dimensions
- `smoke.py`: fits on a small corpus, scores it, and samples from a prompt

## Run

From the repository root:

```bash
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/brelt_like/probe.py
```

```bash
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/brelt_like/smoke.py
```
