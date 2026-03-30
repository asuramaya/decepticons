# Related Work

This repo is an engineering synthesis, not a replica of any one paper. The implementation is anchored in the
following lines of work.

## 1. Predictive Coding Foundations

### David Mumford (1991, 1992)

- *On the computational architecture of the neocortex. I. The role of the thalamo-cortical loop* (1991)
- *On the computational architecture of the neocortex. II. The role of cortico-cortical loops* (1992)

Why it matters here:

- establishes the broad architectural intuition of hierarchical prediction and error-correcting loops
- frames cortical computation as iterative hypothesis maintenance instead of pure feedforward mapping

Links:

- https://pubmed.ncbi.nlm.nih.gov/1912004/
- https://pubmed.ncbi.nlm.nih.gov/1540675/

### Rao and Ballard (1999)

- *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects*

Why it matters here:

- canonical predictive-coding paper for the idea that feedback carries predictions and feedforward pathways carry residual errors
- the repo’s local residual features and patch summaries are conceptually downstream of this view

Links:

- https://pubmed.ncbi.nlm.nih.gov/10195184/
- https://www.nature.com/articles/nn0199_79

### Friston (2005) and Bogacz (2017)

- *A theory of cortical responses* (Friston, 2005)
- *A tutorial on the free-energy framework for modelling perception and learning* (Bogacz, 2017)

Why it matters here:

- extends predictive coding into variational and free-energy language
- useful for understanding iterative latent correction and prediction-error minimization beyond the original vision setting

Links:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC1569488/
- https://pubmed.ncbi.nlm.nih.gov/28298703/

### Whittington and Bogacz (2017)

- *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity*

Why it matters here:

- makes predictive coding legible to machine-learning readers by tying it to practical learning dynamics
- supports the repo’s choice to expose predictive coding as a usable modeling interface rather than only a neuroscience metaphor

Links:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/
- https://pubmed.ncbi.nlm.nih.gov/28333583/

## 2. Reservoir Computing and Fixed Recurrent Substrates

### Jaeger (2001)

- *The "echo state" approach to analysing and training recurrent neural networks*

Why it matters here:

- canonical source for echo-state style training where the recurrent substrate is fixed and only downstream weights are learned
- directly informs the frozen reservoir plus ridge-readout structure used in this repo

Links:

- https://publica.fraunhofer.de/entities/publication/7d4a7eec-a22c-4df0-903d-93f9cd5aca02

### Maass, Natschlager, and Markram (2002)

- *Real-time computing without stable states: a new framework for neural computation based on perturbations*

Why it matters here:

- liquid state machines broaden the same fixed-dynamics intuition and emphasize transient state as computational substrate
- supports the repo’s choice to treat recurrent state as a fading-memory workspace instead of a fully trained latent simulator

Links:

- https://pubmed.ncbi.nlm.nih.gov/12433288/

### Lukosevicius and Jaeger (2009)

- *Reservoir computing approaches to recurrent neural network training*

Why it matters here:

- the clearest canonical survey of reservoir computing variants and readout training strategies
- useful anchor for anyone extending this repo toward different readouts, adapted reservoirs, or online learning rules

Links:

- https://doi.org/10.1016/j.cosrev.2009.03.005
- https://www.sciencedirect.com/science/article/pii/S1574013709000173

## 3. Compression, Rate-Distortion, and Minimal Sufficient Latents

### Shannon (1948)

- *A Mathematical Theory of Communication*

Why it matters here:

- the repo scores byte prediction directly and treats compression as a first-class modeling concern
- this is the root reference for thinking in terms of coding cost instead of only task loss

Links:

- https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf

### Tishby, Pereira, and Bialek (1999/2000)

- *The Information Bottleneck Method*

Why it matters here:

- formalizes the intuition that a representation should preserve relevant information while discarding the rest
- this is the cleanest conceptual anchor for latent commits that are supposed to be shorter, cheaper, and still predictive

Links:

- https://www.princeton.edu/~wbialek/our_papers/tishby%2Bal_99.pdf
- https://research.google/pubs/the-information-bottleneck-method/

## 4. Predictive Coding for Sequential Learning

### Lotter, Kreiman, and Cox (2016)

- *Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning* (`PredNet`)

Why it matters here:

- shows a practical predictive-coding architecture where each layer predicts locally and forwards only residual mismatch
- useful as a modern machine-learning bridge between classic predictive coding and trainable sequence models

Links:

- https://arxiv.org/abs/1605.08104

### Annabi, Pitti, and Quoy (2022)

- *Continual Sequence Modeling With Predictive Coding*

Why it matters here:

- directly relevant to sequence modeling with predictive coding
- especially important because it connects predictive coding to continual learning and reservoir-style sequence memory

Links:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC9171436/
- https://pubmed.ncbi.nlm.nih.gov/35686118/

## 5. Recurrent Refinement and Byte-Level Latent Compression

### Dehghani et al. (2018)

- *Universal Transformers*

Why it matters here:

- provides the recurrent-refinement motif over a compressed internal representation
- this repo does not implement self-attentive UT blocks, but it does borrow the idea that repeated latent updates can be more important than a single flat pass

Links:

- https://arxiv.org/abs/1807.03819

### Pagnoni et al. (2025 ACL, 2024 arXiv)

- *Byte Latent Transformer: Patches Scale Better Than Tokens*

Why it matters here:

- the strongest modern reference for byte-level modeling with adaptive patches
- directly motivates this repo’s choice to expose bytes as the visible interface while allowing a shorter internal latent stream

Links:

- https://aclanthology.org/2025.acl-long.453/
- https://arxiv.org/abs/2412.09871

## How These References Map To The Repo

- `predictive coding`: Rao and Ballard, Friston, Whittington and Bogacz
- `frozen substrate + trained readout`: Jaeger, Maass et al., Lukosevicius and Jaeger
- `compression-aware latent commits`: Shannon, Tishby et al.
- `sequence and continual modeling`: Lotter et al., Annabi et al.
- `adaptive patching and latent-stream reasoning`: Dehghani et al., Pagnoni et al.

## What This Repo Is Not Claiming

- it does not claim biological realism
- it does not claim state-of-the-art compression or language modeling
- it does not claim to reproduce BLT, PredNet, Universal Transformers, or any predictive-coding theorem exactly

The intended reading is narrower:

- this library is a clean open reference point for people who want to combine predictive-coding style residuals,
  adaptive patches, and fixed recurrent substrates in one small Python codebase

