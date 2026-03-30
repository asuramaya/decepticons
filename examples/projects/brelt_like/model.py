from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_predictive_coder import (
    AdaptiveSegmenter,
    ByteLatentFeatureView,
    FitReport,
    LatentCommitter,
    LatentConfig,
    RidgeReadout,
    SequenceReport,
    SegmenterConfig,
    ensure_tokens,
)
from open_predictive_coder.metrics import bits_per_byte_from_logits, softmax


@dataclass(frozen=True)
class BreltLikeConfig:
    vocabulary_size: int = 256
    token_view_dim: int = 32
    segmenter: SegmenterConfig = field(
        default_factory=lambda: SegmenterConfig(
            mode="adaptive",
            patch_size=4,
            min_patch_size=2,
            max_patch_size=10,
            novelty_threshold=0.08,
        )
    )
    latent: LatentConfig = field(
        default_factory=lambda: LatentConfig(
            latent_dim=24,
            global_dim=24,
            reservoir_features=32,
            bridge_scale=0.25,
            global_update_scale=0.3,
            readout_l2=1e-4,
        )
    )
    seed: int = 7

    def __post_init__(self) -> None:
        if self.vocabulary_size != 256:
            raise ValueError("BreltLikeConfig expects byte vocabulary size 256")
        if self.token_view_dim < 1:
            raise ValueError("token_view_dim must be >= 1")
        if self.latent.reservoir_features != self.token_view_dim:
            raise ValueError("latent.reservoir_features must match token_view_dim")


@dataclass(frozen=True)
class BreltLikeTrace:
    features: np.ndarray
    targets: np.ndarray
    boundaries: np.ndarray
    tokens: int
    patches: int
    mean_patch_size: float
    compression_ratio: float


class BytePatchEncoder:
    def __init__(self, vocabulary_size: int, view_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        table = rng.standard_normal((vocabulary_size, view_dim)).astype(np.float64)
        table /= np.sqrt(max(view_dim, 1))
        self.table = np.tanh(table)

    def encode(self, token: int) -> np.ndarray:
        return self.table[int(token)]


class BreltLikeByteModel:
    def __init__(self, config: BreltLikeConfig | None = None):
        self.config = config or BreltLikeConfig()
        self.encoder = BytePatchEncoder(self.config.vocabulary_size, self.config.token_view_dim, seed=self.config.seed + 11)
        self.segmenter = AdaptiveSegmenter(self.config.segmenter)
        self.committer = LatentCommitter(
            self.config.latent,
            substrate_size=self.config.token_view_dim,
            seed=self.config.seed + 13,
        )
        self.feature_view = ByteLatentFeatureView(max_patch_size=self.config.segmenter.max_patch_size)
        self.readout = RidgeReadout(
            input_dim=self.feature_view.feature_dim(self.config.latent),
            output_dim=self.config.vocabulary_size,
            alpha=self.config.latent.readout_l2,
        )

    @property
    def feature_dim(self) -> int:
        return self.feature_view.feature_dim(self.config.latent)

    def _coerce_sequences(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> tuple[np.ndarray, ...]:
        if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence) and data and all(isinstance(item, int) for item in data):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence):
            return tuple(ensure_tokens(item) for item in data)
        return (ensure_tokens(data),)

    def _step(self, state, token: int):
        local_view = self.encoder.encode(int(token))
        observation = self.committer.step(state, local_view, self.segmenter)
        feature = self.feature_view.encode(observation)
        return feature, observation

    def trace(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> BreltLikeTrace:
        tokens = ensure_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        state = self.committer.initial_state()
        features: list[np.ndarray] = []
        boundaries: list[float] = []
        for token in tokens[:-1]:
            feature, observation = self._step(state, int(token))
            features.append(feature)
            boundaries.append(1.0 if observation.boundary else 0.0)

        total_tokens = int(tokens.size)
        segment_stats = AdaptiveSegmenter.summarize(total_tokens, state.patches)
        return BreltLikeTrace(
            features=np.vstack(features),
            targets=tokens[1:].astype(np.int64, copy=False),
            boundaries=np.asarray(boundaries, dtype=np.float64),
            tokens=total_tokens,
            patches=state.patches,
            mean_patch_size=segment_stats.mean_patch_size,
            compression_ratio=segment_stats.compression_ratio,
        )

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> FitReport:
        sequences = self._coerce_sequences(data)
        feature_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        total_tokens = 0
        total_patches = 0
        for sequence in sequences:
            trace = self.trace(sequence)
            feature_batches.append(trace.features)
            target_batches.append(trace.targets)
            total_tokens += trace.tokens
            total_patches += trace.patches

        design = np.concatenate(feature_batches, axis=0)
        targets = np.concatenate(target_batches, axis=0)
        self.readout.fit(design, targets)
        logits = self.readout.logits(design)
        compression = total_tokens / max(total_patches, 1)
        return FitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            patches=total_patches,
            mean_patch_size=compression,
            compression_ratio=compression,
            train_bits_per_byte=bits_per_byte_from_logits(logits, targets),
        )

    def score(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> SequenceReport:
        trace = self.trace(sequence)
        logits = self.readout.logits(trace.features)
        return SequenceReport(
            tokens=trace.tokens,
            patches=trace.patches,
            mean_patch_size=trace.mean_patch_size,
            compression_ratio=trace.compression_ratio,
            bits_per_byte=bits_per_byte_from_logits(logits, trace.targets),
        )

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")

        state = self.committer.initial_state()
        feature = None
        for token in tokens:
            feature, _ = self._step(state, int(token))
        assert feature is not None
        return self.readout.probabilities(feature[None, :])[0]

    def generate(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        *,
        steps: int,
        temperature: float = 1.0,
        greedy: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")

        rng = np.random.default_rng(seed)
        state = self.committer.initial_state()
        feature = None
        output = tokens.astype(np.uint8, copy=True).tolist()
        for token in tokens:
            feature, _ = self._step(state, int(token))
        assert feature is not None

        for _ in range(steps):
            logits = self.readout.logits(feature[None, :])[0]
            if greedy:
                next_token = int(np.argmax(logits))
            else:
                probs = softmax((logits / temperature)[None, :], axis=-1)[0]
                next_token = int(rng.choice(self.config.vocabulary_size, p=probs))
            output.append(next_token)
            feature, _ = self._step(state, next_token)
        return np.asarray(output, dtype=np.uint8)


__all__ = [
    "BreltLikeByteModel",
    "BreltLikeConfig",
    "BreltLikeTrace",
    "BytePatchEncoder",
]
