from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from open_predictive_coder import (
    ByteLatentPredictiveCoder,
    ExactContextConfig,
    ExactContextMemory,
    OpenPredictiveCoderConfig,
    SupportMixConfig,
    SupportWeightedMixer,
    ensure_tokens,
)
from open_predictive_coder.config import LatentConfig, ReservoirConfig, SegmenterConfig


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(data).astype(np.uint8, copy=False)


@dataclass(frozen=True)
class SequenceScore:
    tokens: int
    base_bits_per_byte: float
    exact_bits_per_byte: float
    mixed_bits_per_byte: float
    exact_support: float
    exact_order: int


@dataclass
class EarlyConkerLikeModel:
    base_model: ByteLatentPredictiveCoder
    exact_memory: ExactContextMemory
    mixer: SupportWeightedMixer

    @classmethod
    def build(
        cls,
        *,
        reservoir_size: int = 96,
        latent_dim: int = 24,
        exact_order: int = 3,
    ) -> "EarlyConkerLikeModel":
        config = OpenPredictiveCoderConfig(
            segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
            reservoir=ReservoirConfig(size=reservoir_size, connectivity=0.12, spectral_radius=0.9, leak=0.35, seed=11),
            latent=LatentConfig(
                latent_dim=latent_dim,
                global_dim=latent_dim,
                reservoir_features=min(latent_dim, reservoir_size),
                readout_l2=1e-5,
            ),
        )
        base_model = ByteLatentPredictiveCoder(config=config)
        exact_memory = ExactContextMemory(
            ExactContextConfig(
                vocabulary_size=config.vocabulary_size,
                max_order=exact_order,
                alpha=0.25,
            )
        )
        mixer = SupportWeightedMixer(
            SupportMixConfig(
                base_bias=0.35,
                expert_bias=0.65,
                support_scale=0.35,
            )
        )
        return cls(base_model=base_model, exact_memory=exact_memory, mixer=mixer)

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> dict[str, float]:
        fit_report = self.base_model.fit(data)
        exact_report = self.exact_memory.fit(data)
        return {
            "base_train_bits_per_byte": float(fit_report.train_bits_per_byte),
            "exact_contexts": float(sum(exact_report.contexts_by_order)),
            "exact_supports": float(max(exact_report.tokens - exact_report.sequences, 0)),
        }

    def _exact_components(self, prefix: np.ndarray) -> tuple[np.ndarray, tuple[object, ...], float, int]:
        experts = tuple(prediction for prediction in self.exact_memory.experts(prefix) if prediction.total > 0.0)
        exact_probabilities = self.exact_memory.predictive_distribution(prefix)
        best = None
        for prediction in reversed(experts):
            if prediction.total > 0.0:
                best = prediction
                break
        exact_support = 0.0 if best is None else float(best.support)
        exact_order = 0 if best is None else int(best.order)
        return exact_probabilities, experts, exact_support, exact_order

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> SequenceScore:
        tokens = _coerce_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        base_loss = 0.0
        exact_loss = 0.0
        mixed_loss = 0.0
        exact_support = 0.0
        exact_order = 0

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            target = int(tokens[index])
            base_probabilities = self.base_model.predict_proba(prefix)
            exact_probabilities, experts, exact_support, exact_order = self._exact_components(prefix)
            mixed = self.mixer.mix(base_probs=base_probabilities, experts=experts, base_support=0.0)

            base_loss += -np.log2(np.clip(base_probabilities[target], 1e-12, 1.0))
            exact_loss += -np.log2(np.clip(exact_probabilities[target], 1e-12, 1.0))
            mixed_loss += -np.log2(np.clip(mixed.probabilities[target], 1e-12, 1.0))

        denominator = float(tokens.size - 1)
        return SequenceScore(
            tokens=int(tokens.size),
            base_bits_per_byte=base_loss / denominator,
            exact_bits_per_byte=exact_loss / denominator,
            mixed_bits_per_byte=mixed_loss / denominator,
            exact_support=exact_support,
            exact_order=exact_order,
        )

    def generate(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        *,
        steps: int,
        greedy: bool = True,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> np.ndarray:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        tokens = _coerce_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")

        output = tokens.astype(np.uint8, copy=True).tolist()
        rng = np.random.default_rng(seed)
        context = tokens.copy()

        for _ in range(steps):
            base_probabilities = self.base_model.predict_proba(context)
            _, experts, _, _ = self._exact_components(context)
            blend = self.mixer.mix(base_probs=base_probabilities, experts=experts, base_support=0.0)
            probabilities = blend.probabilities
            if greedy:
                next_token = int(np.argmax(probabilities))
            else:
                scaled = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
                stabilized = np.exp(scaled - np.max(scaled))
                sample_probs = stabilized / np.sum(stabilized)
                next_token = int(rng.choice(len(sample_probs), p=sample_probs))
            output.append(next_token)
            context = np.asarray(output, dtype=np.uint8)

        return np.asarray(output, dtype=np.uint8)


__all__ = [
    "EarlyConkerLikeModel",
    "SequenceScore",
]
