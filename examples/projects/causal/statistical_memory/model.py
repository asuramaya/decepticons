from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from open_predictive_coder import (
    ExactContextConfig,
    ExactContextMemory,
    ExactContextFitReport,
    NgramMemoryConfig,
    NgramMemoryReport,
    StatisticalBackoffConfig,
    StatisticalBackoffMemory,
    SupportWeightedMixer,
    ensure_tokens,
)


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(data).astype(np.int64, copy=False)


@dataclass(frozen=True)
class StatisticalMemoryConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    ngram_bigram_alpha: float = 0.5
    ngram_trigram_alpha: float = 0.5
    ngram_trigram_bucket_count: int = 2048

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.exact_max_order < 1:
            raise ValueError("exact_max_order must be >= 1")
        if self.exact_alpha < 0.0:
            raise ValueError("exact_alpha must be >= 0")
        if self.ngram_bigram_alpha < 0.0:
            raise ValueError("ngram_bigram_alpha must be >= 0")
        if self.ngram_trigram_alpha < 0.0:
            raise ValueError("ngram_trigram_alpha must be >= 0")
        if self.ngram_trigram_bucket_count < 1:
            raise ValueError("ngram_trigram_bucket_count must be >= 1")


@dataclass(frozen=True)
class StatisticalMemoryFitReport:
    ngram: NgramMemoryReport
    exact: ExactContextFitReport


@dataclass(frozen=True)
class StatisticalMemoryScore:
    tokens: int
    ngram_bits_per_byte: float
    exact_bits_per_byte: float
    mixed_bits_per_byte: float
    exact_order: int
    exact_support: float


@dataclass(frozen=True)
class StatisticalMemoryTrace:
    tokens: int
    steps: int
    base_probs: np.ndarray
    exact_probs: np.ndarray
    mixed_probs: np.ndarray
    component_names: tuple[tuple[str, ...], ...]


class StatisticalMemoryModel:
    def __init__(self, config: StatisticalMemoryConfig | None = None):
        self.config = config or StatisticalMemoryConfig()
        self.statistical_backoff = StatisticalBackoffMemory(
            StatisticalBackoffConfig(
                ngram=NgramMemoryConfig(
                    vocabulary_size=self.config.vocabulary_size,
                    bigram_alpha=self.config.ngram_bigram_alpha,
                    trigram_alpha=self.config.ngram_trigram_alpha,
                    trigram_bucket_count=self.config.ngram_trigram_bucket_count,
                )
            )
        )
        self.ngram_memory = self.statistical_backoff.ngram_memory
        self.exact_memory = ExactContextMemory(
            ExactContextConfig(
                vocabulary_size=self.config.vocabulary_size,
                max_order=self.config.exact_max_order,
                alpha=self.config.exact_alpha,
            )
        )
        self.mixer = SupportWeightedMixer()

    @classmethod
    def build(cls, **kwargs: object) -> "StatisticalMemoryModel":
        return cls(StatisticalMemoryConfig(**kwargs))

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> StatisticalMemoryFitReport:
        return StatisticalMemoryFitReport(
            ngram=self.statistical_backoff.fit(data).ngram,
            exact=self.exact_memory.fit(data),
        )

    def _ngram_distribution(self, prefix: np.ndarray) -> np.ndarray:
        return self.statistical_backoff.predict(prefix).highest_order_probs

    def _mixed_distribution(
        self,
        prefix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        base_probs = self._ngram_distribution(prefix)
        exact_experts = self.exact_memory.experts(prefix)
        exact_probs = self.exact_memory.predictive_distribution(prefix)
        blend = self.mixer.mix(
            base_probs=base_probs,
            experts=exact_experts,
            base_name="ngram",
            base_support=float(np.log1p(prefix.size)),
        )
        return base_probs, exact_probs, blend.probabilities, blend.component_names

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> StatisticalMemoryTrace:
        tokens = _coerce_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        base_rows: list[np.ndarray] = []
        exact_rows: list[np.ndarray] = []
        mixed_rows: list[np.ndarray] = []
        component_names: list[tuple[str, ...]] = []

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            base_probs, exact_probs, mixed_probs, names = self._mixed_distribution(prefix)
            base_rows.append(base_probs)
            exact_rows.append(exact_probs)
            mixed_rows.append(mixed_probs)
            component_names.append(names)

        return StatisticalMemoryTrace(
            tokens=int(tokens.size),
            steps=int(tokens.size - 1),
            base_probs=np.vstack(base_rows),
            exact_probs=np.vstack(exact_rows),
            mixed_probs=np.vstack(mixed_rows),
            component_names=tuple(component_names),
        )

    def predict_proba(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> np.ndarray:
        tokens = _coerce_tokens(prompt)
        if tokens.size == 0:
            raise ValueError("prompt must contain at least one token")
        _, _, mixed_probs, _ = self._mixed_distribution(tokens)
        return mixed_probs

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> StatisticalMemoryScore:
        trace = self.trace(sequence)
        tokens = trace.tokens
        targets = _coerce_tokens(sequence)[1:]
        if targets.size == 0:
            raise ValueError("sequence must contain at least two tokens")

        base_bits = -np.log2(np.clip(trace.base_probs[np.arange(targets.size), targets], 1e-12, 1.0))
        exact_bits = -np.log2(np.clip(trace.exact_probs[np.arange(targets.size), targets], 1e-12, 1.0))
        mixed_bits = -np.log2(np.clip(trace.mixed_probs[np.arange(targets.size), targets], 1e-12, 1.0))

        exact_experts = self.exact_memory.experts(_coerce_tokens(sequence)[:-1])
        best_order = 0
        exact_support = 0.0
        for prediction in reversed(exact_experts):
            if prediction.total > 0.0:
                best_order = int(prediction.order)
                exact_support = float(prediction.support)
                break

        return StatisticalMemoryScore(
            tokens=int(tokens),
            ngram_bits_per_byte=float(np.mean(base_bits)),
            exact_bits_per_byte=float(np.mean(exact_bits)),
            mixed_bits_per_byte=float(np.mean(mixed_bits)),
            exact_order=best_order,
            exact_support=exact_support,
        )


__all__ = [
    "StatisticalMemoryConfig",
    "StatisticalMemoryFitReport",
    "StatisticalMemoryModel",
    "StatisticalMemoryScore",
    "StatisticalMemoryTrace",
]
