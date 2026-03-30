from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .codecs import ensure_tokens
from .metrics import bits_per_token_from_probabilities
from .ngram_memory import NgramMemory, NgramMemoryConfig, NgramMemoryReport


def _coerce_sequences(
    data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
) -> tuple[np.ndarray, ...]:
    if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
        return (ensure_tokens(data),)
    if isinstance(data, Sequence) and data and all(isinstance(item, int) for item in data):
        return (ensure_tokens(data),)
    if isinstance(data, Sequence):
        return tuple(ensure_tokens(item) for item in data)
    return (ensure_tokens(data),)


@dataclass(frozen=True)
class StatisticalBackoffConfig:
    ngram: NgramMemoryConfig = field(default_factory=NgramMemoryConfig)
    mixture_steps: int = 128
    mixture_learning_rate: float = 0.25
    mixture_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.mixture_steps < 0:
            raise ValueError("mixture_steps must be >= 0")
        if self.mixture_learning_rate <= 0.0:
            raise ValueError("mixture_learning_rate must be > 0")
        if self.mixture_epsilon <= 0.0:
            raise ValueError("mixture_epsilon must be > 0")


@dataclass(frozen=True)
class StatisticalBackoffPrediction:
    context_order: int
    unigram_probs: np.ndarray
    bigram_probs: np.ndarray
    trigram_probs: np.ndarray
    highest_order_probs: np.ndarray
    mixed_probs: np.ndarray
    mixture_weights: np.ndarray


@dataclass(frozen=True)
class StatisticalBackoffTrace:
    tokens: int
    steps: int
    unigram_probs: np.ndarray
    bigram_probs: np.ndarray
    trigram_probs: np.ndarray
    highest_order_probs: np.ndarray
    mixed_probs: np.ndarray
    mixture_weights: np.ndarray


@dataclass(frozen=True)
class StatisticalBackoffScore:
    tokens: int
    unigram_bits_per_token: float
    bigram_bits_per_token: float
    trigram_bits_per_token: float
    highest_order_bits_per_token: float
    mixed_bits_per_token: float
    mixture_weights: np.ndarray


@dataclass(frozen=True)
class StatisticalBackoffFitReport:
    ngram: NgramMemoryReport
    unigram_bits_per_token: float
    bigram_bits_per_token: float
    trigram_bits_per_token: float
    mixed_bits_per_token: float
    mixture_weights: np.ndarray


class StatisticalBackoffMemory:
    def __init__(self, config: StatisticalBackoffConfig | None = None):
        self.config = config or StatisticalBackoffConfig()
        self.ngram_memory = NgramMemory(self.config.ngram)
        self._mixture_weights = np.full((3,), 1.0 / 3.0, dtype=np.float64)

    @property
    def mixture_weights(self) -> np.ndarray:
        return self._mixture_weights.copy()

    def clear(self) -> None:
        self.ngram_memory.clear()
        self._mixture_weights.fill(1.0 / 3.0)

    def _check_tokens(self, tokens: np.ndarray) -> np.ndarray:
        checked = np.asarray(tokens, dtype=np.int64)
        if checked.ndim != 1:
            raise ValueError("tokens must be rank-1")
        if checked.size == 0:
            return checked
        if int(np.min(checked)) < 0 or int(np.max(checked)) >= self.config.ngram.vocabulary_size:
            raise ValueError("tokens must lie within the configured vocabulary")
        return checked

    def _fit_mixture(self, per_order_probs: np.ndarray) -> np.ndarray:
        epsilon = self.config.mixture_epsilon
        weights = np.full((per_order_probs.shape[1],), 1.0 / float(per_order_probs.shape[1]), dtype=np.float64)
        if per_order_probs.shape[0] == 0 or self.config.mixture_steps == 0:
            return weights
        for _ in range(self.config.mixture_steps):
            mixed = np.clip(per_order_probs @ weights, epsilon, 1.0)
            gradient = -np.mean(per_order_probs / mixed[:, None], axis=0)
            centered = gradient - float(np.dot(gradient, weights))
            weights *= np.exp(-self.config.mixture_learning_rate * centered)
            total = float(np.sum(weights))
            if total <= epsilon:
                weights.fill(1.0 / float(weights.size))
            else:
                weights /= total
        return weights

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> StatisticalBackoffFitReport:
        ngram_report = self.ngram_memory.fit(data)
        sequences = tuple(self._check_tokens(sequence.astype(np.int64, copy=False)) for sequence in _coerce_sequences(data))
        chosen_rows: list[np.ndarray] = []
        for sequence in sequences:
            if sequence.size == 0:
                continue
            chosen_rows.append(
                np.column_stack(
                    [
                        self.ngram_memory.chosen_probs(sequence, order="unigram"),
                        self.ngram_memory.chosen_probs(sequence, order="bigram"),
                        self.ngram_memory.chosen_probs(sequence, order="trigram"),
                    ]
                )
            )
        chosen = np.vstack(chosen_rows) if chosen_rows else np.zeros((0, 3), dtype=np.float64)
        self._mixture_weights = self._fit_mixture(chosen)
        mixed = np.clip(chosen @ self._mixture_weights, self.config.mixture_epsilon, 1.0)
        return StatisticalBackoffFitReport(
            ngram=ngram_report,
            unigram_bits_per_token=float(-np.mean(np.log2(np.clip(chosen[:, 0], self.config.mixture_epsilon, 1.0)))) if chosen.size else 0.0,
            bigram_bits_per_token=float(-np.mean(np.log2(np.clip(chosen[:, 1], self.config.mixture_epsilon, 1.0)))) if chosen.size else 0.0,
            trigram_bits_per_token=float(-np.mean(np.log2(np.clip(chosen[:, 2], self.config.mixture_epsilon, 1.0)))) if chosen.size else 0.0,
            mixed_bits_per_token=float(-np.mean(np.log2(mixed))) if chosen.size else 0.0,
            mixture_weights=self.mixture_weights,
        )

    def predict(self, prefix: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> StatisticalBackoffPrediction:
        tokens = self._check_tokens(ensure_tokens(prefix).astype(np.int64, copy=False))
        unigram = self.ngram_memory.unigram_probs()
        if tokens.size == 0:
            bigram = unigram
            trigram = unigram
            context_order = 0
        elif tokens.size == 1:
            bigram = self.ngram_memory.bigram_probs(int(tokens[-1]))
            trigram = bigram
            context_order = 1
        else:
            bigram = self.ngram_memory.bigram_probs(int(tokens[-1]))
            trigram = self.ngram_memory.trigram_probs(int(tokens[-2]), int(tokens[-1]))
            context_order = 2
        highest = unigram if context_order == 0 else bigram if context_order == 1 else trigram
        mixed = (
            (self._mixture_weights[0] * unigram)
            + (self._mixture_weights[1] * bigram)
            + (self._mixture_weights[2] * trigram)
        )
        mixed /= np.sum(mixed)
        return StatisticalBackoffPrediction(
            context_order=context_order,
            unigram_probs=np.asarray(unigram, dtype=np.float64),
            bigram_probs=np.asarray(bigram, dtype=np.float64),
            trigram_probs=np.asarray(trigram, dtype=np.float64),
            highest_order_probs=np.asarray(highest, dtype=np.float64),
            mixed_probs=np.asarray(mixed, dtype=np.float64),
            mixture_weights=self.mixture_weights,
        )

    def predictive_distribution(
        self,
        prefix: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        *,
        mode: str = "mixed",
    ) -> np.ndarray:
        prediction = self.predict(prefix)
        if mode == "mixed":
            return prediction.mixed_probs
        if mode == "highest_order":
            return prediction.highest_order_probs
        if mode == "unigram":
            return prediction.unigram_probs
        if mode == "bigram":
            return prediction.bigram_probs
        if mode == "trigram":
            return prediction.trigram_probs
        raise ValueError(f"unknown mode: {mode}")

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> StatisticalBackoffTrace:
        tokens = self._check_tokens(ensure_tokens(sequence).astype(np.int64, copy=False))
        if tokens.size == 0:
            return StatisticalBackoffTrace(
                tokens=0,
                steps=0,
                unigram_probs=np.zeros((0, self.config.ngram.vocabulary_size), dtype=np.float64),
                bigram_probs=np.zeros((0, self.config.ngram.vocabulary_size), dtype=np.float64),
                trigram_probs=np.zeros((0, self.config.ngram.vocabulary_size), dtype=np.float64),
                highest_order_probs=np.zeros((0, self.config.ngram.vocabulary_size), dtype=np.float64),
                mixed_probs=np.zeros((0, self.config.ngram.vocabulary_size), dtype=np.float64),
                mixture_weights=self.mixture_weights,
            )

        unigram_rows: list[np.ndarray] = []
        bigram_rows: list[np.ndarray] = []
        trigram_rows: list[np.ndarray] = []
        highest_rows: list[np.ndarray] = []
        mixed_rows: list[np.ndarray] = []
        for index in range(tokens.size):
            prefix = tokens[:index]
            prediction = self.predict(prefix)
            unigram_rows.append(prediction.unigram_probs)
            bigram_rows.append(prediction.bigram_probs)
            trigram_rows.append(prediction.trigram_probs)
            highest_rows.append(prediction.highest_order_probs)
            mixed_rows.append(prediction.mixed_probs)
        return StatisticalBackoffTrace(
            tokens=int(tokens.size),
            steps=int(tokens.size),
            unigram_probs=np.vstack(unigram_rows),
            bigram_probs=np.vstack(bigram_rows),
            trigram_probs=np.vstack(trigram_rows),
            highest_order_probs=np.vstack(highest_rows),
            mixed_probs=np.vstack(mixed_rows),
            mixture_weights=self.mixture_weights,
        )

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> StatisticalBackoffScore:
        tokens = self._check_tokens(ensure_tokens(sequence).astype(np.int64, copy=False))
        trace = self.trace(tokens)
        return StatisticalBackoffScore(
            tokens=int(tokens.size),
            unigram_bits_per_token=bits_per_token_from_probabilities(trace.unigram_probs, tokens),
            bigram_bits_per_token=bits_per_token_from_probabilities(trace.bigram_probs, tokens),
            trigram_bits_per_token=bits_per_token_from_probabilities(trace.trigram_probs, tokens),
            highest_order_bits_per_token=bits_per_token_from_probabilities(trace.highest_order_probs, tokens),
            mixed_bits_per_token=bits_per_token_from_probabilities(trace.mixed_probs, tokens),
            mixture_weights=self.mixture_weights,
        )


__all__ = [
    "StatisticalBackoffConfig",
    "StatisticalBackoffFitReport",
    "StatisticalBackoffMemory",
    "StatisticalBackoffPrediction",
    "StatisticalBackoffScore",
    "StatisticalBackoffTrace",
]
