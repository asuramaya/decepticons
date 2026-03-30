from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .codecs import ensure_tokens
from .exact_context import ExactContextFitReport, ExactContextMemory, ExactContextPrediction
from .ngram_memory import NgramMemoryConfig
from .statistical_backoff import (
    StatisticalBackoffConfig,
    StatisticalBackoffFitReport,
    StatisticalBackoffMemory,
)


@dataclass(frozen=True)
class MemoryPredictionRecord:
    family: str
    name: str
    order: int
    context: tuple[int, ...]
    probabilities: np.ndarray
    support: float
    total: float
    confidence: float

    @property
    def active(self) -> bool:
        return self.total > 0.0


@dataclass(frozen=True)
class MemoryPredictionSummary:
    family: str
    context: tuple[int, ...]
    predictions: tuple[MemoryPredictionRecord, ...]
    active_index: int
    highest_order_index: int
    mixture_weights: np.ndarray | None = None

    @property
    def active_prediction(self) -> MemoryPredictionRecord:
        return self.predictions[self.active_index]

    @property
    def highest_order_prediction(self) -> MemoryPredictionRecord:
        return self.predictions[self.highest_order_index]

    def predictive_distribution(self, *, mode: str = "active") -> np.ndarray:
        if mode == "active":
            return self.active_prediction.probabilities
        if mode == "highest_order":
            return self.highest_order_prediction.probabilities
        if mode == "mixed":
            if self.mixture_weights is None:
                raise ValueError("mixed mode requires mixture_weights")
            weights = np.asarray(self.mixture_weights, dtype=np.float64)
            stacked = np.stack([prediction.probabilities for prediction in self.predictions], axis=0)
            mixed = np.sum(weights[:, None] * stacked, axis=0)
            total = float(np.sum(mixed))
            return mixed if total <= 0.0 else mixed / total
        raise ValueError(f"unknown mode: {mode}")


def _coerce_context(context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(context).astype(np.int64, copy=False)


def _record_from_exact(prediction: ExactContextPrediction) -> MemoryPredictionRecord:
    probabilities = np.asarray(prediction.probabilities, dtype=np.float64)
    return MemoryPredictionRecord(
        family="exact_context",
        name=prediction.name,
        order=int(prediction.order),
        context=tuple(int(token) for token in prediction.context),
        probabilities=probabilities,
        support=float(prediction.support),
        total=float(prediction.total),
        confidence=float(np.max(probabilities)),
    )


class ExactContextCache:
    def __init__(self, memory: ExactContextMemory | None = None):
        self.memory = memory or ExactContextMemory()

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> ExactContextFitReport:
        return self.memory.fit(data)

    def prediction_summary(
        self,
        context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> MemoryPredictionSummary:
        tokens = _coerce_context(context)
        predictions = [
            MemoryPredictionRecord(
                family="exact_context",
                name="unigram",
                order=0,
                context=(),
                probabilities=self.memory.unigram_probabilities(),
                support=1.0,
                total=1.0,
                confidence=float(np.max(self.memory.unigram_probabilities())),
            )
        ]
        predictions.extend(_record_from_exact(prediction) for prediction in self.memory.experts(tokens))
        active_index = 0
        for index, prediction in enumerate(predictions[1:], start=1):
            if prediction.active:
                active_index = index
        return MemoryPredictionSummary(
            family="exact_context",
            context=tuple(int(token) for token in tokens),
            predictions=tuple(predictions),
            active_index=active_index,
            highest_order_index=len(predictions) - 1,
        )

    def predictive_distribution(
        self,
        context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        *,
        mode: str = "active",
    ) -> np.ndarray:
        return self.prediction_summary(context).predictive_distribution(mode=mode)


class StatisticalBackoffCache:
    def __init__(self, memory: StatisticalBackoffMemory | None = None, *, config: StatisticalBackoffConfig | None = None):
        if memory is not None and config is not None:
            raise ValueError("pass either memory or config, not both")
        self.memory = memory or StatisticalBackoffMemory(config)

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary_size: int,
        *,
        bigram_alpha: float = 0.5,
        trigram_alpha: float = 0.5,
        trigram_bucket_count: int = 4096,
        mixture_steps: int = 128,
        mixture_learning_rate: float = 0.25,
    ) -> "StatisticalBackoffCache":
        return cls(
            config=StatisticalBackoffConfig(
                ngram=NgramMemoryConfig(
                    vocabulary_size=vocabulary_size,
                    bigram_alpha=bigram_alpha,
                    trigram_alpha=trigram_alpha,
                    trigram_bucket_count=trigram_bucket_count,
                ),
                mixture_steps=mixture_steps,
                mixture_learning_rate=mixture_learning_rate,
            )
        )

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> StatisticalBackoffFitReport:
        return self.memory.fit(data)

    def prediction_summary(
        self,
        context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> MemoryPredictionSummary:
        tokens = _coerce_context(context)
        prediction = self.memory.predict(tokens)
        records = (
            MemoryPredictionRecord(
                family="statistical_backoff",
                name="unigram",
                order=0,
                context=(),
                probabilities=np.asarray(prediction.unigram_probs, dtype=np.float64),
                support=1.0,
                total=1.0,
                confidence=float(np.max(prediction.unigram_probs)),
            ),
            MemoryPredictionRecord(
                family="statistical_backoff",
                name="bigram",
                order=1,
                context=tuple(int(token) for token in tokens[-1:]),
                probabilities=np.asarray(prediction.bigram_probs, dtype=np.float64),
                support=float(prediction.context_order >= 1),
                total=float(prediction.context_order >= 1),
                confidence=float(np.max(prediction.bigram_probs)),
            ),
            MemoryPredictionRecord(
                family="statistical_backoff",
                name="trigram",
                order=2,
                context=tuple(int(token) for token in tokens[-2:]),
                probabilities=np.asarray(prediction.trigram_probs, dtype=np.float64),
                support=float(prediction.context_order >= 2),
                total=float(prediction.context_order >= 2),
                confidence=float(np.max(prediction.trigram_probs)),
            ),
        )
        active_index = min(int(prediction.context_order), len(records) - 1)
        return MemoryPredictionSummary(
            family="statistical_backoff",
            context=tuple(int(token) for token in tokens),
            predictions=records,
            active_index=active_index,
            highest_order_index=len(records) - 1,
            mixture_weights=np.asarray(prediction.mixture_weights, dtype=np.float64),
        )

    def predictive_distribution(
        self,
        context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        *,
        mode: str = "mixed",
    ) -> np.ndarray:
        return self.prediction_summary(context).predictive_distribution(mode=mode)


__all__ = [
    "ExactContextCache",
    "MemoryPredictionRecord",
    "MemoryPredictionSummary",
    "StatisticalBackoffCache",
]
