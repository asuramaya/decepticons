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
    ensure_tokens,
)
from open_predictive_coder.probability_diagnostics import (
    ProbabilityDiagnosticsConfig,
    probability_diagnostics,
)


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(data).astype(np.int64, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return np.full(probabilities.shape[-1], 1.0 / probabilities.shape[-1], dtype=np.float64)
    return probabilities / total


@dataclass(frozen=True)
class PackedMemoryControllerConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    ngram_bigram_alpha: float = 0.5
    ngram_trigram_alpha: float = 0.5
    ngram_trigram_bucket_count: int = 2048
    controller_l2: float = 1e-2
    candidate_count: int = 4

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
        if self.controller_l2 < 0.0:
            raise ValueError("controller_l2 must be >= 0")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")


@dataclass(frozen=True)
class PackedMemoryControllerFitReport:
    ngram: NgramMemoryReport
    exact: ExactContextFitReport
    controller_weights: np.ndarray
    feature_names: tuple[str, ...]
    exact_win_rate: float


@dataclass(frozen=True)
class PackedMemoryControllerTrace:
    tokens: int
    steps: int
    prior_probs: np.ndarray
    exact_probs: np.ndarray
    mixed_probs: np.ndarray
    memory_trust: np.ndarray
    controller_features: np.ndarray
    feature_names: tuple[str, ...]
    exact_support: np.ndarray
    exact_order: np.ndarray


@dataclass(frozen=True)
class PackedMemoryControllerScore:
    tokens: int
    prior_bits_per_byte: float
    exact_bits_per_byte: float
    mixed_bits_per_byte: float
    mean_memory_trust: float
    exact_win_rate: float
    mean_agreement_mass: float
    mean_candidate4: float


class PackedMemoryControllerModel:
    FEATURE_NAMES = (
        "entropy",
        "peak",
        "candidate4",
        "agreement",
        "agreement_mass",
        "exact_support",
        "exact_order_fraction",
        "bias",
    )

    def __init__(self, config: PackedMemoryControllerConfig | None = None):
        self.config = config or PackedMemoryControllerConfig()
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
        self.diagnostics_config = ProbabilityDiagnosticsConfig(top_k=self.config.candidate_count)
        self._controller_weights = np.zeros(len(self.FEATURE_NAMES), dtype=np.float64)

    @classmethod
    def build(cls, **kwargs: object) -> "PackedMemoryControllerModel":
        return cls(PackedMemoryControllerConfig(**kwargs))

    def _ngram_distribution(self, prefix: np.ndarray) -> np.ndarray:
        return self.statistical_backoff.predict(prefix).highest_order_probs

    def _exact_distribution(self, prefix: np.ndarray) -> tuple[np.ndarray, float, int]:
        experts = self.exact_memory.experts(prefix)
        for prediction in reversed(experts):
            if prediction.total > 0.0:
                return prediction.probabilities, float(prediction.support), int(prediction.order)
        return self.exact_memory.unigram_probabilities(), 0.0, 0

    def _controller_vector(
        self,
        prior_probs: np.ndarray,
        exact_probs: np.ndarray,
        *,
        exact_support: float,
        exact_order: int,
    ) -> np.ndarray:
        diagnostics = probability_diagnostics(
            prior_probs[None, :],
            exact_probs[None, :],
            config=self.diagnostics_config,
        )
        return np.asarray(
            [
                float(diagnostics.entropy[0]),
                float(diagnostics.peak[0]),
                float(diagnostics.top_k_mass[0]),
                float(diagnostics.overlap[0]),
                float(diagnostics.shared_top_k_mass[0]),
                float(exact_support),
                float(exact_order) / float(max(self.config.exact_max_order, 1)),
                1.0,
            ],
            dtype=np.float64,
        )

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> PackedMemoryControllerFitReport:
        tokens = _coerce_tokens(data)
        ngram_report = self.statistical_backoff.fit(tokens).ngram
        exact_report = self.exact_memory.fit(tokens)

        if tokens.size < 2:
            self._controller_weights = np.zeros(len(self.FEATURE_NAMES), dtype=np.float64)
            return PackedMemoryControllerFitReport(
                ngram=ngram_report,
                exact=exact_report,
                controller_weights=self._controller_weights.copy(),
                feature_names=self.FEATURE_NAMES,
                exact_win_rate=0.0,
            )

        design_rows: list[np.ndarray] = []
        labels: list[float] = []
        exact_wins = 0
        for index in range(1, tokens.size):
            prefix = tokens[:index]
            target = int(tokens[index])
            prior_probs = self._ngram_distribution(prefix)
            exact_probs, exact_support, exact_order = self._exact_distribution(prefix)
            design_rows.append(
                self._controller_vector(
                    prior_probs,
                    exact_probs,
                    exact_support=exact_support,
                    exact_order=exact_order,
                )
            )
            exact_better = float(exact_probs[target] > prior_probs[target])
            labels.append(exact_better)
            exact_wins += int(exact_better)

        design = np.vstack(design_rows)
        targets = np.asarray(labels, dtype=np.float64)
        ridge = self.config.controller_l2 * np.eye(design.shape[1], dtype=np.float64)
        self._controller_weights = np.linalg.solve((design.T @ design) + ridge, design.T @ targets)

        return PackedMemoryControllerFitReport(
            ngram=ngram_report,
            exact=exact_report,
            controller_weights=self._controller_weights.copy(),
            feature_names=self.FEATURE_NAMES,
            exact_win_rate=float(exact_wins / max(tokens.size - 1, 1)),
        )

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> PackedMemoryControllerTrace:
        tokens = _coerce_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        prior_rows: list[np.ndarray] = []
        exact_rows: list[np.ndarray] = []
        mixed_rows: list[np.ndarray] = []
        trust_rows: list[float] = []
        feature_rows: list[np.ndarray] = []
        exact_supports: list[float] = []
        exact_orders: list[int] = []

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            prior_probs = self._ngram_distribution(prefix)
            exact_probs, exact_support, exact_order = self._exact_distribution(prefix)
            feature_vector = self._controller_vector(
                prior_probs,
                exact_probs,
                exact_support=exact_support,
                exact_order=exact_order,
            )
            trust = float(_sigmoid(feature_vector @ self._controller_weights))
            mixed_probs = _normalize(((1.0 - trust) * prior_probs) + (trust * exact_probs))

            prior_rows.append(prior_probs)
            exact_rows.append(exact_probs)
            mixed_rows.append(mixed_probs)
            trust_rows.append(trust)
            feature_rows.append(feature_vector)
            exact_supports.append(exact_support)
            exact_orders.append(exact_order)

        return PackedMemoryControllerTrace(
            tokens=int(tokens.size),
            steps=int(tokens.size - 1),
            prior_probs=np.vstack(prior_rows),
            exact_probs=np.vstack(exact_rows),
            mixed_probs=np.vstack(mixed_rows),
            memory_trust=np.asarray(trust_rows, dtype=np.float64),
            controller_features=np.vstack(feature_rows),
            feature_names=self.FEATURE_NAMES,
            exact_support=np.asarray(exact_supports, dtype=np.float64),
            exact_order=np.asarray(exact_orders, dtype=np.int64),
        )

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> PackedMemoryControllerScore:
        tokens = _coerce_tokens(sequence)
        trace = self.trace(tokens)
        targets = tokens[1:].astype(np.int64, copy=False)
        rows = np.arange(targets.size)

        prior_bits = -np.log2(np.clip(trace.prior_probs[rows, targets], 1e-12, 1.0))
        exact_bits = -np.log2(np.clip(trace.exact_probs[rows, targets], 1e-12, 1.0))
        mixed_bits = -np.log2(np.clip(trace.mixed_probs[rows, targets], 1e-12, 1.0))

        exact_wins = np.mean(trace.exact_probs[rows, targets] > trace.prior_probs[rows, targets])
        agreement_mass_index = self.FEATURE_NAMES.index("agreement_mass")
        candidate4_index = self.FEATURE_NAMES.index("candidate4")

        return PackedMemoryControllerScore(
            tokens=int(tokens.size),
            prior_bits_per_byte=float(np.mean(prior_bits)),
            exact_bits_per_byte=float(np.mean(exact_bits)),
            mixed_bits_per_byte=float(np.mean(mixed_bits)),
            mean_memory_trust=float(np.mean(trace.memory_trust)),
            exact_win_rate=float(exact_wins),
            mean_agreement_mass=float(np.mean(trace.controller_features[:, agreement_mass_index])),
            mean_candidate4=float(np.mean(trace.controller_features[:, candidate4_index])),
        )

    def controller_weights(self) -> np.ndarray:
        return self._controller_weights.copy()


__all__ = [
    "PackedMemoryControllerConfig",
    "PackedMemoryControllerFitReport",
    "PackedMemoryControllerModel",
    "PackedMemoryControllerScore",
    "PackedMemoryControllerTrace",
]
