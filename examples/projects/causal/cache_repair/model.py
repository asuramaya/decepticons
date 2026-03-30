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
    ExactContextCache,
    ExactContextConfig,
    ExactContextFitReport,
    ExactContextMemory,
    StatisticalBackoffCache,
    StatisticalBackoffFitReport,
    ensure_tokens,
)
from open_predictive_coder.probability_diagnostics import ProbabilityDiagnosticsConfig, probability_diagnostics


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(data).astype(np.int64, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class CacheRepairConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    ngram_bigram_alpha: float = 0.5
    ngram_trigram_alpha: float = 0.5
    ngram_trigram_bucket_count: int = 2048
    mixture_steps: int = 128
    gate_l2: float = 1e-2
    candidate_count: int = 4


@dataclass(frozen=True)
class CacheRepairFitReport:
    backoff: StatisticalBackoffFitReport
    exact: ExactContextFitReport
    gate_weights: np.ndarray
    feature_names: tuple[str, ...]
    exact_win_rate: float


@dataclass(frozen=True)
class CacheRepairTrace:
    tokens: int
    steps: int
    prior_probs: np.ndarray
    exact_probs: np.ndarray
    mixed_probs: np.ndarray
    repair_strength: np.ndarray
    feature_matrix: np.ndarray
    feature_names: tuple[str, ...]
    exact_order: np.ndarray
    exact_support: np.ndarray


@dataclass(frozen=True)
class CacheRepairScore:
    tokens: int
    prior_bits_per_byte: float
    exact_bits_per_byte: float
    mixed_bits_per_byte: float
    mean_repair_strength: float
    mean_exact_support: float


class CacheRepairModel:
    FEATURE_NAMES = (
        "entropy",
        "peak",
        "candidate4",
        "agreement_mass",
        "exact_support",
        "exact_order_fraction",
        "bias",
    )

    def __init__(self, config: CacheRepairConfig | None = None):
        self.config = config or CacheRepairConfig()
        self.backoff_cache = StatisticalBackoffCache.from_vocabulary(
            self.config.vocabulary_size,
            bigram_alpha=self.config.ngram_bigram_alpha,
            trigram_alpha=self.config.ngram_trigram_alpha,
            trigram_bucket_count=self.config.ngram_trigram_bucket_count,
            mixture_steps=self.config.mixture_steps,
        )
        self.exact_cache = ExactContextCache(
            ExactContextMemory(
                ExactContextConfig(
                    vocabulary_size=self.config.vocabulary_size,
                    max_order=self.config.exact_max_order,
                    alpha=self.config.exact_alpha,
                )
            )
        )
        self.diagnostics_config = ProbabilityDiagnosticsConfig(top_k=self.config.candidate_count)
        self._gate_weights = np.zeros((len(self.FEATURE_NAMES),), dtype=np.float64)

    @classmethod
    def build(cls, **kwargs: object) -> "CacheRepairModel":
        return cls(CacheRepairConfig(**kwargs))

    def _feature_vector(self, prior_probs: np.ndarray, exact_probs: np.ndarray, *, exact_support: float, exact_order: int) -> np.ndarray:
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
    ) -> CacheRepairFitReport:
        tokens = _coerce_tokens(data)
        backoff = self.backoff_cache.fit(tokens)
        exact = self.exact_cache.fit(tokens)
        if tokens.size < 2:
            self._gate_weights.fill(0.0)
            return CacheRepairFitReport(backoff=backoff, exact=exact, gate_weights=self._gate_weights.copy(), feature_names=self.FEATURE_NAMES, exact_win_rate=0.0)

        design_rows: list[np.ndarray] = []
        labels: list[float] = []
        exact_wins = 0
        for index in range(1, tokens.size):
            prefix = tokens[:index]
            target = int(tokens[index])
            prior = self.backoff_cache.prediction_summary(prefix)
            exact_summary = self.exact_cache.prediction_summary(prefix)
            prior_probs = prior.predictive_distribution(mode="mixed")
            exact_probs = exact_summary.predictive_distribution(mode="active")
            exact_order = int(exact_summary.active_prediction.order)
            exact_support = float(exact_summary.active_prediction.support)
            design_rows.append(
                self._feature_vector(
                    prior_probs,
                    exact_probs,
                    exact_support=exact_support,
                    exact_order=exact_order,
                )
            )
            exact_better = float(exact_order > 0 and exact_probs[target] > prior_probs[target])
            labels.append(exact_better)
            exact_wins += int(exact_better)

        design = np.vstack(design_rows)
        targets = np.asarray(labels, dtype=np.float64)
        ridge = self.config.gate_l2 * np.eye(design.shape[1], dtype=np.float64)
        self._gate_weights = np.linalg.solve((design.T @ design) + ridge, design.T @ targets)
        return CacheRepairFitReport(
            backoff=backoff,
            exact=exact,
            gate_weights=self._gate_weights.copy(),
            feature_names=self.FEATURE_NAMES,
            exact_win_rate=float(exact_wins / max(tokens.size - 1, 1)),
        )

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> CacheRepairTrace:
        tokens = _coerce_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        prior_rows: list[np.ndarray] = []
        exact_rows: list[np.ndarray] = []
        mixed_rows: list[np.ndarray] = []
        repair_strengths: list[float] = []
        features: list[np.ndarray] = []
        exact_orders: list[int] = []
        exact_supports: list[float] = []

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            prior = self.backoff_cache.prediction_summary(prefix)
            exact_summary = self.exact_cache.prediction_summary(prefix)
            prior_probs = prior.predictive_distribution(mode="mixed")
            exact_probs = exact_summary.predictive_distribution(mode="active")
            exact_order = int(exact_summary.active_prediction.order)
            exact_support = float(exact_summary.active_prediction.support)
            feature_vector = self._feature_vector(
                prior_probs,
                exact_probs,
                exact_support=exact_support,
                exact_order=exact_order,
            )
            repair_strength = float(_sigmoid(feature_vector @ self._gate_weights))
            mixed = ((1.0 - repair_strength) * prior_probs) + (repair_strength * exact_probs)
            mixed /= np.sum(mixed)

            prior_rows.append(prior_probs)
            exact_rows.append(exact_probs)
            mixed_rows.append(mixed)
            repair_strengths.append(repair_strength)
            features.append(feature_vector)
            exact_orders.append(exact_order)
            exact_supports.append(exact_support)

        return CacheRepairTrace(
            tokens=int(tokens.size),
            steps=int(tokens.size - 1),
            prior_probs=np.vstack(prior_rows),
            exact_probs=np.vstack(exact_rows),
            mixed_probs=np.vstack(mixed_rows),
            repair_strength=np.asarray(repair_strengths, dtype=np.float64),
            feature_matrix=np.vstack(features),
            feature_names=self.FEATURE_NAMES,
            exact_order=np.asarray(exact_orders, dtype=np.int64),
            exact_support=np.asarray(exact_supports, dtype=np.float64),
        )

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        tokens = _coerce_tokens(prompt)
        if tokens.size == 0:
            raise ValueError("prompt must contain at least one token")
        trace = self.trace(np.concatenate([tokens, tokens[-1:]]))
        return trace.mixed_probs[-1]

    def score(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> CacheRepairScore:
        tokens = _coerce_tokens(sequence)
        trace = self.trace(tokens)
        targets = tokens[1:]
        indices = np.arange(targets.size)
        prior_bits = -np.log2(np.clip(trace.prior_probs[indices, targets], 1e-12, 1.0))
        exact_bits = -np.log2(np.clip(trace.exact_probs[indices, targets], 1e-12, 1.0))
        mixed_bits = -np.log2(np.clip(trace.mixed_probs[indices, targets], 1e-12, 1.0))
        return CacheRepairScore(
            tokens=int(tokens.size),
            prior_bits_per_byte=float(np.mean(prior_bits)),
            exact_bits_per_byte=float(np.mean(exact_bits)),
            mixed_bits_per_byte=float(np.mean(mixed_bits)),
            mean_repair_strength=float(np.mean(trace.repair_strength)),
            mean_exact_support=float(np.mean(trace.exact_support)),
        )


__all__ = [
    "CacheRepairConfig",
    "CacheRepairFitReport",
    "CacheRepairModel",
    "CacheRepairScore",
    "CacheRepairTrace",
]
