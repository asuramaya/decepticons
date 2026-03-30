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
    ExactContextFitReport,
    ExactContextMemory,
    NgramMemoryConfig,
    NgramMemoryReport,
    ReplaySpan,
    RidgeReadout,
    SpanSelectionConfig,
    StatisticalBackoffConfig,
    StatisticalBackoffMemory,
    ensure_tokens,
    replay_spans_from_scores,
)
from open_predictive_coder.metrics import bits_per_byte_from_probabilities
from open_predictive_coder.probability_diagnostics import (
    ProbabilityDiagnosticsConfig,
    probability_diagnostics,
)


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(data).astype(np.int64, copy=False)


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


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return np.full(probabilities.shape[-1], 1.0 / probabilities.shape[-1], dtype=np.float64)
    return probabilities / total


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class ProgramControllerConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    ngram_bigram_alpha: float = 0.5
    ngram_trigram_alpha: float = 0.5
    ngram_trigram_bucket_count: int = 2048
    route_l2: float = 1e-2
    repair_l2: float = 1e-2
    candidate_count: int = 4
    repair_span_threshold: float = 0.45
    repair_span_min: int = 2
    repair_span_max_gap: int = 1

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
        if self.route_l2 < 0.0:
            raise ValueError("route_l2 must be >= 0")
        if self.repair_l2 < 0.0:
            raise ValueError("repair_l2 must be >= 0")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if not 0.0 <= self.repair_span_threshold <= 1.0:
            raise ValueError("repair_span_threshold must be within [0, 1]")
        if self.repair_span_min < 1:
            raise ValueError("repair_span_min must be >= 1")
        if self.repair_span_max_gap < 0:
            raise ValueError("repair_span_max_gap must be >= 0")


@dataclass(frozen=True)
class ProgramControllerFitReport:
    ngram: NgramMemoryReport
    exact: ExactContextFitReport
    route_weights: np.ndarray
    repair_weights: np.ndarray
    feature_names: tuple[str, ...]
    route_names: tuple[str, ...]
    route_accuracy: float
    repair_accuracy: float
    mean_repair_strength: float
    repair_span_count: int


@dataclass(frozen=True)
class ProgramControllerTrace:
    tokens: int
    steps: int
    prior_probs: np.ndarray
    exact_probs: np.ndarray
    repair_probs: np.ndarray
    mixed_probs: np.ndarray
    route_probs: np.ndarray
    repair_strength: np.ndarray
    controller_features: np.ndarray
    feature_names: tuple[str, ...]
    route_names: tuple[str, ...]
    route_choice: np.ndarray
    repair_spans: tuple[ReplaySpan, ...]


@dataclass(frozen=True)
class ProgramControllerScore:
    tokens: int
    prior_bits_per_byte: float
    exact_bits_per_byte: float
    repair_bits_per_byte: float
    mixed_bits_per_byte: float
    mean_route_entropy: float
    mean_repair_strength: float
    mean_prior_route_weight: float
    mean_exact_route_weight: float
    mean_repair_route_weight: float
    repair_span_count: int


class ProgramControllerModel:
    FEATURE_NAMES = (
        "entropy",
        "peak",
        "candidate4",
        "agreement",
        "agreement_mass",
        "exact_support",
        "exact_order_fraction",
        "prefix_fraction",
        "bias",
    )
    ROUTE_NAMES = ("prior", "exact", "repair")

    def __init__(self, config: ProgramControllerConfig | None = None):
        self.config = config or ProgramControllerConfig()
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
        self.route_controller = RidgeReadout(
            input_dim=len(self.FEATURE_NAMES),
            output_dim=len(self.ROUTE_NAMES),
            alpha=self.config.route_l2,
        )
        self.repair_controller = RidgeReadout(
            input_dim=len(self.FEATURE_NAMES),
            output_dim=2,
            alpha=self.config.repair_l2,
        )

    @classmethod
    def build(cls, **kwargs: object) -> "ProgramControllerModel":
        return cls(ProgramControllerConfig(**kwargs))

    def _ngram_distribution(self, prefix: np.ndarray) -> np.ndarray:
        return self.statistical_backoff.predict(prefix).highest_order_probs

    def _exact_distribution(self, prefix: np.ndarray) -> tuple[np.ndarray, float, int]:
        experts = self.exact_memory.experts(prefix)
        for prediction in reversed(experts):
            if prediction.total > 0.0:
                return prediction.probabilities, float(prediction.support), int(prediction.order)
        return self.exact_memory.unigram_probabilities(), 0.0, 0

    def _feature_vector(
        self,
        prior_probs: np.ndarray,
        exact_probs: np.ndarray,
        *,
        exact_support: float,
        exact_order: int,
        prefix_fraction: float,
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
                float(prefix_fraction),
                1.0,
            ],
            dtype=np.float64,
        )

    def _repair_strength_target(self, prior_probs: np.ndarray, exact_probs: np.ndarray, target: int) -> float:
        prior = float(max(prior_probs[target], 1e-12))
        exact = float(max(exact_probs[target], 1e-12))
        return exact / (prior + exact)

    def _route_probabilities(self, features: np.ndarray) -> np.ndarray:
        if self.route_controller.weights is None:
            return np.full(len(self.ROUTE_NAMES), 1.0 / len(self.ROUTE_NAMES), dtype=np.float64)
        return self.route_controller.probabilities(features[None, :])[0]

    def _repair_strength(self, features: np.ndarray) -> float:
        if self.repair_controller.weights is None:
            return 0.5
        return float(self.repair_controller.probabilities(features[None, :])[0, 1])

    def _route_summary(self, route_probs: np.ndarray) -> tuple[float, np.ndarray]:
        clipped = np.clip(np.asarray(route_probs, dtype=np.float64), 1e-12, 1.0)
        route_entropy = -np.sum(clipped * np.log(clipped), axis=1)
        route_entropy /= np.log(float(len(self.ROUTE_NAMES)))
        route_means = np.mean(route_probs, axis=0)
        return float(np.mean(route_entropy)), np.asarray(route_means, dtype=np.float64)

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> ProgramControllerFitReport:
        tokens = _coerce_tokens(data)
        ngram_report = self.statistical_backoff.fit(tokens).ngram
        exact_report = self.exact_memory.fit(tokens)

        if tokens.size < 2:
            self.route_controller.weights = None
            self.repair_controller.weights = None
            return ProgramControllerFitReport(
                ngram=ngram_report,
                exact=exact_report,
                route_weights=np.zeros((len(self.FEATURE_NAMES) + 1, len(self.ROUTE_NAMES)), dtype=np.float64),
                repair_weights=np.zeros((len(self.FEATURE_NAMES) + 1, 2), dtype=np.float64),
                feature_names=self.FEATURE_NAMES,
                route_names=self.ROUTE_NAMES,
                route_accuracy=0.0,
                repair_accuracy=0.0,
                mean_repair_strength=0.0,
                repair_span_count=0,
            )

        feature_rows: list[np.ndarray] = []
        route_labels: list[int] = []
        repair_labels: list[int] = []
        repair_strength_targets: list[float] = []
        repair_dominance: list[float] = []

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            target = int(tokens[index])
            prior_probs = self._ngram_distribution(prefix)
            exact_probs, exact_support, exact_order = self._exact_distribution(prefix)
            prefix_fraction = float(index) / float(max(tokens.size - 1, 1))
            features = self._feature_vector(
                prior_probs,
                exact_probs,
                exact_support=exact_support,
                exact_order=exact_order,
                prefix_fraction=prefix_fraction,
            )
            repair_strength_target = self._repair_strength_target(prior_probs, exact_probs, target)
            repair_probs = _normalize((1.0 - repair_strength_target) * prior_probs + repair_strength_target * exact_probs)
            route_losses = np.asarray(
                [
                    -np.log(max(float(prior_probs[target]), 1e-12)),
                    -np.log(max(float(exact_probs[target]), 1e-12)),
                    -np.log(max(float(repair_probs[target]), 1e-12)),
                ],
                dtype=np.float64,
            )
            feature_rows.append(features)
            route_labels.append(int(np.argmin(route_losses)))
            repair_labels.append(int(repair_strength_target >= 0.5))
            repair_strength_targets.append(repair_strength_target)
            repair_dominance.append(0.5 * (repair_strength_target + float(route_labels[-1] == 2)))

        design = np.vstack(feature_rows)
        route_targets = np.asarray(route_labels, dtype=np.int64)
        repair_targets = np.asarray(repair_labels, dtype=np.int64)

        self.route_controller.fit(design, route_targets)
        self.repair_controller.fit(design, repair_targets)

        route_probs = self.route_controller.probabilities(design)
        repair_probs = self.repair_controller.probabilities(design)
        route_predictions = np.argmax(route_probs, axis=1)
        repair_predictions = np.argmax(repair_probs, axis=1)

        repair_spans = replay_spans_from_scores(
            np.asarray(repair_dominance, dtype=np.float64),
            SpanSelectionConfig(
                threshold=self.config.repair_span_threshold,
                min_span=self.config.repair_span_min,
                max_gap=self.config.repair_span_max_gap,
            ),
            label="repair",
        )

        return ProgramControllerFitReport(
            ngram=ngram_report,
            exact=exact_report,
            route_weights=self.route_controller.weights.copy(),
            repair_weights=self.repair_controller.weights.copy(),
            feature_names=self.FEATURE_NAMES,
            route_names=self.ROUTE_NAMES,
            route_accuracy=float(np.mean(route_predictions == route_targets)),
            repair_accuracy=float(np.mean(repair_predictions == repair_targets)),
            mean_repair_strength=float(np.mean(repair_strength_targets)),
            repair_span_count=len(repair_spans),
        )

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> ProgramControllerTrace:
        tokens = _coerce_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        prior_rows: list[np.ndarray] = []
        exact_rows: list[np.ndarray] = []
        repair_rows: list[np.ndarray] = []
        mixed_rows: list[np.ndarray] = []
        route_rows: list[np.ndarray] = []
        repair_strength_rows: list[float] = []
        feature_rows: list[np.ndarray] = []
        route_choices: list[int] = []
        repair_scores: list[float] = []

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            prior_probs = self._ngram_distribution(prefix)
            exact_probs, exact_support, exact_order = self._exact_distribution(prefix)
            prefix_fraction = float(index) / float(max(tokens.size - 1, 1))
            features = self._feature_vector(
                prior_probs,
                exact_probs,
                exact_support=exact_support,
                exact_order=exact_order,
                prefix_fraction=prefix_fraction,
            )
            route_probs = self._route_probabilities(features)
            repair_strength = self._repair_strength(features)
            repair_probs = _normalize((1.0 - repair_strength) * prior_probs + repair_strength * exact_probs)
            mixed_probs = _normalize(
                route_probs[0] * prior_probs + route_probs[1] * exact_probs + route_probs[2] * repair_probs
            )

            prior_rows.append(prior_probs)
            exact_rows.append(exact_probs)
            repair_rows.append(repair_probs)
            mixed_rows.append(mixed_probs)
            route_rows.append(route_probs)
            repair_strength_rows.append(repair_strength)
            feature_rows.append(features)
            route_choices.append(int(np.argmax(route_probs)))
            repair_scores.append(0.5 * (float(route_probs[2]) + repair_strength))

        repair_spans = replay_spans_from_scores(
            np.asarray(repair_scores, dtype=np.float64),
            SpanSelectionConfig(
                threshold=self.config.repair_span_threshold,
                min_span=self.config.repair_span_min,
                max_gap=self.config.repair_span_max_gap,
            ),
            label="repair",
        )

        return ProgramControllerTrace(
            tokens=int(tokens.size),
            steps=int(tokens.size - 1),
            prior_probs=np.vstack(prior_rows),
            exact_probs=np.vstack(exact_rows),
            repair_probs=np.vstack(repair_rows),
            mixed_probs=np.vstack(mixed_rows),
            route_probs=np.vstack(route_rows),
            repair_strength=np.asarray(repair_strength_rows, dtype=np.float64),
            controller_features=np.vstack(feature_rows),
            feature_names=self.FEATURE_NAMES,
            route_names=self.ROUTE_NAMES,
            route_choice=np.asarray(route_choices, dtype=np.int64),
            repair_spans=repair_spans,
        )

    def predict_proba(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> np.ndarray:
        tokens = _coerce_tokens(prompt)
        if tokens.size == 0:
            raise ValueError("prompt must contain at least one token")
        prefix = tokens
        prior_probs = self._ngram_distribution(prefix)
        exact_probs, exact_support, exact_order = self._exact_distribution(prefix)
        features = self._feature_vector(
            prior_probs,
            exact_probs,
            exact_support=exact_support,
            exact_order=exact_order,
            prefix_fraction=1.0,
        )
        route_probs = self._route_probabilities(features)
        repair_strength = self._repair_strength(features)
        repair_probs = _normalize((1.0 - repair_strength) * prior_probs + repair_strength * exact_probs)
        return _normalize(route_probs[0] * prior_probs + route_probs[1] * exact_probs + route_probs[2] * repair_probs)

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> ProgramControllerScore:
        trace = self.trace(sequence)
        tokens = _coerce_tokens(sequence)
        targets = tokens[1:]
        if targets.size == 0:
            raise ValueError("sequence must contain at least two tokens")

        prior_bits = bits_per_byte_from_probabilities(trace.prior_probs, targets)
        exact_bits = bits_per_byte_from_probabilities(trace.exact_probs, targets)
        repair_bits = bits_per_byte_from_probabilities(trace.repair_probs, targets)
        mixed_bits = bits_per_byte_from_probabilities(trace.mixed_probs, targets)
        mean_route_entropy, route_means = self._route_summary(trace.route_probs)

        return ProgramControllerScore(
            tokens=int(tokens.size),
            prior_bits_per_byte=float(prior_bits),
            exact_bits_per_byte=float(exact_bits),
            repair_bits_per_byte=float(repair_bits),
            mixed_bits_per_byte=float(mixed_bits),
            mean_route_entropy=mean_route_entropy,
            mean_repair_strength=float(np.mean(trace.repair_strength)),
            mean_prior_route_weight=float(route_means[0]),
            mean_exact_route_weight=float(route_means[1]),
            mean_repair_route_weight=float(route_means[2]),
            repair_span_count=len(trace.repair_spans),
        )


__all__ = [
    "ProgramControllerConfig",
    "ProgramControllerFitReport",
    "ProgramControllerModel",
    "ProgramControllerScore",
    "ProgramControllerTrace",
]
