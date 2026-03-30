"""Shared Conker replica helpers.

Kernel code owns reusable primitives such as substrates, views, readouts,
and runtime evaluation surfaces. This module stays in the project layer and
only hosts the Conker-family composition policy plus the small builders that
multiple Conker descendants reuse.

It also provides a very thin project-layer replica base so the descendant
wrappers do not repeat the same fit/score delegation boilerplate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from open_predictive_coder import (
    DelayLineConfig,
    DelayLineSubstrate,
    EchoStateSubstrate,
    FrozenReadoutExpert,
    HierarchicalFeatureView,
    HierarchicalSubstrate,
    HierarchicalSubstrateConfig,
    LinearMemoryConfig,
    LinearMemoryFeatureView,
    LinearMemorySubstrate,
    ReservoirConfig,
    RidgeReadout,
    ensure_tokens,
)
from open_predictive_coder.metrics import bits_per_byte_from_probabilities, softmax


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


def _distribution_summary(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    probabilities = probabilities / np.sum(probabilities)
    top2 = np.partition(probabilities, -2)[-2:]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return np.asarray(
        [
            float(np.max(probabilities)),
            float(top2[-1] - top2[-2]),
            entropy,
            float(np.sum(np.square(probabilities))),
        ],
        dtype=np.float64,
    )


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    return clipped / np.sum(clipped)


def _fit_experts(experts: Sequence[FrozenReadoutExpert], sequences: tuple[np.ndarray, ...]) -> dict[str, float]:
    return {expert.name: expert.fit(sequences).bits_per_byte for expert in experts}


def _component_batches(
    experts: Sequence[FrozenReadoutExpert],
    sequence: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    tokens = ensure_tokens(sequence)
    if tokens.size < 2:
        raise ValueError("sequence must contain at least two tokens")
    batches = [expert.sequence_probabilities(tokens)[0] for expert in experts]
    targets = tokens[1:].astype(np.int64, copy=False)
    return tokens, batches, targets


@dataclass(frozen=True)
class MixtureScore:
    tokens: int
    component_bits_per_byte: dict[str, float]
    mixed_bits_per_byte: float


@dataclass(frozen=True)
class ResidualScore:
    tokens: int
    base_bits_per_byte: float
    local_bits_per_byte: float
    corrected_bits_per_byte: float


class ConkerReplicaBase:
    model: object

    def fit(self, data: object):
        return self.model.fit(data)

    def score(self, sequence: object):
        return self.model.score(sequence)


class DelayWindowView:
    def __init__(self, substrate: DelayLineSubstrate):
        self.substrate = substrate
        self.feature_dim = substrate.state_dim + (2 * substrate.config.history_length)

    def encode(self, state: np.ndarray, previous_state: np.ndarray | None = None) -> np.ndarray:
        history = self.substrate.history_view(state)
        means = np.mean(history, axis=1)
        energies = np.mean(np.square(history), axis=1)
        return np.concatenate([state, means, energies])


class EchoStateView:
    def __init__(self, state_dim: int):
        self.feature_dim = state_dim + 3

    def encode(self, state: np.ndarray, previous_state: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if previous_state is None:
            delta = 0.0
        else:
            delta = float(np.mean(np.abs(state - np.asarray(previous_state, dtype=np.float64))))
        return np.concatenate(
            [
                state,
                np.asarray(
                    [
                        float(np.mean(state)),
                        float(np.mean(np.square(state))),
                        delta,
                    ],
                    dtype=np.float64,
                ),
            ]
        )


class HierarchicalStabilityView:
    def __init__(self, config: HierarchicalSubstrateConfig):
        self._view = HierarchicalFeatureView(config)
        self.feature_dim = 9

    def encode(self, state: np.ndarray, previous_state: np.ndarray | None = None) -> np.ndarray:
        summary = self._view.pooled_summary(state)
        fast, mid, slow = self._view.split(state)
        if previous_state is None:
            slow_drift = 0.0
        else:
            previous_slow = self._view.split(previous_state)[2]
            slow_drift = float(np.mean(np.abs(slow - previous_slow)))
        bridge_width = min(mid.shape[0], slow.shape[0])
        bridge = mid[:bridge_width] - np.tanh(slow[:bridge_width])
        return np.asarray(
            [
                float(summary.fast_mean[0]),
                float(summary.mid_mean[0]),
                float(summary.slow_mean[0]),
                float(summary.fast_energy),
                float(summary.mid_energy),
                float(summary.slow_energy),
                slow_drift,
                float(np.mean(np.abs(bridge))),
                float(np.max(np.abs(bridge))) if bridge.size else 0.0,
            ],
            dtype=np.float64,
        )


class ExpertMixtureModel:
    def __init__(self, experts: Sequence[FrozenReadoutExpert], alpha: float = 1e-3):
        if len(experts) < 2:
            raise ValueError("ExpertMixtureModel requires at least two experts")
        self.experts = tuple(experts)
        self.mixer = RidgeReadout(input_dim=4 * len(self.experts), output_dim=len(self.experts), alpha=alpha)

    def _mixer_features(self, component_probabilities: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([_distribution_summary(probabilities) for probabilities in component_probabilities])

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> dict[str, float]:
        sequences = _coerce_sequences(data)
        train_scores = _fit_experts(self.experts, sequences)
        feature_rows: list[np.ndarray] = []
        labels: list[int] = []
        for sequence in sequences:
            _, component_batches, targets = _component_batches(self.experts, sequence)
            for index, target in enumerate(targets):
                step_probabilities = [batch[index] for batch in component_batches]
                feature_rows.append(self._mixer_features(step_probabilities))
                labels.append(int(np.argmax([probabilities[int(target)] for probabilities in step_probabilities])))
        design = np.vstack(feature_rows)
        self.mixer.fit(design, np.asarray(labels, dtype=np.int64))
        return train_scores

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        component_probabilities = [expert.predict_proba(prompt) for expert in self.experts]
        features = self._mixer_features(component_probabilities)
        weights = softmax(self.mixer.logits(features[None, :]), axis=-1)[0]
        mixed = np.zeros_like(component_probabilities[0])
        for weight, probabilities in zip(weights, component_probabilities):
            mixed += weight * probabilities
        return _normalize(mixed)

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> MixtureScore:
        tokens, component_batches, targets = _component_batches(self.experts, sequence)
        component_bits = {
            expert.name: bits_per_byte_from_probabilities(batch, targets)
            for expert, batch in zip(self.experts, component_batches)
        }
        mixed_rows = []
        for index in range(targets.shape[0]):
            step_probabilities = [batch[index] for batch in component_batches]
            features = self._mixer_features(step_probabilities)
            weights = softmax(self.mixer.logits(features[None, :]), axis=-1)[0]
            mixed = np.zeros_like(step_probabilities[0])
            for weight, probabilities in zip(weights, step_probabilities):
                mixed += weight * probabilities
            mixed_rows.append(_normalize(mixed))
        return MixtureScore(
            tokens=int(tokens.size),
            component_bits_per_byte=component_bits,
            mixed_bits_per_byte=bits_per_byte_from_probabilities(np.vstack(mixed_rows), targets),
        )


class ResidualCorrectionModel:
    def __init__(
        self,
        *,
        base_expert: FrozenReadoutExpert,
        local_expert: FrozenReadoutExpert,
        vocabulary_size: int = 256,
        alpha: float = 1e-3,
    ):
        self.base_expert = base_expert
        self.local_expert = local_expert
        self.vocabulary_size = vocabulary_size
        self.selector = RidgeReadout(input_dim=12, output_dim=2, alpha=alpha)
        self._unigram = np.full(vocabulary_size, 1.0 / vocabulary_size, dtype=np.float64)

    def _selector_features(self, base_probs: np.ndarray, local_probs: np.ndarray) -> np.ndarray:
        delta = np.abs(local_probs - base_probs)
        return np.concatenate(
            [
                _distribution_summary(base_probs),
                _distribution_summary(local_probs),
                np.asarray(
                    [
                        float(np.max(delta)),
                        float(np.mean(delta)),
                        float(np.sum(delta * delta)),
                        float(np.argmax(local_probs) == np.argmax(base_probs)),
                    ],
                    dtype=np.float64,
                ),
            ]
        )

    def _blend(self, base_probs: np.ndarray, local_probs: np.ndarray, local_weight: float) -> np.ndarray:
        residual = np.maximum(local_probs - self._unigram, 0.0)
        corrected = np.clip(base_probs + (0.5 * local_weight * residual), 1e-12, None)
        return corrected / np.sum(corrected)

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> dict[str, float]:
        sequences = _coerce_sequences(data)
        train_scores = _fit_experts((self.base_expert, self.local_expert), sequences)
        unigram = np.zeros(self.vocabulary_size, dtype=np.float64)
        feature_rows: list[np.ndarray] = []
        labels: list[int] = []
        for sequence in sequences:
            tokens = ensure_tokens(sequence)
            unigram += np.bincount(tokens, minlength=self.vocabulary_size)
            base_batch, targets = self.base_expert.sequence_probabilities(tokens)
            local_batch, _ = self.local_expert.sequence_probabilities(tokens)
            for index, target in enumerate(targets):
                base_probs = base_batch[index]
                local_probs = local_batch[index]
                feature_rows.append(self._selector_features(base_probs, local_probs))
                labels.append(int(local_probs[int(target)] > base_probs[int(target)]))
        self._unigram = _normalize(unigram)
        self.selector.fit(np.vstack(feature_rows), np.asarray(labels, dtype=np.int64))
        return train_scores

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        base_probs = self.base_expert.predict_proba(prompt)
        local_probs = self.local_expert.predict_proba(prompt)
        selector = softmax(self.selector.logits(self._selector_features(base_probs, local_probs)[None, :]), axis=-1)[0]
        return self._blend(base_probs, local_probs, float(selector[1]))

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> ResidualScore:
        tokens = ensure_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")
        base_batch, targets = self.base_expert.sequence_probabilities(tokens)
        local_batch, _ = self.local_expert.sequence_probabilities(tokens)
        corrected_rows = []
        for index in range(targets.shape[0]):
            base_probs = base_batch[index]
            local_probs = local_batch[index]
            selector = softmax(
                self.selector.logits(self._selector_features(base_probs, local_probs)[None, :]),
                axis=-1,
            )[0]
            corrected_rows.append(self._blend(base_probs, local_probs, float(selector[1])))
        return ResidualScore(
            tokens=int(tokens.size),
            base_bits_per_byte=bits_per_byte_from_probabilities(base_batch, targets),
            local_bits_per_byte=bits_per_byte_from_probabilities(local_batch, targets),
            corrected_bits_per_byte=bits_per_byte_from_probabilities(np.vstack(corrected_rows), targets),
        )


def build_linear_memory_expert(
    *,
    name: str,
    embedding_dim: int = 12,
    decays: tuple[float, ...] = (0.35, 0.6, 0.82, 0.93),
    seed: int = 7,
    alpha: float = 1e-3,
) -> FrozenReadoutExpert:
    substrate = LinearMemorySubstrate(
        LinearMemoryConfig(
            embedding_dim=embedding_dim,
            decays=decays,
            seed=seed,
        )
    )
    view = LinearMemoryFeatureView(substrate)
    return FrozenReadoutExpert(
        name=name,
        substrate=substrate,
        feature_dim=view.feature_dim,
        vocabulary_size=substrate.config.vocabulary_size,
        feature_fn=view.encode,
        alpha=alpha,
    )


def build_delay_local_expert(
    *,
    name: str,
    history_length: int = 4,
    embedding_dim: int = 16,
    seed: int = 17,
    alpha: float = 1e-3,
) -> FrozenReadoutExpert:
    substrate = DelayLineSubstrate(
        DelayLineConfig(
            history_length=history_length,
            embedding_dim=embedding_dim,
            seed=seed,
        )
    )
    view = DelayWindowView(substrate)
    return FrozenReadoutExpert(
        name=name,
        substrate=substrate,
        feature_dim=view.feature_dim,
        vocabulary_size=substrate.config.vocabulary_size,
        feature_fn=view.encode,
        alpha=alpha,
    )


def build_echo_correction_expert(
    *,
    name: str,
    size: int = 64,
    seed: int = 23,
    alpha: float = 1e-3,
) -> FrozenReadoutExpert:
    substrate = EchoStateSubstrate(ReservoirConfig(size=size, connectivity=0.15, spectral_radius=0.92, leak=0.3, seed=seed))
    view = EchoStateView(substrate.state_dim)
    return FrozenReadoutExpert(
        name=name,
        substrate=substrate,
        feature_dim=view.feature_dim,
        vocabulary_size=256,
        feature_fn=view.encode,
        alpha=alpha,
    )


def build_hierarchical_stability_expert(
    *,
    name: str,
    seed: int = 29,
    alpha: float = 1e-3,
) -> FrozenReadoutExpert:
    config = HierarchicalSubstrateConfig(
        fast_size=24,
        mid_size=32,
        slow_size=40,
        fast_connectivity=0.18,
        mid_connectivity=0.12,
        slow_connectivity=0.08,
        fast_leak=0.45,
        mid_leak=0.22,
        slow_leak=0.1,
        upward_scale=0.08,
        slow_update_stride=2,
        seed=seed,
    )
    substrate = HierarchicalSubstrate(config)
    view = HierarchicalStabilityView(config)
    return FrozenReadoutExpert(
        name=name,
        substrate=substrate,
        feature_dim=view.feature_dim,
        vocabulary_size=config.vocabulary_size,
        feature_fn=view.encode,
        alpha=alpha,
    )


__all__ = [
    "ConkerReplicaBase",
    "ExpertMixtureModel",
    "FrozenReadoutExpert",
    "MixtureScore",
    "ResidualCorrectionModel",
    "ResidualScore",
    "build_delay_local_expert",
    "build_echo_correction_expert",
    "build_hierarchical_stability_expert",
    "build_linear_memory_expert",
]
