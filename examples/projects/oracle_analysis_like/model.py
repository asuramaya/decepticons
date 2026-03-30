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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.tools.diagnostics import summarize_alignment
from open_predictive_coder import (
    ControllerSummary,
    HierarchicalFeatureView,
    HierarchicalSubstrate,
    OpenPredictiveCoderConfig,
    RoutingConfig,
    SampledMultiscaleReadout,
    SampledReadoutBandConfig,
    SampledReadoutConfig,
    SummaryRouter,
    TrainModeConfig,
    ensure_tokens,
    hierarchical_small,
)


@dataclass(frozen=True)
class OracleAnalysisLikeConfig:
    model: OpenPredictiveCoderConfig = field(default_factory=hierarchical_small)
    train_mode: TrainModeConfig = field(
        default_factory=lambda: TrainModeConfig(
            state_mode="through_state",
            slow_update_stride=3,
            rollout_checkpoints=(8, 16, 24),
            rollout_checkpoint_stride=12,
        )
    )
    fast_sample_size: int = 8
    mid_sample_size: int = 8
    slow_sample_size: int = 12
    route_oracle_bias: float = 0.05
    route_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.model.substrate_kind != "hierarchical":
            raise ValueError("OracleAnalysisLikeConfig requires a hierarchical model config")
        hierarchical = self.model.hierarchical
        if self.fast_sample_size < 1 or self.fast_sample_size > hierarchical.fast_size:
            raise ValueError("fast_sample_size must lie within the fast bank size")
        if self.mid_sample_size < 1 or self.mid_sample_size > hierarchical.mid_size:
            raise ValueError("mid_sample_size must lie within the mid bank size")
        if self.slow_sample_size < 1 or self.slow_sample_size > hierarchical.slow_size:
            raise ValueError("slow_sample_size must lie within the slow bank size")
        if self.route_temperature <= 0.0:
            raise ValueError("route_temperature must be > 0")


@dataclass(frozen=True)
class OracleAnalysisPoint:
    checkpoint: int
    slow_update_active: bool
    route_names: tuple[str, ...]
    route_weights: np.ndarray
    selected_route: str
    alignment_pearson: float
    alignment_cosine: float
    alignment_mae: float
    alignment_rmse: float

    def __post_init__(self) -> None:
        route_weights = np.asarray(self.route_weights, dtype=np.float64).reshape(-1)
        if route_weights.size < 1:
            raise ValueError("OracleAnalysisPoint requires route weights")
        object.__setattr__(self, "route_weights", route_weights)


@dataclass(frozen=True)
class OracleAnalysisReport:
    tokens: int
    checkpoints: tuple[int, ...]
    points: tuple[OracleAnalysisPoint, ...]
    mean_alignment_pearson: float
    mean_alignment_cosine: float
    mean_alignment_mae: float
    oracle_preference_rate: float


class OracleAnalysisLikeModel:
    def __init__(self, config: OracleAnalysisLikeConfig | None = None):
        self.config = config or OracleAnalysisLikeConfig()
        hierarchical = self.config.model.hierarchical
        self.substrate = HierarchicalSubstrate(hierarchical)
        self.feature_view = HierarchicalFeatureView(hierarchical)
        self.sampled_readout = SampledMultiscaleReadout(
            SampledReadoutConfig(
                state_dim=hierarchical.state_dim,
                seed=hierarchical.seed + 31,
                bands=(
                    SampledReadoutBandConfig(
                        name="fast",
                        start=0,
                        stop=hierarchical.fast_size,
                        sample_count=self.config.fast_sample_size,
                        include_mean=True,
                        include_energy=True,
                        include_drift=True,
                    ),
                    SampledReadoutBandConfig(
                        name="mid",
                        start=hierarchical.fast_size,
                        stop=hierarchical.fast_size + hierarchical.mid_size,
                        sample_count=self.config.mid_sample_size,
                        include_mean=True,
                        include_energy=True,
                        include_drift=True,
                    ),
                    SampledReadoutBandConfig(
                        name="slow",
                        start=hierarchical.fast_size + hierarchical.mid_size,
                        stop=hierarchical.state_dim,
                        sample_count=self.config.slow_sample_size,
                        include_mean=True,
                        include_energy=True,
                        include_drift=True,
                    ),
                ),
            )
        )
        projection_weights = np.linspace(1.0, 0.25, num=self.sampled_readout.feature_dim, dtype=np.float64)
        self.router = SummaryRouter(
            RoutingConfig(
                mode="projection",
                projection_weights=tuple(float(value) for value in projection_weights),
                route_biases=(0.0, self.config.route_oracle_bias),
                temperature=self.config.route_temperature,
            )
        )

    def _coerce_sequence(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        return ensure_tokens(sequence)

    def _scan_states(self, tokens: np.ndarray) -> list[np.ndarray]:
        state = self.substrate.initial_state()
        states = [state.copy()]
        for token in tokens:
            state = self.substrate.step(state, int(token))
            states.append(state.copy())
        return states

    def _feature_for_state(self, state: np.ndarray, previous_state: np.ndarray | None) -> np.ndarray:
        if self.config.train_mode.uses_through_state:
            return self.sampled_readout.encode(state, previous_state=previous_state)
        return self.sampled_readout.encode(state, previous_state=None)

    def analyze(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> OracleAnalysisReport:
        tokens = self._coerce_sequence(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        total_steps = int(tokens.size)
        checkpoints = self.config.train_mode.resolve_rollout_checkpoints(total_steps)
        forward_states = self._scan_states(tokens)
        reverse_states = self._scan_states(tokens[::-1])

        points: list[OracleAnalysisPoint] = []
        pearsons: list[float] = []
        cosines: list[float] = []
        maes: list[float] = []
        oracle_selected = 0
        for checkpoint in checkpoints:
            suffix_len = total_steps - checkpoint
            causal_state = forward_states[checkpoint]
            causal_prev = forward_states[checkpoint - 1] if checkpoint > 0 else None
            oracle_state = reverse_states[suffix_len]
            oracle_prev = reverse_states[suffix_len - 1] if suffix_len > 0 else None

            causal_feature = self._feature_for_state(causal_state, causal_prev)
            oracle_feature = self._feature_for_state(oracle_state, oracle_prev)
            causal_summary = ControllerSummary(causal_feature, name="causal")
            oracle_summary = ControllerSummary(oracle_feature, name="oracle")
            decision = self.router.route((causal_summary, oracle_summary), names=("causal", "oracle"))
            alignment = summarize_alignment(causal_feature, oracle_feature, source_name="causal", target_name="oracle")
            slow_update_active = self.config.train_mode.should_update_slow(max(checkpoint - 1, 0))
            oracle_selected += int(decision.selected_index == 1)

            points.append(
                OracleAnalysisPoint(
                    checkpoint=checkpoint,
                    slow_update_active=slow_update_active,
                    route_names=decision.route_names,
                    route_weights=decision.weights.copy(),
                    selected_route=decision.route_names[decision.selected_index],
                    alignment_pearson=alignment.pearson,
                    alignment_cosine=alignment.cosine,
                    alignment_mae=alignment.mae,
                    alignment_rmse=alignment.rmse,
                )
            )
            pearsons.append(alignment.pearson)
            cosines.append(alignment.cosine)
            maes.append(alignment.mae)

        return OracleAnalysisReport(
            tokens=int(tokens.size),
            checkpoints=checkpoints,
            points=tuple(points),
            mean_alignment_pearson=float(np.mean(pearsons)),
            mean_alignment_cosine=float(np.mean(cosines)),
            mean_alignment_mae=float(np.mean(maes)),
            oracle_preference_rate=float(oracle_selected / max(len(points), 1)),
        )


__all__ = [
    "OracleAnalysisLikeConfig",
    "OracleAnalysisLikeModel",
    "OracleAnalysisPoint",
    "OracleAnalysisReport",
]
