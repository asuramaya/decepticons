from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
HERE = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from open_predictive_coder import (
    ControllerSummary,
    ControllerSummaryBuilder,
    ControllerSummaryConfig,
    FitReport,
    HierarchicalFeatureView,
    HierarchicalSubstrate,
    HormoneModulationConfig,
    HormoneModulator,
    HormoneState,
    OpenPredictiveCoderConfig,
    PathwayGateConfig,
    PathwayGateController,
    PathwayGateState,
    PathwayGateValues,
    RoutingConfig,
    RidgeReadout,
    SampledMultiscaleReadout,
    SampledReadoutBandConfig,
    SampledReadoutConfig,
    SequenceReport,
    ensure_tokens,
    hierarchical_small,
    SummaryRouter,
    TrainModeConfig,
)
from open_predictive_coder.metrics import bits_per_byte_from_logits, softmax
from predictor import HierarchicalPredictor, HierarchicalPredictorConfig


GateSource = Literal["slow", "surprise", "routed"]
AuxSource = Literal["prediction", "zeros", "random"]


@dataclass(frozen=True)
class CarvingMachineKernelConfig:
    model: OpenPredictiveCoderConfig = field(default_factory=hierarchical_small)
    use_pathway_gates: bool = True
    gate_fast_mid: bool = True
    gate_mid_slow: bool = True
    gate_source: GateSource = "slow"
    aux_source: AuxSource = "prediction"
    train_mode: TrainModeConfig = field(default_factory=TrainModeConfig)
    controller_view_dim: int = 32
    controller_width: int = 64
    fast_sample_size: int = 12
    mid_sample_size: int = 12
    slow_sample_size: int = 16
    prediction_l2: float = 1e-3
    hormone: HormoneModulationConfig = field(
        default_factory=lambda: HormoneModulationConfig(
            refresh_stride=1,
            hormone_count=2,
            output_indices=(0, 1),
            output_biases=(0.0, 0.0),
            output_scales=(1.0, 1.0),
        )
    )
    routing: RoutingConfig = field(
        default_factory=lambda: RoutingConfig(
            mode="projection",
            projection_weights=(1.0, 1.0),
            route_biases=(0.0, 0.0),
        )
    )
    pathway_gates: PathwayGateConfig = field(
        default_factory=lambda: PathwayGateConfig(
            refresh_stride=1,
            fast_to_mid_index=0,
            mid_to_slow_index=1,
            fast_to_mid_bias=1.5,
            fast_to_mid_scale=1.0,
            mid_to_slow_bias=1.5,
            mid_to_slow_scale=1.0,
        )
    )
    slow_mean_summary: ControllerSummaryConfig = field(
        default_factory=lambda: ControllerSummaryConfig(reduction="mean", normalize=False)
    )
    slow_abs_summary: ControllerSummaryConfig = field(
        default_factory=lambda: ControllerSummaryConfig(reduction="mean_abs", normalize=False)
    )
    surprise_abs_summary: ControllerSummaryConfig = field(
        default_factory=lambda: ControllerSummaryConfig(reduction="mean_abs", normalize=False)
    )
    surprise_max_summary: ControllerSummaryConfig = field(
        default_factory=lambda: ControllerSummaryConfig(reduction="max_abs", normalize=False)
    )

    def __post_init__(self) -> None:
        if self.model.substrate_kind != "hierarchical":
            raise ValueError("CarvingMachineKernelConfig requires a hierarchical model config")
        hierarchical = self.model.hierarchical
        if self.fast_sample_size < 1 or self.fast_sample_size > hierarchical.fast_size:
            raise ValueError("fast_sample_size must lie within the fast bank size")
        if self.mid_sample_size < 1 or self.mid_sample_size > hierarchical.mid_size:
            raise ValueError("mid_sample_size must lie within the mid bank size")
        if self.slow_sample_size < 1 or self.slow_sample_size > hierarchical.slow_size:
            raise ValueError("slow_sample_size must lie within the slow bank size")
        if self.controller_view_dim < 1:
            raise ValueError("controller_view_dim must be >= 1")
        if self.controller_width < 1:
            raise ValueError("controller_width must be >= 1")
        if self.prediction_l2 < 0.0:
            raise ValueError("prediction_l2 must be >= 0")
        if self.gate_source not in {"slow", "surprise", "routed"}:
            raise ValueError("gate_source must be 'slow', 'surprise', or 'routed'")


@dataclass(frozen=True)
class CarvingMachineTrace:
    features: np.ndarray
    targets: np.ndarray
    gates: np.ndarray
    hormones: np.ndarray
    routes: np.ndarray
    checkpoint_steps: tuple[int, ...]
    tokens: int


@dataclass
class _State:
    substrate_state: np.ndarray
    gate_state: PathwayGateState
    hormone_state: HormoneState
    route_index: int
    step_index: int


@dataclass(frozen=True)
class _Step:
    feature: np.ndarray
    gates: np.ndarray
    hormones: np.ndarray
    route_index: int
    prediction: np.ndarray
    surprise: np.ndarray


@dataclass(frozen=True)
class _TraceBundle:
    features: list[np.ndarray]
    targets: np.ndarray
    gates: list[np.ndarray]
    hormones: list[np.ndarray]
    routes: list[int]
    fast_rows: list[np.ndarray]
    mid_rows: list[np.ndarray]
    slow_rows: list[np.ndarray]


class CarvingMachineKernelAdapter:
    def __init__(self, config: CarvingMachineKernelConfig | None = None):
        self.config = config or CarvingMachineKernelConfig()
        self.substrate = HierarchicalSubstrate(self.config.model.hierarchical)
        self.feature_view = HierarchicalFeatureView(self.config.model.hierarchical)
        self.predictor = HierarchicalPredictor(
            fast_size=self.config.model.hierarchical.fast_size,
            mid_size=self.config.model.hierarchical.mid_size,
            slow_size=self.config.model.hierarchical.slow_size,
            config=HierarchicalPredictorConfig(
                controller_view_dim=self.config.controller_view_dim,
                controller_width=self.config.controller_width,
                prediction_l2=self.config.prediction_l2,
                seed=self.config.model.hierarchical.seed + 23,
            ),
        )
        self.gate_controller = PathwayGateController(self.config.pathway_gates)
        self.hormone_modulator = HormoneModulator(2, self.config.hormone)
        self.summary_router = SummaryRouter(self.config.routing)
        self.slow_mean_builder = ControllerSummaryBuilder(self.config.slow_mean_summary)
        self.slow_abs_builder = ControllerSummaryBuilder(self.config.slow_abs_summary)
        self.surprise_abs_builder = ControllerSummaryBuilder(self.config.surprise_abs_summary)
        self.surprise_max_builder = ControllerSummaryBuilder(self.config.surprise_max_summary)
        hierarchical = self.config.model.hierarchical
        self.sampled_readout = SampledMultiscaleReadout(
            SampledReadoutConfig(
                state_dim=hierarchical.fast_size + hierarchical.mid_size + hierarchical.slow_size,
                seed=hierarchical.seed + 10,
                bands=(
                    SampledReadoutBandConfig(
                        name="surprise",
                        start=0,
                        stop=hierarchical.fast_size,
                        sample_count=self.config.fast_sample_size,
                        include_mean=False,
                        include_energy=False,
                        include_drift=False,
                    ),
                    SampledReadoutBandConfig(
                        name="mid",
                        start=hierarchical.fast_size,
                        stop=hierarchical.fast_size + hierarchical.mid_size,
                        sample_count=self.config.mid_sample_size,
                        include_mean=False,
                        include_energy=False,
                        include_drift=False,
                    ),
                    SampledReadoutBandConfig(
                        name="slow",
                        start=hierarchical.fast_size + hierarchical.mid_size,
                        stop=hierarchical.fast_size + hierarchical.mid_size + hierarchical.slow_size,
                        sample_count=self.config.slow_sample_size,
                        include_mean=False,
                        include_energy=False,
                        include_drift=False,
                    ),
                ),
            )
        )
        self._surprise_band_dim = self.sampled_readout.config.bands[0].feature_dim
        self._surprise_sample_dim = self.sampled_readout.config.bands[0].resolved_sample_count
        self._surprise_sample_indices = self.sampled_readout.band_indices[0]
        rng = np.random.default_rng(hierarchical.seed + 11)
        self.aux_projection = rng.standard_normal((hierarchical.fast_size, self._surprise_sample_dim)).astype(np.float64)
        self.aux_projection /= np.sqrt(max(hierarchical.fast_size, 1))
        self.readout = RidgeReadout(
            input_dim=self.sampled_readout.feature_dim + self._surprise_sample_dim,
            output_dim=self.config.model.vocabulary_size,
            alpha=self.config.model.latent.readout_l2,
        )

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

    def _summary_width(self) -> int:
        return max(self.config.pathway_gates.fast_to_mid_index, self.config.pathway_gates.mid_to_slow_index) + 1

    def _zero_gate_state(self) -> PathwayGateState:
        base = self.gate_controller.initial_state()
        zeros = ControllerSummary(np.zeros(self._summary_width(), dtype=np.float64), name="init")
        return self.gate_controller.advance(base, zeros, step=-1)

    def _zero_hormone_state(self) -> HormoneState:
        base = self.hormone_modulator.initial_state()
        zeros = ControllerSummary(np.zeros(2, dtype=np.float64), name="init")
        return self.hormone_modulator.advance(base, zeros, step=-1)

    def _initial_state(self) -> _State:
        return _State(
            substrate_state=self.substrate.initial_state(),
            gate_state=self._zero_gate_state(),
            hormone_state=self._zero_hormone_state(),
            route_index=0,
            step_index=0,
        )

    def _split(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.feature_view.split(state)

    def _predict(self, slow: np.ndarray, mid: np.ndarray) -> np.ndarray:
        slow_batch = slow[None, :]
        mid_batch = mid[None, :]
        return self.predictor.predict(slow_batch, mid_batch)[0]

    def _aux_features(self, fast: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        fast = np.asarray(fast, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if self.config.aux_source == "prediction":
            return prediction[self._surprise_sample_indices]
        if self.config.aux_source == "zeros":
            return np.zeros((self._surprise_sample_dim,), dtype=np.float64)
        if self.config.aux_source == "random":
            return fast @ self.aux_projection
        raise ValueError(f"Unknown aux_source: {self.config.aux_source}")

    def _slow_summary(self, slow: np.ndarray) -> ControllerSummary:
        mean_value = float(self.slow_mean_builder.encode(slow).values[0])
        abs_value = float(self.slow_abs_builder.encode(slow).values[0])
        return ControllerSummary(np.asarray([mean_value, abs_value], dtype=np.float64), name="slow")

    def _surprise_summary(self, surprise: np.ndarray) -> ControllerSummary:
        abs_value = float(self.surprise_abs_builder.encode(surprise).values[0])
        max_value = float(self.surprise_max_builder.encode(surprise).values[0])
        return ControllerSummary(np.asarray([abs_value, max_value], dtype=np.float64), name="surprise")

    def _control_summary(self, state: _State, next_slow: np.ndarray, surprise: np.ndarray) -> tuple[ControllerSummary, int]:
        previous_slow = self._split(state.substrate_state)[2]
        slow_basis = next_slow if self.config.train_mode.uses_through_state else previous_slow
        slow_summary = self._slow_summary(slow_basis)
        surprise_summary = self._surprise_summary(surprise)
        if self.config.gate_source == "slow":
            return slow_summary, 0
        if self.config.gate_source == "surprise":
            return surprise_summary, 1
        decision = self.summary_router.route((slow_summary, surprise_summary), names=("slow", "surprise"))
        return (slow_summary if decision.selected_index == 0 else surprise_summary), int(decision.selected_index)

    def _gate_scales(self, hormone_state: HormoneState) -> np.ndarray:
        outputs = np.asarray(hormone_state.outputs, dtype=np.float64).reshape(-1)
        if outputs.size == 0:
            return np.ones(2, dtype=np.float64)
        if outputs.size == 1:
            outputs = np.repeat(outputs, 2)
        return 0.5 + 0.5 * outputs[:2]

    def _compose_readout_feature(
        self,
        fast: np.ndarray,
        mid: np.ndarray,
        slow: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fast = np.asarray(fast, dtype=np.float64)
        mid = np.asarray(mid, dtype=np.float64)
        slow = np.asarray(slow, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        surprise = fast * (1.0 - prediction)
        sampled_state = np.concatenate([surprise, mid, slow])
        sampled_feature = self.sampled_readout.encode(sampled_state)
        head = sampled_feature[: self._surprise_band_dim]
        tail = sampled_feature[self._surprise_band_dim :]
        aux = self._aux_features(fast, prediction)
        return np.concatenate([head, aux, tail]), surprise

    def _step_gates(self, state: _State) -> PathwayGateState:
        return state.gate_state

    def _apply_gate_mask(self, gate_values: PathwayGateValues, hormone_state: HormoneState) -> PathwayGateValues:
        gate_scales = self._gate_scales(hormone_state)
        return PathwayGateValues(
            fast_to_mid=(gate_values.fast_to_mid if self.config.gate_fast_mid else 1.0) * float(gate_scales[0]),
            mid_to_slow=(gate_values.mid_to_slow if self.config.gate_mid_slow else 1.0) * float(gate_scales[1]),
            step=gate_values.step,
            refreshed=gate_values.refreshed,
            summary_name=gate_values.summary_name,
        )

    def _advance_control_state(
        self,
        state: _State,
        next_slow: np.ndarray,
        surprise: np.ndarray,
    ) -> tuple[PathwayGateState, HormoneState, int]:
        summary, route_index = self._control_summary(state, next_slow, surprise)
        next_gate_state = state.gate_state
        if self.config.use_pathway_gates:
            next_gate_state = self.gate_controller.advance(state.gate_state, summary, step=state.step_index)
        next_hormone_state = self.hormone_modulator.advance(state.hormone_state, summary, step=state.step_index)
        return next_gate_state, next_hormone_state, route_index

    def _gated_step(self, state: np.ndarray, token: int, gate_values: PathwayGateValues, *, step_index: int) -> np.ndarray:
        token_id = int(token)
        if token_id < 0 or token_id >= self.config.model.vocabulary_size:
            raise ValueError("token out of range")

        fast_state, mid_state, slow_state = self._split(state)
        cfg = self.substrate.config

        fast_drive = self.substrate.fast_recurrent @ fast_state + self.substrate.fast_input[:, token_id]
        next_fast = (1.0 - cfg.fast_leak) * fast_state + cfg.fast_leak * np.tanh(fast_drive)

        mid_drive = self.substrate.mid_recurrent @ mid_state + gate_values.fast_to_mid * (self.substrate.fast_up @ next_fast)
        next_mid = (1.0 - cfg.mid_leak) * mid_state + cfg.mid_leak * np.tanh(mid_drive)

        slow_active = (
            self.config.train_mode.should_update_slow(step_index)
            and (cfg.slow_update_stride == 1 or ((self.substrate._step_index + 1) % cfg.slow_update_stride == 0))
        )
        if slow_active:
            slow_drive = self.substrate.slow_recurrent @ slow_state + gate_values.mid_to_slow * (self.substrate.mid_up @ next_mid)
            next_slow = (1.0 - cfg.slow_leak) * slow_state + cfg.slow_leak * np.tanh(slow_drive)
        else:
            next_slow = slow_state.copy()

        self.substrate._step_index += 1
        return np.concatenate([next_fast, next_mid, next_slow])

    def _simulate(
        self,
        tokens: np.ndarray,
        *,
        collect_features: bool,
    ) -> _TraceBundle:
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        state = self._initial_state()
        features: list[np.ndarray] = []
        gates: list[np.ndarray] = []
        hormones: list[np.ndarray] = []
        routes: list[int] = []
        fast_rows: list[np.ndarray] = []
        mid_rows: list[np.ndarray] = []
        slow_rows: list[np.ndarray] = []
        for token in tokens[:-1]:
            gate_state = self._step_gates(state)
            gate_values = self._apply_gate_mask(
                gate_state.values
                if self.config.use_pathway_gates
                else PathwayGateValues(
                    fast_to_mid=1.0,
                    mid_to_slow=1.0,
                    step=state.step_index,
                    refreshed=False,
                    summary_name="disabled",
                ),
                state.hormone_state,
            )
            next_state = self._gated_step(state.substrate_state, int(token), gate_values, step_index=state.step_index)
            next_fast, next_mid, next_slow = self._split(next_state)
            prediction = self._predict(next_slow, next_mid)
            feature, surprise = self._compose_readout_feature(next_fast, next_mid, next_slow, prediction)
            next_gate_state, next_hormone_state, route_index = self._advance_control_state(state, next_slow, surprise)
            if collect_features:
                features.append(feature)
                gates.append(np.asarray([gate_values.fast_to_mid, gate_values.mid_to_slow], dtype=np.float64))
                hormones.append(next_hormone_state.outputs.copy())
                routes.append(route_index)
            fast_rows.append(next_fast)
            mid_rows.append(next_mid)
            slow_rows.append(next_slow)
            state.substrate_state = next_state
            state.gate_state = next_gate_state
            state.hormone_state = next_hormone_state
            state.route_index = route_index
            state.step_index += 1
        return _TraceBundle(
            features=features,
            targets=tokens[1:].astype(np.int64, copy=False),
            gates=gates,
            hormones=hormones,
            routes=routes,
            fast_rows=fast_rows,
            mid_rows=mid_rows,
            slow_rows=slow_rows,
        )

    def _advance(self, state: _State, token: int) -> _Step:
        gate_state = self._step_gates(state)
        gate_values = self._apply_gate_mask(
            gate_state.values
            if self.config.use_pathway_gates
            else PathwayGateValues(
                fast_to_mid=1.0,
                mid_to_slow=1.0,
                step=state.step_index,
                refreshed=False,
                summary_name="disabled",
            ),
            state.hormone_state,
        )
        next_state = self._gated_step(state.substrate_state, token, gate_values, step_index=state.step_index)
        next_fast, next_mid, next_slow = self._split(next_state)
        prediction = self._predict(next_slow, next_mid)
        feature, surprise = self._compose_readout_feature(next_fast, next_mid, next_slow, prediction)
        next_gate_state, next_hormone_state, route_index = self._advance_control_state(state, next_slow, surprise)
        state.substrate_state = next_state
        state.gate_state = next_gate_state
        state.hormone_state = next_hormone_state
        state.route_index = route_index
        state.step_index += 1
        return _Step(
            feature=feature,
            gates=np.asarray([gate_values.fast_to_mid, gate_values.mid_to_slow], dtype=np.float64),
            hormones=next_hormone_state.outputs.copy(),
            route_index=route_index,
            prediction=prediction,
            surprise=surprise,
        )

    def trace(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> CarvingMachineTrace:
        tokens = ensure_tokens(sequence)
        bundle = self._simulate(tokens, collect_features=True)
        return CarvingMachineTrace(
            features=np.vstack(bundle.features),
            targets=bundle.targets,
            gates=np.vstack(bundle.gates),
            hormones=np.vstack(bundle.hormones),
            routes=np.asarray(bundle.routes, dtype=np.int64),
            checkpoint_steps=self.config.train_mode.resolve_rollout_checkpoints(int(tokens.size) - 1),
            tokens=int(tokens.size),
        )

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> FitReport:
        sequences = self._coerce_sequences(data)
        predictor_slow_batches = []
        predictor_mid_batches = []
        predictor_fast_batches = []
        feature_batches = []
        target_batches = []
        total_tokens = 0
        for sequence in sequences:
            bundle = self._simulate(sequence, collect_features=False)
            predictor_slow_batches.append(np.vstack(bundle.slow_rows))
            predictor_mid_batches.append(np.vstack(bundle.mid_rows))
            predictor_fast_batches.append(np.vstack(bundle.fast_rows))
            total_tokens += int(ensure_tokens(sequence).size)

        self.predictor.fit(
            np.concatenate(predictor_slow_batches, axis=0),
            np.concatenate(predictor_mid_batches, axis=0),
            np.concatenate(predictor_fast_batches, axis=0),
        )

        for sequence in sequences:
            trace = self.trace(sequence)
            feature_batches.append(trace.features)
            target_batches.append(trace.targets)
        design = np.concatenate(feature_batches, axis=0)
        targets = np.concatenate(target_batches, axis=0)
        self.readout.fit(design, targets)
        logits = self.readout.logits(design)
        return FitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            patches=max(total_tokens - len(sequences), 0),
            mean_patch_size=1.0,
            compression_ratio=1.0,
            train_bits_per_byte=bits_per_byte_from_logits(logits, targets),
        )

    def score(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> SequenceReport:
        trace = self.trace(sequence)
        logits = self.readout.logits(trace.features)
        return SequenceReport(
            tokens=trace.tokens,
            patches=max(trace.tokens - 1, 0),
            mean_patch_size=1.0,
            compression_ratio=1.0,
            bits_per_byte=bits_per_byte_from_logits(logits, trace.targets),
        )

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")
        state = self._initial_state()
        step: _Step | None = None
        for token in tokens:
            step = self._advance(state, int(token))
        assert step is not None
        return self.readout.probabilities(step.feature[None, :])[0]

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
        state = self._initial_state()
        step: _Step | None = None
        output = tokens.astype(np.uint8, copy=True).tolist()
        for token in tokens:
            step = self._advance(state, int(token))
        assert step is not None

        for _ in range(steps):
            logits = self.readout.logits(step.feature[None, :])[0]
            if greedy:
                next_token = int(np.argmax(logits))
            else:
                probs = softmax((logits / temperature)[None, :], axis=-1)[0]
                next_token = int(rng.choice(self.config.model.vocabulary_size, p=probs))
            output.append(next_token)
            step = self._advance(state, next_token)
        return np.asarray(output, dtype=np.uint8)


__all__ = [
    "CarvingMachineKernelAdapter",
    "CarvingMachineKernelConfig",
    "CarvingMachineTrace",
]
