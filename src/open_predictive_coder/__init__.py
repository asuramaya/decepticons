"""Public API for the open_predictive_coder kernel.

The package is intentionally organized in layers:

1. foundational types and configs
2. reusable kernel primitives for substrates, control, memory, views, readouts, and runtime
3. the first concrete adapter (`ByteLatentPredictiveCoder`)

The full package map and layer boundary are documented in `docs/architecture.md`.
"""

# Foundation and configuration surfaces.
from .artifacts import ArtifactAccounting, ArtifactMetadata, ReplaySpan
from .codecs import ByteCodec, ensure_tokens
from .config import (
    ByteLatentPredictiveCoderConfig,
    DelayLineConfig,
    HierarchicalSubstrateConfig,
    LatentConfig,
    LatentControllerConfig,
    LinearMemoryConfig,
    MemoryMergeMode,
    MixedMemoryConfig,
    OpenPredictiveCoderConfig,
    ReservoirConfig,
    ReservoirTopology,
    SampledReadoutBandConfig,
    SampledReadoutConfig,
    SegmenterConfig,
    SegmenterMode,
    SubstrateKind,
)

# Control, routing, modulation, and predictive side channels.
from .control import (
    ControllerSummary,
    ControllerSummaryBuilder,
    ControllerSummaryConfig,
    stack_summaries,
)
from .controllers import (
    PredictiveController,
    PredictiveObservation,
    PredictiveState,
)
from .gating import PathwayGateConfig, PathwayGateController, PathwayGateState, PathwayGateValues
from .modulation import HormoneModulationConfig, HormoneModulator, HormoneState
from .predictive_surprise import PredictionState, PredictiveSurpriseConfig, PredictiveSurpriseController, SummaryMode
from .routing import RoutingConfig, RoutingDecision, RoutingMode, SummaryRouter

# Memory, latent, and feature-view primitives.
from .exact_context import (
    ExactContextConfig,
    ExactContextFitReport,
    ExactContextMemory,
    ExactContextPrediction,
    SupportBlend,
    SupportMixConfig,
    SupportWeightedMixer,
)
from .latents import LatentCommitter, LatentObservation, LatentState
from .hierarchical_views import HierarchicalFeatureView, HierarchicalSummary
from .linear_views import LinearMemoryFeatureView
from .sampled_readout import SampledBandSummary, SampledMultiscaleReadout
from .views import ByteLatentFeatureView

# Substrates, factories, and presets.
from .factories import (
    create_delay_line_substrate,
    create_echo_state_substrate,
    create_hierarchical_substrate,
    create_mixed_memory_substrate,
    create_substrate,
    create_substrate_for_model,
)
from .presets import delay_small, echo_state_small, hierarchical_small, mixed_memory_small
from .segmenters import AdaptiveSegmenter, SegmentStats
from .substrates import (
    DelayLineSubstrate,
    EchoStateSubstrate,
    HierarchicalSubstrate,
    LinearMemorySubstrate,
    MixedMemorySubstrate,
    TokenSubstrate,
)

# Readouts, experts, datasets, and runtime surfaces.
from .datasets import ByteSequenceDataset
from .eval import NextStepScore, RolloutEvaluation, RolloutMode, evaluate_rollout, score_next_step
from .experts import ExpertFitReport, ExpertScore, FrozenReadoutExpert
from .metrics import bits_per_byte_from_logits, bits_per_byte_from_probabilities
from .readouts import RidgeReadout
from .runtime import FitReport, SequenceReport, SequenceTrace
from .train_modes import TrainModeConfig, TrainStateMode
from .train_eval import (
    DatasetEvaluation,
    RolloutCheckpoint,
    RolloutCurve,
    RolloutCurveMode,
    RolloutCurveEvaluation,
    RolloutCurvePoint,
    TransferEvaluation,
    TransferProbeReport,
    evaluate_dataset,
    evaluate_rollout_curve,
    evaluate_transfer_probe,
    score_dataset,
)

# Concrete adapter surface.
from .model import ByteLatentPredictiveCoder, OpenPredictiveCoder

__all__ = [
    "AdaptiveSegmenter",
    "ArtifactAccounting",
    "ArtifactMetadata",
    "ByteCodec",
    "ByteLatentFeatureView",
    "ByteLatentPredictiveCoder",
    "ByteLatentPredictiveCoderConfig",
    "ByteSequenceDataset",
    "bits_per_byte_from_logits",
    "bits_per_byte_from_probabilities",
    "ControllerSummary",
    "ControllerSummaryBuilder",
    "ControllerSummaryConfig",
    "SubstrateKind",
    "create_delay_line_substrate",
    "create_echo_state_substrate",
    "create_hierarchical_substrate",
    "create_mixed_memory_substrate",
    "create_substrate",
    "create_substrate_for_model",
    "DelayLineConfig",
    "DelayLineSubstrate",
    "delay_small",
    "DatasetEvaluation",
    "EchoStateSubstrate",
    "echo_state_small",
    "ExpertFitReport",
    "ExpertScore",
    "ExactContextConfig",
    "ExactContextFitReport",
    "ExactContextMemory",
    "ExactContextPrediction",
    "evaluate_rollout",
    "FitReport",
    "FrozenReadoutExpert",
    "HormoneModulationConfig",
    "HormoneModulator",
    "HormoneState",
    "HierarchicalFeatureView",
    "HierarchicalSummary",
    "HierarchicalSubstrate",
    "HierarchicalSubstrateConfig",
    "hierarchical_small",
    "LatentConfig",
    "LatentCommitter",
    "LatentControllerConfig",
    "LatentObservation",
    "LatentState",
    "LinearMemoryConfig",
    "LinearMemoryFeatureView",
    "LinearMemorySubstrate",
    "MemoryMergeMode",
    "MixedMemoryConfig",
    "MixedMemorySubstrate",
    "mixed_memory_small",
    "NextStepScore",
    "OpenPredictiveCoder",
    "OpenPredictiveCoderConfig",
    "PathwayGateConfig",
    "PathwayGateController",
    "PathwayGateState",
    "PathwayGateValues",
    "PredictiveController",
    "PredictiveObservation",
    "PredictiveState",
    "PredictionState",
    "PredictiveSurpriseConfig",
    "PredictiveSurpriseController",
    "ReplaySpan",
    "ReservoirConfig",
    "ReservoirTopology",
    "SampledBandSummary",
    "SampledMultiscaleReadout",
    "SampledReadoutBandConfig",
    "SampledReadoutConfig",
    "RidgeReadout",
    "RolloutCurveEvaluation",
    "RolloutCurve",
    "RolloutCheckpoint",
    "RolloutCurveMode",
    "RolloutCurvePoint",
    "RolloutEvaluation",
    "RolloutMode",
    "RoutingConfig",
    "RoutingDecision",
    "RoutingMode",
    "SegmenterConfig",
    "SegmenterMode",
    "SegmentStats",
    "SequenceReport",
    "SequenceTrace",
    "score_next_step",
    "TransferProbeReport",
    "TransferEvaluation",
    "TokenSubstrate",
    "ensure_tokens",
    "evaluate_dataset",
    "evaluate_rollout_curve",
    "evaluate_transfer_probe",
    "score_dataset",
    "stack_summaries",
    "SupportBlend",
    "SupportMixConfig",
    "SupportWeightedMixer",
    "SummaryMode",
    "SummaryRouter",
    "TrainModeConfig",
    "TrainStateMode",
]
