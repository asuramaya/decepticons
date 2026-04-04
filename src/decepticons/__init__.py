"""Public API for the decepticons kernel.

This package is the shared mechanism layer for downstream systems such as
Chronohorn: reusable substrates, control primitives, memory blocks, readouts,
lightweight runtime helpers, and backend-neutral family metadata.

The full package map and layer boundary are documented in `docs/architecture.md`.
"""

# Foundation and configuration surfaces.
from decepticons.memory_protocol import (
    MEMORY_KINDS,
    MemoryAttachmentConfig,
)

from .artifacts import (
    ArtifactAccounting,
    ArtifactMetadata,
    ReplaySpan,
    coerce_artifact_metadata,
    make_artifact_accounting,
    make_replay_span,
)
from .artifacts_audits import ArtifactAuditRecord, ArtifactAuditSummary, audit_artifact, summarize_artifact_audits

# Memory, latent, and feature-view primitives.
from .bidirectional_context import (
    BidirectionalContextConfig,
    BidirectionalContextLeaveOneOutStats,
    BidirectionalContextNeighborhood,
    BidirectionalContextProbe,
    BidirectionalContextStats,
)
from .bridge_export import (
    BridgeExportAdapter,
    BridgeExportConfig,
    BridgeExportFitReport,
    BridgeExportReport,
)
from .bridge_features import BridgeFeatureArrays, BridgeFeatureConfig, bridge_feature_arrays
from .causal_bank import (
    CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED,
    CAUSAL_BANK_FAMILY,
    CAUSAL_BANK_FAMILY_ID,
    CAUSAL_BANK_INPUT_PROJ_SCHEMES,
    CAUSAL_BANK_OSCILLATORY_SCHEDULES,
    CAUSAL_BANK_READOUT_KINDS,
    CAUSAL_BANK_VARIANTS,
    CausalBankConfig,
    CausalBankFamilySpec,
)
from .causal_bank import (
    apply_variant as apply_causal_bank_variant,
)
from .causal_bank import (
    build_linear_bank as build_causal_bank_linear_bank,
)
from .causal_bank import (
    learnable_substrate_keys as learnable_causal_bank_substrate_keys,
)
from .causal_bank import (
    osc_pair_count as causal_bank_osc_pair_count,
)
from .causal_bank import (
    scale_config as scale_causal_bank_config,
)
from .causal_bank import (
    validate_config as validate_causal_bank_config,
)

# Concrete adapter surface.
from .causal_predictive import CausalPredictiveAdapter, CausalPredictiveFitReport, CausalPredictiveScore
from .codecs import ByteCodec, ensure_byte_tokens, ensure_tokens
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
    OscillatoryMemoryConfig,
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

# Readouts, experts, datasets, and runtime surfaces.
from .datasets import ByteSequenceDataset
from .eval import NextStepScore, RolloutEvaluation, RolloutMode, evaluate_rollout, score_next_step
from .exact_context import (
    ExactContextConfig,
    ExactContextFitReport,
    ExactContextMemory,
    ExactContextPrediction,
    SupportBlend,
    SupportMixConfig,
    SupportWeightedMixer,
)
from .experts import ExpertFitReport, ExpertScore, FrozenReadoutExpert

# Substrates, factories, and presets.
from .factories import (
    create_delay_line_substrate,
    create_echo_state_substrate,
    create_hierarchical_substrate,
    create_mixed_memory_substrate,
    create_oscillatory_memory_substrate,
    create_substrate,
    create_substrate_for_model,
)
from .gating import PathwayGateConfig, PathwayGateController, PathwayGateState, PathwayGateValues
from .hierarchical_views import HierarchicalFeatureView, HierarchicalSummary
from .latents import LatentCommitter, LatentObservation, LatentState
from .learned_segmentation import (
    BoundaryDecision,
    BoundaryFeatures,
    BoundaryScorerConfig,
    LearnedBoundaryScorer,
    LearnedSegmenter,
    LearnedSegmenterConfig,
)
from .linear_views import LinearMemoryFeatureView
from .memory_cache import ExactContextCache, MemoryPredictionRecord, MemoryPredictionSummary, StatisticalBackoffCache
from .metrics import (
    bits_per_byte_from_logits,
    bits_per_byte_from_probabilities,
    bits_per_token_from_logits,
    bits_per_token_from_probabilities,
)
from .model import ByteLatentPredictiveCoder, OpenPredictiveCoder
from .modulation import HormoneModulationConfig, HormoneModulator, HormoneState
from .ngram_memory import NgramMemory, NgramMemoryConfig, NgramMemoryReport
from .noncausal_reconstructive import (
    NoncausalReconstructiveAdapter,
    NoncausalReconstructiveConfig,
    NoncausalReconstructiveFitReport,
    NoncausalReconstructiveReport,
    NoncausalReconstructiveTrace,
)
from .online_memory import OnlineCausalMemory, OnlineMemoryConfig
from .oracle_analysis import (
    OracleAnalysisAdapter,
    OracleAnalysisConfig,
    OracleAnalysisFitReport,
    OracleAnalysisPoint,
    OracleAnalysisReport,
)
from .patch_latent_blocks import (
    GlobalLocalBridge,
    GlobalLocalBridgeConfig,
    LocalByteEncoder,
    LocalByteEncoderConfig,
    PatchPooler,
    PatchPoolerConfig,
)
from .predictive_surprise import PredictionState, PredictiveSurpriseConfig, PredictiveSurpriseController, SummaryMode
from .presets import delay_small, echo_state_small, hierarchical_small, mixed_memory_small
from .probability_diagnostics import (
    ProbabilityDiagnostics,
    ProbabilityDiagnosticsConfig,
    normalized_entropy,
    overlap_mass,
    probability_diagnostics,
    shared_top_k_mass,
    top1_agreement,
    top1_peak,
    top2_margin,
    top_k_mass,
)
from .readouts import RidgeReadout
from .routing import RoutingConfig, RoutingDecision, RoutingMode, SummaryRouter
from .runtime import (
    CausalFitReport,
    CausalSequenceReport,
    CausalTrace,
    FitReport,
    SequenceReport,
    SequenceTrace,
    tag_metadata,
)
from .sampled_readout import SampledBandSummary, SampledMultiscaleReadout
from .segmenters import AdaptiveSegmenter, SegmentStats
from .span_selection import ScoredSpan, SpanSelectionConfig, replay_spans_from_scores, select_scored_spans
from .statistical_backoff import (
    StatisticalBackoffConfig,
    StatisticalBackoffFitReport,
    StatisticalBackoffMemory,
    StatisticalBackoffPrediction,
    StatisticalBackoffScore,
    StatisticalBackoffTrace,
)
from .substrates import (
    DelayLineSubstrate,
    EchoStateSubstrate,
    HierarchicalSubstrate,
    LinearMemorySubstrate,
    MixedMemorySubstrate,
    OscillatoryMemorySubstrate,
    TokenSubstrate,
)
from .teacher_export import TeacherExportAdapter, TeacherExportConfig, TeacherExportRecord, TeacherExportReport
from .train_eval import (
    DatasetEvaluation,
    RolloutCheckpoint,
    RolloutCurve,
    RolloutCurveEvaluation,
    RolloutCurveMode,
    RolloutCurvePoint,
    TransferEvaluation,
    TransferProbeReport,
    evaluate_dataset,
    evaluate_rollout_curve,
    evaluate_transfer_probe,
    score_dataset,
)
from .train_modes import TrainModeConfig, TrainStateMode
from .views import ByteLatentFeatureView

__all__ = [
    "AdaptiveSegmenter",
    "ArtifactAccounting",
    "ArtifactAuditRecord",
    "ArtifactAuditSummary",
    "ArtifactMetadata",
    "audit_artifact",
    "apply_causal_bank_variant",
    "coerce_artifact_metadata",
    "build_causal_bank_linear_bank",
    "CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED",
    "CAUSAL_BANK_FAMILY",
    "CAUSAL_BANK_FAMILY_ID",
    "CAUSAL_BANK_INPUT_PROJ_SCHEMES",
    "CAUSAL_BANK_OSCILLATORY_SCHEDULES",
    "CAUSAL_BANK_READOUT_KINDS",
    "CAUSAL_BANK_VARIANTS",
    "causal_bank_osc_pair_count",
    "learnable_causal_bank_substrate_keys",
    "CausalBankConfig",
    "CausalBankFamilySpec",
    "ByteCodec",
    "ensure_byte_tokens",
    "ByteLatentFeatureView",
    "ByteLatentPredictiveCoder",
    "ByteLatentPredictiveCoderConfig",
    "ByteSequenceDataset",
    "bits_per_byte_from_logits",
    "bits_per_byte_from_probabilities",
    "bits_per_token_from_logits",
    "bits_per_token_from_probabilities",
    "BidirectionalContextConfig",
    "BidirectionalContextLeaveOneOutStats",
    "BidirectionalContextNeighborhood",
    "BidirectionalContextProbe",
    "BidirectionalContextStats",
    "BridgeExportAdapter",
    "BridgeExportConfig",
    "BridgeExportFitReport",
    "BridgeExportReport",
    "BoundaryDecision",
    "BoundaryFeatures",
    "BoundaryScorerConfig",
    "BridgeFeatureArrays",
    "BridgeFeatureConfig",
    "bridge_feature_arrays",
    "CausalFitReport",
    "CausalPredictiveAdapter",
    "CausalPredictiveFitReport",
    "CausalPredictiveScore",
    "CausalSequenceReport",
    "CausalTrace",
    "ControllerSummary",
    "ControllerSummaryBuilder",
    "ControllerSummaryConfig",
    "scale_causal_bank_config",
    "SubstrateKind",
    "create_delay_line_substrate",
    "create_echo_state_substrate",
    "create_hierarchical_substrate",
    "create_mixed_memory_substrate",
    "create_oscillatory_memory_substrate",
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
    "GlobalLocalBridge",
    "GlobalLocalBridgeConfig",
    "LatentConfig",
    "LatentCommitter",
    "LatentControllerConfig",
    "LatentObservation",
    "LatentState",
    "LearnedBoundaryScorer",
    "LearnedSegmenter",
    "LearnedSegmenterConfig",
    "LinearMemoryConfig",
    "LinearMemoryFeatureView",
    "LinearMemorySubstrate",
    "LocalByteEncoder",
    "LocalByteEncoderConfig",
    "make_artifact_accounting",
    "make_replay_span",
    "MEMORY_KINDS",
    "MemoryAttachmentConfig",
    "MemoryPredictionRecord",
    "MemoryPredictionSummary",
    "MemoryMergeMode",
    "MixedMemoryConfig",
    "MixedMemorySubstrate",
    "mixed_memory_small",
    "NextStepScore",
    "NgramMemory",
    "NgramMemoryConfig",
    "NgramMemoryReport",
    "OnlineCausalMemory",
    "OnlineMemoryConfig",
    "NoncausalReconstructiveAdapter",
    "NoncausalReconstructiveConfig",
    "NoncausalReconstructiveFitReport",
    "NoncausalReconstructiveReport",
    "NoncausalReconstructiveTrace",
    "OracleAnalysisAdapter",
    "OracleAnalysisConfig",
    "OracleAnalysisFitReport",
    "OracleAnalysisPoint",
    "OracleAnalysisReport",
    "OpenPredictiveCoder",
    "OpenPredictiveCoderConfig",
    "OscillatoryMemoryConfig",
    "OscillatoryMemorySubstrate",
    "PathwayGateConfig",
    "PathwayGateController",
    "PathwayGateState",
    "PathwayGateValues",
    "PatchPooler",
    "PatchPoolerConfig",
    "ProbabilityDiagnostics",
    "ProbabilityDiagnosticsConfig",
    "probability_diagnostics",
    "normalized_entropy",
    "overlap_mass",
    "shared_top_k_mass",
    "top1_agreement",
    "top1_peak",
    "top2_margin",
    "top_k_mass",
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
    "ScoredSpan",
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
    "StatisticalBackoffConfig",
    "StatisticalBackoffFitReport",
    "StatisticalBackoffMemory",
    "StatisticalBackoffPrediction",
    "StatisticalBackoffScore",
    "StatisticalBackoffTrace",
    "StatisticalBackoffCache",
    "tag_metadata",
    "TeacherExportAdapter",
    "TeacherExportConfig",
    "TeacherExportRecord",
    "TeacherExportReport",
    "TransferProbeReport",
    "TransferEvaluation",
    "TokenSubstrate",
    "ExactContextCache",
    "ensure_tokens",
    "evaluate_dataset",
    "evaluate_rollout_curve",
    "evaluate_transfer_probe",
    "score_dataset",
    "select_scored_spans",
    "SpanSelectionConfig",
    "stack_summaries",
    "SupportBlend",
    "SupportMixConfig",
    "SupportWeightedMixer",
    "summarize_artifact_audits",
    "SummaryMode",
    "SummaryRouter",
    "TrainModeConfig",
    "TrainStateMode",
    "validate_causal_bank_config",
    "replay_spans_from_scores",
]
