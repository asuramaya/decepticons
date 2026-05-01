"""Public API for the decepticons kernel.

A backend-neutral kernel of predictive primitives — substrates, memory, gating,
routing, readouts — that downstream systems combine into trained models without
forking the kernel itself. The shared mechanism layer for predictive descendants.

The full package map and layer boundary are documented in `docs/architecture.md`.
The boundary against the runtime descendant is documented in
`docs/chronohorn_boundary.md`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("decepticons")
except PackageNotFoundError:
    # Running from a source tree without `pip install -e .`.
    __version__ = "0.0.0+unknown"

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
    "__version__",
    "AdaptiveSegmenter",
    "apply_causal_bank_variant",
    "ArtifactAccounting",
    "ArtifactAuditRecord",
    "ArtifactAuditSummary",
    "ArtifactMetadata",
    "audit_artifact",
    "BidirectionalContextConfig",
    "BidirectionalContextLeaveOneOutStats",
    "BidirectionalContextNeighborhood",
    "BidirectionalContextProbe",
    "BidirectionalContextStats",
    "bits_per_byte_from_logits",
    "bits_per_byte_from_probabilities",
    "bits_per_token_from_logits",
    "bits_per_token_from_probabilities",
    "BoundaryDecision",
    "BoundaryFeatures",
    "BoundaryScorerConfig",
    "bridge_feature_arrays",
    "BridgeExportAdapter",
    "BridgeExportConfig",
    "BridgeExportFitReport",
    "BridgeExportReport",
    "BridgeFeatureArrays",
    "BridgeFeatureConfig",
    "build_causal_bank_linear_bank",
    "ByteCodec",
    "ByteLatentFeatureView",
    "ByteLatentPredictiveCoder",
    "ByteLatentPredictiveCoderConfig",
    "ByteSequenceDataset",
    "CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED",
    "CAUSAL_BANK_FAMILY",
    "CAUSAL_BANK_FAMILY_ID",
    "CAUSAL_BANK_INPUT_PROJ_SCHEMES",
    "causal_bank_osc_pair_count",
    "CAUSAL_BANK_OSCILLATORY_SCHEDULES",
    "CAUSAL_BANK_READOUT_KINDS",
    "CAUSAL_BANK_VARIANTS",
    "CausalBankConfig",
    "CausalBankFamilySpec",
    "CausalFitReport",
    "CausalPredictiveAdapter",
    "CausalPredictiveFitReport",
    "CausalPredictiveScore",
    "CausalSequenceReport",
    "CausalTrace",
    "coerce_artifact_metadata",
    "ControllerSummary",
    "ControllerSummaryBuilder",
    "ControllerSummaryConfig",
    "create_delay_line_substrate",
    "create_echo_state_substrate",
    "create_hierarchical_substrate",
    "create_mixed_memory_substrate",
    "create_oscillatory_memory_substrate",
    "create_substrate",
    "create_substrate_for_model",
    "DatasetEvaluation",
    "delay_small",
    "DelayLineConfig",
    "DelayLineSubstrate",
    "echo_state_small",
    "EchoStateSubstrate",
    "ensure_byte_tokens",
    "ensure_tokens",
    "evaluate_dataset",
    "evaluate_rollout",
    "evaluate_rollout_curve",
    "evaluate_transfer_probe",
    "ExactContextCache",
    "ExactContextConfig",
    "ExactContextFitReport",
    "ExactContextMemory",
    "ExactContextPrediction",
    "ExpertFitReport",
    "ExpertScore",
    "FitReport",
    "FrozenReadoutExpert",
    "GlobalLocalBridge",
    "GlobalLocalBridgeConfig",
    "hierarchical_small",
    "HierarchicalFeatureView",
    "HierarchicalSubstrate",
    "HierarchicalSubstrateConfig",
    "HierarchicalSummary",
    "HormoneModulationConfig",
    "HormoneModulator",
    "HormoneState",
    "LatentCommitter",
    "LatentConfig",
    "LatentControllerConfig",
    "LatentObservation",
    "LatentState",
    "learnable_causal_bank_substrate_keys",
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
    "MemoryMergeMode",
    "MemoryPredictionRecord",
    "MemoryPredictionSummary",
    "mixed_memory_small",
    "MixedMemoryConfig",
    "MixedMemorySubstrate",
    "NextStepScore",
    "NgramMemory",
    "NgramMemoryConfig",
    "NgramMemoryReport",
    "NoncausalReconstructiveAdapter",
    "NoncausalReconstructiveConfig",
    "NoncausalReconstructiveFitReport",
    "NoncausalReconstructiveReport",
    "NoncausalReconstructiveTrace",
    "normalized_entropy",
    "OnlineCausalMemory",
    "OnlineMemoryConfig",
    "OpenPredictiveCoder",
    "OpenPredictiveCoderConfig",
    "OracleAnalysisAdapter",
    "OracleAnalysisConfig",
    "OracleAnalysisFitReport",
    "OracleAnalysisPoint",
    "OracleAnalysisReport",
    "OscillatoryMemoryConfig",
    "OscillatoryMemorySubstrate",
    "overlap_mass",
    "PatchPooler",
    "PatchPoolerConfig",
    "PathwayGateConfig",
    "PathwayGateController",
    "PathwayGateState",
    "PathwayGateValues",
    "PredictionState",
    "PredictiveController",
    "PredictiveObservation",
    "PredictiveState",
    "PredictiveSurpriseConfig",
    "PredictiveSurpriseController",
    "probability_diagnostics",
    "ProbabilityDiagnostics",
    "ProbabilityDiagnosticsConfig",
    "replay_spans_from_scores",
    "ReplaySpan",
    "ReservoirConfig",
    "ReservoirTopology",
    "RidgeReadout",
    "RolloutCheckpoint",
    "RolloutCurve",
    "RolloutCurveEvaluation",
    "RolloutCurveMode",
    "RolloutCurvePoint",
    "RolloutEvaluation",
    "RolloutMode",
    "RoutingConfig",
    "RoutingDecision",
    "RoutingMode",
    "SampledBandSummary",
    "SampledMultiscaleReadout",
    "SampledReadoutBandConfig",
    "SampledReadoutConfig",
    "scale_causal_bank_config",
    "score_dataset",
    "score_next_step",
    "ScoredSpan",
    "SegmenterConfig",
    "SegmenterMode",
    "SegmentStats",
    "select_scored_spans",
    "SequenceReport",
    "SequenceTrace",
    "shared_top_k_mass",
    "SpanSelectionConfig",
    "stack_summaries",
    "StatisticalBackoffCache",
    "StatisticalBackoffConfig",
    "StatisticalBackoffFitReport",
    "StatisticalBackoffMemory",
    "StatisticalBackoffPrediction",
    "StatisticalBackoffScore",
    "StatisticalBackoffTrace",
    "SubstrateKind",
    "summarize_artifact_audits",
    "SummaryMode",
    "SummaryRouter",
    "SupportBlend",
    "SupportMixConfig",
    "SupportWeightedMixer",
    "tag_metadata",
    "TeacherExportAdapter",
    "TeacherExportConfig",
    "TeacherExportRecord",
    "TeacherExportReport",
    "TokenSubstrate",
    "top1_agreement",
    "top1_peak",
    "top2_margin",
    "top_k_mass",
    "TrainModeConfig",
    "TrainStateMode",
    "TransferEvaluation",
    "TransferProbeReport",
    "validate_causal_bank_config",
]
