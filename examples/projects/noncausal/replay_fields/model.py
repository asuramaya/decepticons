from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_predictive_coder.artifacts import ArtifactAccounting, ArtifactMetadata, ReplaySpan, make_replay_span
from open_predictive_coder.codecs import ensure_tokens
from open_predictive_coder.noncausal_reconstructive import (
    NoncausalReconstructiveAdapter,
    NoncausalReconstructiveConfig,
    NoncausalReconstructiveFitReport,
    NoncausalReconstructiveReport,
    NoncausalReconstructiveTrace,
)


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object]) -> np.ndarray:
    return ensure_tokens(data).astype(np.int64, copy=False)


@dataclass(frozen=True)
class ReplayFieldsConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    bidirectional_left_order: int = 2
    bidirectional_right_order: int = 2
    blend_temperature: float = 1.0
    agreement_threshold: float = 0.75
    replay_threshold: float = 0.55
    min_replay_span: int = 2
    field_separator: int = ord("|")
    artifact_name: str = "replay_fields"
    metadata: ArtifactMetadata | None = None

    def to_adapter_config(self) -> NoncausalReconstructiveConfig:
        return NoncausalReconstructiveConfig(
            vocabulary_size=self.vocabulary_size,
            exact_max_order=self.exact_max_order,
            exact_alpha=self.exact_alpha,
            bidirectional_left_order=self.bidirectional_left_order,
            bidirectional_right_order=self.bidirectional_right_order,
            blend_temperature=self.blend_temperature,
            agreement_threshold=self.agreement_threshold,
            replay_threshold=self.replay_threshold,
            min_replay_span=self.min_replay_span,
        )


@dataclass(frozen=True)
class ReplayFieldsFitReport:
    reconstruction: NoncausalReconstructiveFitReport


@dataclass(frozen=True)
class ReplayFieldsTrace:
    reconstruction: NoncausalReconstructiveTrace
    field_spans: tuple[ReplaySpan, ...]
    replay_field_overlap: np.ndarray


@dataclass(frozen=True)
class ReplayFieldsReport:
    reconstruction: NoncausalReconstructiveReport
    field_span_count: int
    replay_field_overlap_count: int
    replay_field_ratio: float


class ReplayFieldsModel:
    def __init__(self, config: ReplayFieldsConfig | None = None):
        self.config = config or ReplayFieldsConfig()
        self.adapter = NoncausalReconstructiveAdapter(
            self.config.to_adapter_config(),
            artifact_name=self.config.artifact_name,
            metadata=self.config.metadata,
        )

    @classmethod
    def build(cls, **kwargs: object) -> "ReplayFieldsModel":
        return cls(ReplayFieldsConfig(**kwargs))

    def _field_spans(self, tokens: np.ndarray) -> tuple[ReplaySpan, ...]:
        spans: list[ReplaySpan] = []
        start = 0
        for index, token in enumerate(tokens):
            if int(token) == self.config.field_separator:
                if index > start:
                    spans.append(make_replay_span(start, index, label="field"))
                start = index + 1
        if start < tokens.size:
            spans.append(make_replay_span(start, int(tokens.size), label="field"))
        return tuple(spans)

    def _replay_field_overlap(self, tokens: np.ndarray, replay_spans: tuple[ReplaySpan, ...]) -> np.ndarray:
        mask = np.zeros((tokens.size,), dtype=bool)
        for field in self._field_spans(tokens):
            mask[field.start : field.stop] = True
        overlap = np.zeros((tokens.size,), dtype=bool)
        for span in replay_spans:
            overlap[span.start : span.stop] = mask[span.start : span.stop]
        return overlap

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> ReplayFieldsFitReport:
        return ReplayFieldsFitReport(reconstruction=self.adapter.fit(data))

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> ReplayFieldsTrace:
        tokens = _coerce_tokens(sequence)
        reconstruction = self.adapter.trace(tokens)
        field_spans = self._field_spans(tokens)
        overlap = self._replay_field_overlap(tokens, reconstruction.replay_spans)
        return ReplayFieldsTrace(
            reconstruction=reconstruction,
            field_spans=field_spans,
            replay_field_overlap=overlap,
        )

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> ReplayFieldsReport:
        tokens = _coerce_tokens(sequence)
        reconstruction = self.adapter.score(tokens)
        trace = self.trace(tokens)
        overlap_count = int(np.sum(trace.replay_field_overlap))
        return ReplayFieldsReport(
            reconstruction=reconstruction,
            field_span_count=len(trace.field_spans),
            replay_field_overlap_count=overlap_count,
            replay_field_ratio=(overlap_count / float(tokens.size)) if tokens.size else 0.0,
        )

    def accounting(self) -> ArtifactAccounting:
        return self.adapter.accounting()


__all__ = [
    "ReplayFieldsConfig",
    "ReplayFieldsFitReport",
    "ReplayFieldsModel",
    "ReplayFieldsReport",
    "ReplayFieldsTrace",
]
