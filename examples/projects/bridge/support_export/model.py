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

from open_predictive_coder import (
    ExactContextCache,
    ExactContextConfig,
    ExactContextFitReport,
    ExactContextMemory,
    StatisticalBackoffCache,
    StatisticalBackoffFitReport,
    TeacherExportAdapter,
    TeacherExportConfig,
    ensure_tokens,
)


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(data).astype(np.int64, copy=False)


@dataclass(frozen=True)
class SupportExportConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    ngram_bigram_alpha: float = 0.5
    ngram_trigram_alpha: float = 0.5
    ngram_trigram_bucket_count: int = 2048
    candidate_count: int = 4
    mixture_steps: int = 128


@dataclass(frozen=True)
class SupportExportFitReport:
    backoff: StatisticalBackoffFitReport
    exact: ExactContextFitReport


@dataclass(frozen=True)
class SupportExportTrace:
    tokens: int
    steps: int
    teacher_probs: np.ndarray
    student_probs: np.ndarray
    targets: np.ndarray
    teacher_labels: np.ndarray
    student_labels: np.ndarray
    exact_support: np.ndarray
    exact_order: np.ndarray


@dataclass(frozen=True)
class SupportExportReport:
    tokens: int
    steps: int
    teacher_bits_per_byte: float
    student_bits_per_byte: float
    mean_bits_per_byte: float
    label_flip_rate: float
    mean_entropy: float
    mean_peak: float
    mean_candidate4: float
    mean_agreement_mass: float
    mean_exact_support: float
    mean_exact_order: float


class SupportExportModel:
    def __init__(self, config: SupportExportConfig | None = None):
        self.config = config or SupportExportConfig()
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
        self.export_adapter = TeacherExportAdapter(
            TeacherExportConfig(
                vocabulary_size=self.config.vocabulary_size,
                source_names=("teacher", "student"),
            )
        )

    @classmethod
    def build(cls, **kwargs: object) -> "SupportExportModel":
        return cls(SupportExportConfig(**kwargs))

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> SupportExportFitReport:
        tokens = _coerce_tokens(data)
        return SupportExportFitReport(
            backoff=self.backoff_cache.fit(tokens),
            exact=self.exact_cache.fit(tokens),
        )

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> SupportExportTrace:
        tokens = _coerce_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        teacher_rows: list[np.ndarray] = []
        student_rows: list[np.ndarray] = []
        exact_supports: list[float] = []
        exact_orders: list[int] = []

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            teacher = self.exact_cache.prediction_summary(prefix)
            student = self.backoff_cache.prediction_summary(prefix)
            teacher_rows.append(teacher.predictive_distribution(mode="active"))
            student_rows.append(student.predictive_distribution(mode="mixed"))
            exact_supports.append(float(teacher.active_prediction.support))
            exact_orders.append(int(teacher.active_prediction.order))

        teacher_probs = np.vstack(teacher_rows)
        student_probs = np.vstack(student_rows)
        teacher_labels = np.argmax(teacher_probs, axis=-1).astype(np.int64, copy=False)
        student_labels = np.argmax(student_probs, axis=-1).astype(np.int64, copy=False)
        return SupportExportTrace(
            tokens=int(tokens.size),
            steps=int(tokens.size - 1),
            teacher_probs=teacher_probs,
            student_probs=student_probs,
            targets=tokens[1:],
            teacher_labels=teacher_labels,
            student_labels=student_labels,
            exact_support=np.asarray(exact_supports, dtype=np.float64),
            exact_order=np.asarray(exact_orders, dtype=np.int64),
        )

    def report(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> SupportExportReport:
        trace = self.trace(sequence)
        shared = self.export_adapter.export(
            trace.teacher_probs,
            trace.student_probs,
            targets=trace.targets,
        )
        return SupportExportReport(
            tokens=trace.tokens,
            steps=trace.steps,
            teacher_bits_per_byte=float(shared.teacher_bits_per_byte if shared.teacher_bits_per_byte is not None else 0.0),
            student_bits_per_byte=float(shared.student_bits_per_byte if shared.student_bits_per_byte is not None else 0.0),
            mean_bits_per_byte=float(shared.mean_bits_per_byte if shared.mean_bits_per_byte is not None else 0.0),
            label_flip_rate=float(shared.label_flip_rate),
            mean_entropy=float(shared.mean_entropy),
            mean_peak=float(shared.mean_peak),
            mean_candidate4=float(shared.mean_top_k_mass),
            mean_agreement_mass=float(shared.mean_shared_top_k_mass),
            mean_exact_support=float(np.mean(trace.exact_support)),
            mean_exact_order=float(np.mean(trace.exact_order)),
        )


__all__ = [
    "SupportExportConfig",
    "SupportExportFitReport",
    "SupportExportModel",
    "SupportExportReport",
    "SupportExportTrace",
]
