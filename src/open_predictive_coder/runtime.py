from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceTrace:
    features: np.ndarray
    targets: np.ndarray
    boundaries: np.ndarray
    tokens: int
    patches: int


@dataclass(frozen=True)
class SequenceReport:
    tokens: int
    patches: int
    mean_patch_size: float
    compression_ratio: float
    bits_per_byte: float


@dataclass(frozen=True)
class FitReport:
    sequences: int
    tokens: int
    patches: int
    mean_patch_size: float
    compression_ratio: float
    train_bits_per_byte: float


__all__ = [
    "FitReport",
    "SequenceReport",
    "SequenceTrace",
]
