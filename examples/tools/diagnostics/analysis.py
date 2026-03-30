from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def _as_float_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("expected a non-empty array")
    return array


@dataclass(frozen=True)
class SignalSummary:
    name: str
    shape: tuple[int, ...]
    size: int
    mean: float
    std: float
    min: float
    max: float
    q10: float
    median: float
    q90: float
    energy: float
    abs_mean: float


@dataclass(frozen=True)
class BinaryMaskSummary:
    name: str
    shape: tuple[int, ...]
    size: int
    mean: float
    gt05: float
    gt09: float
    lt01: float
    soft: float
    active_dims: int
    strong_active_dims: int
    dim_mean_mean: float
    dim_mean_std: float
    dim_std_mean: float
    dim_std_max: float


@dataclass(frozen=True)
class AlignmentSummary:
    source_name: str
    target_name: str
    shape: tuple[int, ...]
    pearson: float
    cosine: float
    mae: float
    rmse: float
    mean_delta: float
    abs_mean_delta: float


def summarize_signal(values: object, *, name: str = "signal") -> SignalSummary:
    array = _as_float_array(values)
    flat = array.reshape(-1)
    q10, median, q90 = np.quantile(flat, [0.1, 0.5, 0.9])
    return SignalSummary(
        name=name,
        shape=tuple(int(dim) for dim in array.shape),
        size=int(array.size),
        mean=float(flat.mean()),
        std=float(flat.std()),
        min=float(flat.min()),
        max=float(flat.max()),
        q10=float(q10),
        median=float(median),
        q90=float(q90),
        energy=float(np.mean(flat**2)),
        abs_mean=float(np.mean(np.abs(flat))),
    )


def summarize_binary_mask(
    values: object,
    *,
    name: str = "mask",
    active_threshold: float = 0.5,
    strong_threshold: float = 0.2,
) -> BinaryMaskSummary:
    array = _as_float_array(values)
    if array.ndim == 0:
        vectors = array.reshape(1, 1)
    elif array.ndim == 1:
        vectors = array.reshape(1, -1)
    else:
        vectors = array.reshape(-1, array.shape[-1])

    flat = vectors.reshape(-1)
    dim_mean = vectors.mean(axis=0)
    dim_std = vectors.std(axis=0)

    active_dims = int(np.sum(dim_mean < active_threshold))
    strong_active_dims = int(np.sum(dim_mean < strong_threshold))
    soft = float(np.mean((flat >= 0.1) & (flat <= 0.9)))
    return BinaryMaskSummary(
        name=name,
        shape=tuple(int(dim) for dim in array.shape),
        size=int(array.size),
        mean=float(flat.mean()),
        gt05=float(np.mean(flat > 0.5)),
        gt09=float(np.mean(flat > 0.9)),
        lt01=float(np.mean(flat < 0.1)),
        soft=soft,
        active_dims=active_dims,
        strong_active_dims=strong_active_dims,
        dim_mean_mean=float(dim_mean.mean()),
        dim_mean_std=float(dim_mean.std()),
        dim_std_mean=float(dim_std.mean()),
        dim_std_max=float(dim_std.max()),
    )


def summarize_alignment(
    source: object,
    target: object,
    *,
    source_name: str = "source",
    target_name: str = "target",
) -> AlignmentSummary:
    source_array = _as_float_array(source)
    target_array = _as_float_array(target)
    if source_array.shape != target_array.shape:
        raise ValueError("alignment summaries require matching shapes")

    src = source_array.reshape(-1)
    tgt = target_array.reshape(-1)
    src_centered = src - src.mean()
    tgt_centered = tgt - tgt.mean()
    denom = float(np.linalg.norm(src_centered) * np.linalg.norm(tgt_centered))
    pearson = float(np.dot(src_centered, tgt_centered) / denom) if denom > 0 else 0.0
    cosine_denom = float(np.linalg.norm(src) * np.linalg.norm(tgt))
    cosine = float(np.dot(src, tgt) / cosine_denom) if cosine_denom > 0 else 0.0
    delta = src - tgt
    return AlignmentSummary(
        source_name=source_name,
        target_name=target_name,
        shape=tuple(int(dim) for dim in source_array.shape),
        pearson=pearson,
        cosine=cosine,
        mae=float(np.mean(np.abs(delta))),
        rmse=float(np.sqrt(np.mean(delta**2))),
        mean_delta=float(src.mean() - tgt.mean()),
        abs_mean_delta=float(np.abs(src.mean() - tgt.mean())),
    )


def format_signal_summary(summary: SignalSummary) -> str:
    return (
        f"{summary.name}: shape={summary.shape} mean={summary.mean:.4f} std={summary.std:.4f} "
        f"min={summary.min:.4f} max={summary.max:.4f} q10={summary.q10:.4f} "
        f"median={summary.median:.4f} q90={summary.q90:.4f}"
    )


def format_binary_mask_summary(summary: BinaryMaskSummary) -> str:
    return (
        f"{summary.name}: shape={summary.shape} mean={summary.mean:.4f} "
        f">0.5={summary.gt05*100:.1f}% >0.9={summary.gt09*100:.1f}% <0.1={summary.lt01*100:.1f}% "
        f"soft={summary.soft*100:.1f}% active_dims={summary.active_dims} "
        f"strong_active_dims={summary.strong_active_dims}"
    )


def format_alignment_summary(summary: AlignmentSummary) -> str:
    return (
        f"{summary.source_name}->{summary.target_name}: shape={summary.shape} "
        f"pearson={summary.pearson:.4f} cosine={summary.cosine:.4f} "
        f"mae={summary.mae:.4f} rmse={summary.rmse:.4f}"
    )
