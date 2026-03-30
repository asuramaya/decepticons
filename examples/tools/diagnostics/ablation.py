from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AblationComparison:
    name: str
    baseline_name: str
    variant_name: str
    baseline: float
    variant: float
    delta: float
    relative_change: float


@dataclass(frozen=True)
class TwoFactorDecomposition:
    baseline_name: str
    first_name: str
    second_name: str
    both_name: str
    baseline: float
    first_only: float
    second_only: float
    both: float
    first_effect: float
    second_effect: float
    interaction: float


def compare_ablation(
    baseline_name: str,
    baseline: float,
    variant_name: str,
    variant: float,
    *,
    name: str | None = None,
) -> AblationComparison:
    label = name or f"{baseline_name} vs {variant_name}"
    delta = float(variant - baseline)
    relative_change = float((variant / baseline - 1.0) * 100.0) if baseline != 0 else 0.0
    return AblationComparison(
        name=label,
        baseline_name=baseline_name,
        variant_name=variant_name,
        baseline=float(baseline),
        variant=float(variant),
        delta=delta,
        relative_change=relative_change,
    )


def compare_ablation_map(
    baseline_name: str,
    baseline: float,
    variants: Mapping[str, float],
) -> tuple[AblationComparison, ...]:
    return tuple(
        compare_ablation(baseline_name, baseline, name, value)
        for name, value in variants.items()
    )


def decompose_two_factor(
    baseline_name: str,
    baseline: float,
    first_name: str,
    first_only: float,
    second_name: str,
    second_only: float,
    both_name: str,
    both: float,
) -> TwoFactorDecomposition:
    first_effect = float(first_only - baseline)
    second_effect = float(second_only - baseline)
    interaction = float((both - baseline) - first_effect - second_effect)
    return TwoFactorDecomposition(
        baseline_name=baseline_name,
        first_name=first_name,
        second_name=second_name,
        both_name=both_name,
        baseline=float(baseline),
        first_only=float(first_only),
        second_only=float(second_only),
        both=float(both),
        first_effect=first_effect,
        second_effect=second_effect,
        interaction=interaction,
    )


def format_ablation_comparison(summary: AblationComparison) -> str:
    return (
        f"{summary.name}: baseline={summary.baseline:.4f} variant={summary.variant:.4f} "
        f"delta={summary.delta:+.4f} rel={summary.relative_change:+.1f}%"
    )


def format_two_factor_decomposition(summary: TwoFactorDecomposition) -> str:
    return (
        f"{summary.baseline_name} -> {summary.both_name}: baseline={summary.baseline:.4f} "
        f"{summary.first_name}={summary.first_only:.4f} {summary.second_name}={summary.second_only:.4f} "
        f"both={summary.both:.4f} first_effect={summary.first_effect:+.4f} "
        f"second_effect={summary.second_effect:+.4f} interaction={summary.interaction:+.4f}"
    )
