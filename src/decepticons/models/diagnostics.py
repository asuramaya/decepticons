"""Causal-bank model diagnostics — the kernel introspects itself.

Measures what the model learned, not just what it outputs.
Feed a model and a batch of tokens, get a diagnostic dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def diagnose(model, tokens, *, vocab_size: int = 1024) -> dict[str, Any]:  # noqa: S307
    """Run diagnostics on a CausalBankModel.

    model: a CausalBankModel (torch)
    tokens: [batch, seq_len] integer tensor
    Returns diagnostic dict with mode liveness, readout selectivity, etc.
    """
    import torch

    was_training = model.training
    model.eval()  # noqa: S307
    config = model.config
    results: dict[str, Any] = {}

    # --- Mode liveness: which modes have non-trivial activation variance? ---
    with torch.no_grad():
        states, embed = model._linear_states(tokens)

    states_np = states.cpu().numpy()
    mode_variance = np.var(states_np, axis=(0, 1))
    n_modes = len(mode_variance)

    alive_threshold = np.median(mode_variance) * 0.01
    alive_mask = mode_variance > alive_threshold

    results["modes"] = {
        "total": n_modes,
        "alive": int(alive_mask.sum()),
        "dead": int((~alive_mask).sum()),
        "dead_pct": round(100 * (~alive_mask).sum() / n_modes, 1),
        "variance_median": float(np.median(mode_variance)),
        "variance_max": float(np.max(mode_variance)),
        "variance_min": float(np.min(mode_variance)),
    }

    # --- Timescale utilization: group modes by half-life band ---
    decays = model.linear_decays.detach().cpu().numpy().clip(1e-8, 1 - 1e-8)
    half_lives = np.log(0.5) / np.log(decays)
    n_bands = 4
    band_size = n_modes // n_bands
    band_names = ["fast", "medium_fast", "medium_slow", "slow"]
    timescale_bands = {}
    for i, name in enumerate(band_names):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else n_modes
        band_var = mode_variance[start:end]
        band_alive = alive_mask[start:end]
        band_hl = half_lives[start:end]
        timescale_bands[name] = {
            "mode_range": [int(start), int(end)],
            "half_life_range": [round(float(band_hl.min()), 2), round(float(band_hl.max()), 2)],
            "alive": int(band_alive.sum()),
            "dead": int((~band_alive).sum()),
            "mean_variance": round(float(band_var.mean()), 6),
            "contribution_pct": round(100 * band_var.sum() / max(mode_variance.sum(), 1e-12), 1),
        }
    results["timescale_bands"] = timescale_bands

    # --- Phase analysis: how old is each mode's signal? ---
    # For each mode, compute the autocorrelation across sequence positions.
    # High autocorrelation = mode carries persistent signal (slow effective timescale).
    # Low autocorrelation = mode responds to local input (fast effective timescale).
    # Compare effective timescale to the mode's half-life — mismatch means
    # the mode is being used differently than its decay rate suggests.
    if states_np.shape[1] > 2:
        # Lag-1 autocorrelation per mode (averaged across batch)
        s_prev = states_np[:, :-1, :]
        s_next = states_np[:, 1:, :]
        # Normalize
        s_prev_centered = s_prev - s_prev.mean(axis=1, keepdims=True)
        s_next_centered = s_next - s_next.mean(axis=1, keepdims=True)
        s_prev_std = s_prev_centered.std(axis=1, keepdims=True).clip(1e-8)
        s_next_std = s_next_centered.std(axis=1, keepdims=True).clip(1e-8)
        corr = (s_prev_centered * s_next_centered).mean(axis=1) / (s_prev_std * s_next_std).mean(axis=1)
        lag1_autocorr = corr.mean(axis=0)  # [modes]

        # Expected lag-1 autocorr from decay: simply the decay rate itself
        expected_autocorr = decays

        # Mismatch: modes behaving faster/slower than their half-life predicts
        autocorr_mismatch = lag1_autocorr - expected_autocorr

        results["phase"] = {
            "lag1_autocorr_mean": round(float(lag1_autocorr.mean()), 4),
            "lag1_autocorr_by_band": {
                name: round(float(lag1_autocorr[
                    timescale_bands[name]["mode_range"][0]:timescale_bands[name]["mode_range"][1]
                ].mean()), 4)
                for name in band_names
            },
            "expected_autocorr_by_band": {
                name: round(float(expected_autocorr[
                    timescale_bands[name]["mode_range"][0]:timescale_bands[name]["mode_range"][1]
                ].mean()), 4)
                for name in band_names
            },
            "mismatch_by_band": {
                name: round(float(autocorr_mismatch[
                    timescale_bands[name]["mode_range"][0]:timescale_bands[name]["mode_range"][1]
                ].mean()), 4)
                for name in band_names
            },
        }

    # --- Input projection structure ---
    proj = model.linear_in_proj.detach().cpu().numpy()
    col_norms = np.linalg.norm(proj, axis=0)
    row_norms = np.linalg.norm(proj, axis=1)
    results["input_projection"] = {
        "shape": list(proj.shape),
        "learnable": model.linear_in_proj.requires_grad,
        "col_norm_mean": round(float(col_norms.mean()), 4),
        "col_norm_std": round(float(col_norms.std()), 4),
        "row_norm_mean": round(float(row_norms.mean()), 4),
        "row_norm_std": round(float(row_norms.std()), 4),
        "dead_cols": int((col_norms < col_norms.mean() * 0.01).sum()),
    }

    # --- Readout selectivity: how does the MLP weight modes? ---
    readout = model.linear_readout
    if hasattr(readout, 'layers'):
        first_weight = readout.layers[0].weight.detach().cpu().numpy()
        mode_weights = first_weight[:, :n_modes]
        embed_weights = first_weight[:, n_modes:]

        mode_importance = np.abs(mode_weights).mean(axis=0)
        embed_importance = np.abs(embed_weights).mean(axis=0)

        results["readout_selectivity"] = {
            "mode_importance_mean": round(float(mode_importance.mean()), 6),
            "mode_importance_std": round(float(mode_importance.std()), 6),
            "embed_importance_mean": round(float(embed_importance.mean()), 6),
            "mode_vs_embed_ratio": round(
                float(mode_importance.mean() / max(embed_importance.mean(), 1e-8)), 3
            ),
            "top_10_modes": [int(i) for i in np.argsort(mode_importance)[-10:][::-1]],
            "bottom_10_modes": [int(i) for i in np.argsort(mode_importance)[:10]],
        }

        band_importance = {}
        for i, name in enumerate(band_names):
            start = i * band_size
            end = (i + 1) * band_size if i < n_bands - 1 else n_modes
            band_importance[name] = round(float(mode_importance[start:end].mean()), 6)
        results["readout_selectivity"]["by_timescale"] = band_importance

    # --- Findings: what the numbers mean ---
    findings = []
    alive_pct = 100 * alive_mask.sum() / n_modes
    if alive_pct < 50:
        findings.append(f"CRITICAL: {100-alive_pct:.0f}% modes dead — substrate capacity wasted")
    elif alive_pct < 90:
        findings.append(f"WARNING: {100-alive_pct:.0f}% modes dead — some capacity wasted")

    dominant = max(timescale_bands, key=lambda k: timescale_bands[k]["contribution_pct"])
    dominant_pct = timescale_bands[dominant]["contribution_pct"]
    if dominant_pct > 70:
        findings.append(f"IMBALANCED: {dominant} band carries {dominant_pct:.0f}% of variance — other timescales underused")

    if "readout_selectivity" in results:
        rs = results["readout_selectivity"]
        by_ts = rs.get("by_timescale", {})
        if by_ts:
            ts_values = list(by_ts.values())
            ts_range = max(ts_values) - min(ts_values) if ts_values else 0
            if ts_range < max(ts_values) * 0.1:
                findings.append("UNIFORM: readout weights modes equally across timescales — no specialization learned")
            else:
                most_weighted = max(by_ts, key=by_ts.get)
                findings.append(f"SPECIALIZED: readout favors {most_weighted} modes")

        ratio = rs.get("mode_vs_embed_ratio", 1.0)
        if ratio < 0.3:
            findings.append("EMBED-DOMINATED: readout relies on embeddings more than bank — substrate underutilized")
        elif ratio > 3.0:
            findings.append("MODE-DOMINATED: readout relies on bank more than embeddings — bank is working")

    proj = results.get("input_projection", {})
    if proj.get("dead_cols", 0) > n_modes * 0.1:
        findings.append(f"PROJECTION: {proj['dead_cols']} dead columns — modes receiving no input")

    if not findings:
        findings.append("HEALTHY: no anomalies detected")

    results["findings"] = findings

    # --- Summary ---
    results["summary"] = {
        "modes_alive_pct": round(alive_pct, 1),
        "dominant_timescale": dominant,
        "input_projection_learnable": model.linear_in_proj.requires_grad,
        "finding_count": len(findings),
    }

    if was_training:
        model.train()

    return results


def format_diagnostics(diag: dict[str, Any]) -> str:
    """One-page human-readable summary."""
    lines = []
    s = diag.get("summary", {})
    m = diag.get("modes", {})
    lines.append(f"MODES: {m.get('alive')}/{m.get('total')} alive ({m.get('dead_pct')}% dead)")
    lines.append(f"DOMINANT TIMESCALE: {s.get('dominant_timescale')}")
    lines.append(
        f"INPUT PROJECTION: {'learnable' if s.get('input_projection_learnable') else 'frozen'}"
    )

    bands = diag.get("timescale_bands", {})
    lines.append("\nTIMESCALE BANDS:")
    for name, b in bands.items():
        lines.append(
            f"  {name:15s} hl={b['half_life_range'][0]:6.1f}-{b['half_life_range'][1]:6.1f}  "
            f"alive={b['alive']:4d}  contribution={b['contribution_pct']:5.1f}%"
        )

    rs = diag.get("readout_selectivity", {})
    if rs:
        lines.append(f"\nREADOUT: mode/embed ratio = {rs.get('mode_vs_embed_ratio')}")
        by_ts = rs.get("by_timescale", {})
        if by_ts:
            lines.append(
                "  by timescale: " + "  ".join(f"{k}={v:.4f}" for k, v in by_ts.items())
            )

    findings = diag.get("findings", [])
    if findings:
        lines.append("\nFINDINGS:")
        for f in findings:
            lines.append(f"  {f}")

    return "\n".join(lines)
