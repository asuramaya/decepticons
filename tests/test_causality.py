"""Causality verification: model outputs must not depend on future tokens.

Feed two sequences identical up to position t, different after t.
If logits at position t differ, the model sees the future.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from decepticons.causal_bank import CausalBankConfig, scale_config
from decepticons.models.causal_bank_torch import CausalBankModel


def _check_causality(model, vocab_size=1024, seq_len=64, check_positions=None):
    """Verify causal masking: future tokens must not affect past logits."""
    model.eval()  # noqa: S307
    torch.manual_seed(99)

    seq_a = torch.randint(0, vocab_size, (1, seq_len))

    if check_positions is None:
        check_positions = [0, 1, seq_len // 4, seq_len // 2, seq_len - 2]

    violations = []
    with torch.no_grad():
        logits_a = model(seq_a)

        for t in check_positions:
            if t >= seq_len - 1:
                continue
            seq_b = seq_a.clone()
            seq_b[0, t + 1:] = torch.randint(0, vocab_size, (seq_len - t - 1,))

            logits_b = model(seq_b)

            diff = (logits_a[0, t] - logits_b[0, t]).abs().max().item()
            if diff > 1e-5:
                violations.append(f"position {t}: max logit diff = {diff:.6f}")

    return violations


def test_frozen_substrate_is_causal():
    cfg = scale_config(CausalBankConfig(substrate_mode="frozen"), 4.0)
    model = CausalBankModel(256, cfg)
    violations = _check_causality(model, vocab_size=256, seq_len=32)
    assert violations == [], f"Causality violations:\n" + "\n".join(violations)


def test_learnable_mixing_is_causal():
    cfg = scale_config(CausalBankConfig(substrate_mode="learnable_mixing"), 4.0)
    model = CausalBankModel(256, cfg)
    violations = _check_causality(model, vocab_size=256, seq_len=32)
    assert violations == [], f"Causality violations:\n" + "\n".join(violations)


def test_learnable_decays_is_causal():
    cfg = scale_config(CausalBankConfig(substrate_mode="learnable_decays"), 4.0)
    model = CausalBankModel(256, cfg)
    violations = _check_causality(model, vocab_size=256, seq_len=32)
    assert violations == [], f"Causality violations:\n" + "\n".join(violations)


def test_selective_scan_augment_is_causal():
    cfg = scale_config(CausalBankConfig(state_dim=8), 4.0)
    model = CausalBankModel(256, cfg)
    violations = _check_causality(model, vocab_size=256, seq_len=32)
    assert violations == [], f"Causality violations:\n" + "\n".join(violations)


def test_readout_bands_is_causal():
    cfg = scale_config(CausalBankConfig(readout_bands=4), 4.0)
    model = CausalBankModel(256, cfg)
    violations = _check_causality(model, vocab_size=256, seq_len=32)
    assert violations == [], f"Causality violations:\n" + "\n".join(violations)


def test_routed_experts_is_causal():
    cfg = scale_config(CausalBankConfig(
        linear_readout_kind="routed_sqrelu_experts",
        linear_readout_num_experts=4,
    ), 4.0)
    model = CausalBankModel(256, cfg)
    violations = _check_causality(model, vocab_size=256, seq_len=32)
    assert violations == [], f"Causality violations:\n" + "\n".join(violations)
