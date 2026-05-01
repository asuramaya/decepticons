"""Tests for patch-at-readout (CausalBankConfig.patch_n > 1)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from decepticons.causal_bank import CausalBankConfig
from decepticons.models.causal_bank_torch import CausalBankModel


def _cfg(**overrides) -> CausalBankConfig:
    base = dict(
        embedding_dim=8,
        linear_modes=16,
        max_seq_len=32,
        linear_half_life_max=8.0,
        linear_hidden=(32,),
        linear_readout_kind="routed_sqrelu_experts",
        linear_readout_num_experts=2,
        readout_bands=1,
        local_window=4,
        local_scale=0.0,
        linear_impl="scan",
        substrate_mode="frozen",
        adaptive_substrate=True,
        hrr_omega_init=True,
        num_heads=1,
    )
    base.update(overrides)
    return CausalBankConfig(**base)


def test_patch_n_default_shape_unchanged():
    """N=1 (default) must produce [B, T, vocab] logits — baseline invariant."""
    cfg = _cfg()
    model = CausalBankModel(256, cfg)
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 256), f"N=1 should give [B,T,V], got {logits.shape}"


def test_patch_n_four_adds_head_dim():
    """N=4 must produce [B, T, N, vocab] logits."""
    cfg = _cfg(patch_n=4)
    model = CausalBankModel(256, cfg)
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 4, 256), f"N=4 should give [B,T,N,V], got {logits.shape}"


def test_patch_n_two_with_mlp_readout():
    """Validates patch_n also works with mlp readout kind."""
    cfg = _cfg(patch_n=2, linear_readout_kind="mlp")
    model = CausalBankModel(256, cfg)
    x = torch.randint(0, 256, (1, 8))
    logits = model(x)
    assert logits.shape == (1, 8, 2, 256)


def test_patch_n_rejects_unsupported_readouts():
    """patch_n > 1 with tied_recursive / tied_embed / recurrent should raise."""
    for kind in ("tied_recursive", "tied_embed_readout", "recurrent"):
        with pytest.raises(ValueError, match="patch_n"):
            CausalBankModel(256, _cfg(patch_n=2, linear_readout_kind=kind))


def test_patch_n_rejects_bands():
    """patch_n > 1 with readout_bands > 1 should raise."""
    with pytest.raises(ValueError, match="patch_n"):
        CausalBankModel(256, _cfg(patch_n=2, readout_bands=2, linear_modes=32))
