"""Gradient-sign tests for adaptive_substrate plug-ins.

Both hash_memory and local_path silently failed in session 11 because the
adaptive_substrate branch of _linear_logits / _forward_raw early-returned
before the plug-in modules were invoked. The bug went undetected for 3
mutation cycles (~68 min wasted GPU time) before Heinrich's MRI proved
the modules received zero gradient (bit-identical weights from step 5k
to 20k).

These tests assert that any adaptive-substrate-enabled module receives
non-zero gradient on a single forward+backward pass. They take ~5 ms
and would have caught both bypass bugs at test time.
"""

from __future__ import annotations

import pytest
import torch

from decepticons.causal_bank import CausalBankConfig
from decepticons.models.causal_bank_torch import CausalBankModel


def _adaptive_cfg(**overrides) -> CausalBankConfig:
    """Minimal config exercising the adaptive substrate path."""
    base = dict(
        embedding_dim=8,
        linear_modes=16,
        max_seq_len=32,
        linear_half_life_max=8.0,
        linear_hidden=(16,),
        linear_readout_kind="mlp",
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


def _max_abs_grad_for(model: torch.nn.Module, name_substr: str) -> float:
    """Max |grad| across parameters whose qualified name contains substr."""
    grads = []
    for name, param in model.named_parameters():
        if name_substr in name and param.grad is not None:
            grads.append(param.grad.abs().max().item())
    if not grads:
        raise AssertionError(f"No parameters matched '{name_substr}'")
    return max(grads)


def test_hash_memory_receives_gradient_on_adaptive_substrate():
    """hash_memory params must receive non-zero gradient when enabled.

    The session-11 bug: the adaptive branch returned readout(features)
    before invoking self._hash_memory(). All hash_memory weights were
    bit-identical step 5k → 20k. Fixed in commit 50b5f3d.
    """
    cfg = _adaptive_cfg(
        hash_memory=True, hash_memory_slots=8, hash_memory_dim=8,
    )
    model = CausalBankModel(64, cfg)
    x = torch.randint(0, 64, (2, 16))
    loss = model(x).sum()
    loss.backward()

    max_grad = _max_abs_grad_for(model, "_hash_memory")
    assert max_grad > 0, (
        f"hash_memory params received zero gradient (max |grad| = {max_grad}). "
        "The adaptive substrate path is bypassing hash_memory again — see "
        "decepticons/models/causal_bank_torch.py::_linear_logits."
    )


def test_local_path_receives_gradient_on_adaptive_substrate():
    """local_readout params must receive non-zero gradient when local_scale > 0.

    The session-11 bug: _forward_raw adaptive branch hard-returned
    _linear_logits(chars).clone() without mixing in _local_logits, so
    the local_readout never received gradient even with local_scale=0.25
    configured. Fixed in commit 50b5f3d.
    """
    cfg = _adaptive_cfg(local_scale=0.25, enable_local=True)
    model = CausalBankModel(64, cfg)
    x = torch.randint(0, 64, (2, 16))
    loss = model(x).sum()
    loss.backward()

    max_grad = _max_abs_grad_for(model, "local_readout")
    assert max_grad > 0, (
        f"local_readout received zero gradient (max |grad| = {max_grad}). "
        "The adaptive substrate path is bypassing local_path again — see "
        "decepticons/models/causal_bank_torch.py::_forward_raw."
    )


def test_local_path_disabled_when_scale_zero():
    """local_scale=0 should NOT route gradient through the local readout.

    Sanity check: confirm we don't accidentally compute local logits when
    the user has disabled the path. Either local_readout doesn't exist
    (parameter-saving), or it exists but receives zero gradient.
    """
    cfg = _adaptive_cfg(local_scale=0.0, enable_local=True)
    model = CausalBankModel(64, cfg)
    x = torch.randint(0, 64, (2, 16))
    loss = model(x).sum()
    loss.backward()

    grads = []
    for name, param in model.named_parameters():
        if "local_readout" in name and param.grad is not None:
            grads.append(param.grad.abs().max().item())
    if grads:
        max_grad = max(grads)
        assert max_grad == 0.0, (
            f"local_readout received gradient with local_scale=0 "
            f"(max |grad| = {max_grad}). Adaptive path should skip "
            "local_logits when scale is zero."
        )
