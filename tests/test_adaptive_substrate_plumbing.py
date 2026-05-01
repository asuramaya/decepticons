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

torch = pytest.importorskip("torch")

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


def _nonadaptive_cfg(**overrides) -> CausalBankConfig:
    """Minimal config exercising the non-adaptive substrate path. This is
    the branch whose forward was reimplemented inline in the loader until the
    session-12 P2 collapse; this config drives the regression fixture that
    catches re-introduction of a parallel forward path there."""
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
        local_scale=0.25,   # exercise local-path composition
        enable_local=True,
        linear_impl="scan",
        substrate_mode="frozen",
        adaptive_substrate=False,   # the whole point of this config
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


def _max_abs_grad(params) -> float:
    m = 0.0
    for p in params:
        if p.grad is not None and p.requires_grad:
            g = float(p.grad.abs().max().item())
            if g > m:
                m = g
    return m


def test_all_submodules_receive_gradient():
    """Generalized plug-in preflight: every submodule with requires_grad params
    must receive a non-zero gradient on a single forward+backward.

    This is the P13 fix — retires the hardcoded KNOWN_PLUGIN_PREFIXES list.
    Any new plug-in added to the model is checked automatically without the
    test needing to be updated.

    Modules reported as dormant when every one of their parameters has zero
    gradient. This catches:
    - Plug-ins enabled but not invoked (session-11 hash_memory / localon bug)
    - Dead code paths that instantiate weights but never use them
    - Config flags that flip a module on without plumbing it
    """
    cfg = _adaptive_cfg(
        hash_memory=True, hash_memory_slots=8, hash_memory_dim=8,
        local_scale=0.25, enable_local=True,
    )
    model = CausalBankModel(64, cfg)
    x = torch.randint(0, 64, (2, 16))
    loss = model(x).sum()
    loss.backward()

    dormant = []
    checked = 0
    for mod_name, module in model.named_modules():
        if not mod_name:  # skip the root
            continue
        params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if not params:
            continue
        checked += 1
        if _max_abs_grad(params) == 0.0:
            dormant.append(mod_name)

    assert not dormant, (
        f"{len(dormant)}/{checked} submodules received zero gradient: "
        f"{dormant[:8]}. These parameters are instantiated but unreachable "
        "from the forward path — same bug class as session-11 hashmem/localon."
    )


def test_forward_captured_matches_training_forward():
    """P14: forward_captured(x).logits must equal model(x) bit-identically.

    The session-11 drift: heinrich's loader.forward_captured reimplemented the
    adaptive-substrate forward inline and diverged from model(x) when
    hash_memory / local_path plumbing landed in the training path (50b5f3d).
    The path-collapse refactor (f52865e) made them structurally identical —
    this test guards against regression if anyone reintroduces a parallel path.
    """
    try:
        from decepticons.loader import CausalBankInference
    except ImportError:
        pytest.skip("decepticons.loader not importable")
    import numpy as np

    cfg = _adaptive_cfg(
        hash_memory=True, hash_memory_slots=8, hash_memory_dim=8,
        local_scale=0.25, enable_local=True,
    )
    model = CausalBankModel(64, cfg)
    model.train(False)  # puts model in inference mode
    # Construct a minimal stub. CausalBankInference is a frozen dataclass,
    # so we use its constructor with dummy values for fields we don't use.
    inf = CausalBankInference(
        config={},
        half_lives=np.zeros((cfg.linear_modes,), dtype=np.float32),
        tokenizer=None,
        _model=model,
        _device="cpu",
    )

    x = torch.randint(0, 64, (2, 16))
    x_np = x.numpy()

    with torch.inference_mode():
        logits_train = model(x)
    if logits_train.ndim == 4:  # fat-readout patching
        logits_train = logits_train[:, :, 0, :]

    result = inf.forward_captured(x_np)
    logits_cap = torch.from_numpy(result["logits"])

    max_diff = float((logits_train - logits_cap).abs().max().item())
    assert max_diff == 0.0, (
        f"model(x) and forward_captured(x) diverged by max |Δ|={max_diff}. "
        "The loader's forward path drifted from the model's. Check that "
        "forward_captured calls model(x) rather than reimplementing the "
        "substrate forward — see f52865e."
    )


def test_forward_captured_matches_training_forward_nonadaptive():
    """P2 (session 12): forward_captured(x).logits must equal model(x)
    bit-identically on the NON-adaptive path too.

    Until session 12 the loader reimplemented the non-adaptive forward
    inline — calling _linear_states, then _linear_logits or manual band
    readouts, then _local_logits, then recomposing logits by hand. Every
    plugin landed in _linear_logits had to be mirrored here or silently
    dropped out of MRI captures (same drift class as the adaptive branch
    bypass fixed in f52865e).

    The collapse replaces all of that with `model(x)` plus stash reads. This
    fixture asserts the collapse is still bit-identical; regression here
    means someone has reintroduced a parallel non-adaptive forward path.
    """
    try:
        from decepticons.loader import CausalBankInference
    except ImportError:
        pytest.skip("decepticons.loader not importable")
    import numpy as np

    cfg = _nonadaptive_cfg()
    model = CausalBankModel(64, cfg)
    model.train(False)
    inf = CausalBankInference(
        config={},
        half_lives=np.zeros((cfg.linear_modes,), dtype=np.float32),
        tokenizer=None,
        _model=model,
        _device="cpu",
    )

    x = torch.randint(0, 64, (2, 16))
    x_np = x.numpy()

    with torch.inference_mode():
        logits_train = model(x)
    if logits_train.ndim == 4:
        logits_train = logits_train[:, :, 0, :]

    result = inf.forward_captured(x_np)
    logits_cap = torch.from_numpy(result["logits"])

    max_diff = float((logits_train - logits_cap).abs().max().item())
    assert max_diff == 0.0, (
        f"model(x) and forward_captured(x) diverged by max |Δ|={max_diff} "
        "on the non-adaptive path. The loader's forward has drifted from "
        "model(x) — check that the non-adaptive branch of forward_captured "
        "still calls self._model(x) and reads stashes rather than "
        "reimplementing the forward inline."
    )

    # Substrate and local_logits should also be populated from the stashes
    # (rather than silently None), since this config enables both.
    assert result["substrate_states"] is not None
    assert result["local_logits"] is not None


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
