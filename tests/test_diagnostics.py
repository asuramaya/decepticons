"""Tests for model diagnostics — verify introspection produces valid structure."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from decepticons.causal_bank import CausalBankConfig, scale_config
from decepticons.models.causal_bank_torch import CausalBankModel
from decepticons.models.diagnostics import diagnose, format_diagnostics


def _fresh_model(scale=4.0, **kwargs):
    cfg = scale_config(CausalBankConfig(**kwargs), scale)
    return CausalBankModel(256, cfg)


def test_diagnose_returns_all_sections():
    model = _fresh_model()
    tokens = torch.randint(0, 256, (2, 32))
    diag = diagnose(model, tokens, vocab_size=256)
    assert "modes" in diag
    assert "timescale_bands" in diag
    assert "input_projection" in diag
    assert "summary" in diag


def test_fresh_model_all_modes_alive():
    model = _fresh_model()
    tokens = torch.randint(0, 256, (2, 32))
    diag = diagnose(model, tokens, vocab_size=256)
    assert diag["modes"]["dead_pct"] < 5.0


def test_timescale_bands_sum_to_100():
    model = _fresh_model()
    tokens = torch.randint(0, 256, (2, 32))
    diag = diagnose(model, tokens, vocab_size=256)
    total = sum(b["contribution_pct"] for b in diag["timescale_bands"].values())
    assert abs(total - 100.0) < 1.0


def test_learnable_mixing_detected():
    model = _fresh_model(substrate_mode="learnable_mixing")
    tokens = torch.randint(0, 256, (2, 32))
    diag = diagnose(model, tokens, vocab_size=256)
    assert diag["input_projection"]["learnable"] is True
    assert diag["summary"]["input_projection_learnable"] is True


def test_frozen_projection_detected():
    model = _fresh_model(substrate_mode="frozen")
    tokens = torch.randint(0, 256, (2, 32))
    diag = diagnose(model, tokens, vocab_size=256)
    assert diag["input_projection"]["learnable"] is False


def test_format_diagnostics_produces_string():
    model = _fresh_model()
    tokens = torch.randint(0, 256, (2, 32))
    diag = diagnose(model, tokens, vocab_size=256)
    text = format_diagnostics(diag)
    assert "MODES:" in text
    assert "TIMESCALE BANDS:" in text
    assert "frozen" in text
