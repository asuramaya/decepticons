"""Load a causal bank checkpoint for inference and analysis.

This is the stable interface between decepticons (model code) and
external analysis tools (heinrich, notebooks, etc). No training code,
no chronohorn dependency.

Usage:
    from decepticons.loader import load_checkpoint

    model = load_checkpoint("path/to/run.checkpoint.pt", result_json="path/to/run.json")
    out = model.forward_captured(token_ids)
    # out["logits"], out["substrate_states"], out["route_weights"], out["band_logits"]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CausalBankInference:
    """Inference wrapper for a trained causal bank model.

    All methods return numpy arrays. The torch model is an internal detail.
    """

    config: dict
    half_lives: np.ndarray
    tokenizer: Any  # SentencePieceProcessor or None
    _model: Any = field(repr=False)  # CausalBankModel (torch)
    _device: str = field(default="cpu", repr=False)

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """[batch, seq] -> [batch, seq, embed_dim]"""
        import torch

        x = torch.from_numpy(np.asarray(token_ids, dtype=np.int64)).to(self._device)
        with torch.inference_mode():
            emb = self._model._embed_linear(x)
        return emb.cpu().numpy()

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """[batch, seq] -> [batch, seq, vocab] logits"""
        import torch

        x = torch.from_numpy(np.asarray(token_ids, dtype=np.int64)).to(self._device)
        with torch.inference_mode():
            logits = self._model(x)
        return logits.cpu().numpy()

    def forward_captured(self, token_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Forward pass returning all internals.

        Returns dict with:
            logits:                   [batch, seq, vocab]
            substrate_states:         [batch, seq, n_modes]
            embedding:                [batch, seq, embed_dim]
            route_weights:            [batch*seq, n_experts] or None
            band_logits:              list of [batch, seq, vocab] per band, or None
            local_logits:             [batch, seq, vocab] or None
            temporal_attn_weights:    [batch, heads, seq, M] or None
            temporal_attn_output:     [batch, seq, state_dim] or None
            temporal_snapshot_interval: int or None
            overwrite_gate_values:    [batch, seq, n_modes] or None
            mode_selector_mask:       [batch, seq, n_modes] or None
            magnitude_before_norm:    [batch, seq, 1] or None
        """
        import torch

        x = torch.from_numpy(np.asarray(token_ids, dtype=np.int64)).to(self._device)
        result = {}

        with torch.inference_mode():
            # Substrate path
            states, x_embed = self._model._linear_states(x)
            result["substrate_states"] = states.cpu().numpy()
            result["embedding"] = x_embed.cpu().numpy()

            # Temporal attention internals (stored during _linear_states)
            ta = getattr(self._model, '_temporal_attention', None)
            if ta is not None and hasattr(ta, '_last_attn_weights') and ta._last_attn_weights is not None:
                result["temporal_attn_weights"] = ta._last_attn_weights.cpu().numpy()
                result["temporal_attn_output"] = ta._last_output.cpu().numpy()
                result["temporal_snapshot_interval"] = getattr(
                    self._model, '_temporal_snapshot_interval', None)
            else:
                result["temporal_attn_weights"] = None
                result["temporal_attn_output"] = None
                result["temporal_snapshot_interval"] = None

            # Lasso matrix (stored during _gated_delta_states if lasso_rotation)
            lasso = getattr(self._model, '_last_lasso_matrix', None)
            result["lasso_matrix"] = lasso.cpu().numpy() if lasso is not None else None

            # Gated delta gate values (stored during _gated_delta_states)
            gd_write = getattr(self._model, '_last_gated_delta_write', None)
            result["gated_delta_write"] = gd_write.cpu().numpy() if gd_write is not None else None
            result["gated_delta_retain"] = (
                self._model._last_gated_delta_retain.cpu().numpy()
                if getattr(self._model, '_last_gated_delta_retain', None) is not None else None
            )
            result["gated_delta_erase"] = (
                self._model._last_gated_delta_erase.cpu().numpy()
                if getattr(self._model, '_last_gated_delta_erase', None) is not None else None
            )

            # Overwrite gate values (stored during _linear_states)
            og = getattr(self._model, '_overwrite_gate', None)
            result["overwrite_gate_values"] = (
                og._last_gate_values.cpu().numpy()
                if og is not None and hasattr(og, '_last_gate_values') and og._last_gate_values is not None
                else None
            )

            # Mode selector mask (stored during _linear_states)
            ms = getattr(self._model, '_mode_selector', None)
            result["mode_selector_mask"] = (
                ms._last_weights.cpu().numpy()
                if ms is not None and hasattr(ms, '_last_weights') and ms._last_weights is not None
                else None
            )

            # Magnitude before normalization (stored during _linear_states)
            mn = getattr(self._model, '_magnitude_normalizer', None)
            result["magnitude_before_norm"] = (
                mn._last_pre_norm.cpu().numpy()
                if mn is not None and hasattr(mn, '_last_pre_norm') and mn._last_pre_norm is not None
                else None
            )

            # Sticky register internals (if present)
            if getattr(self._model, "_sticky_registers", 0) > 0:
                ws = torch.sigmoid(self._model._sticky_write_gate(x_embed))
                result["sticky_write_strength"] = ws.cpu().numpy()
            else:
                result["sticky_write_strength"] = None

            # Readout: capture per-band and routing
            n_bands = getattr(self._model.config, "readout_bands", 1)
            if n_bands > 1 and hasattr(self._model, "_band_readouts"):
                modes = states.shape[-1]
                band_size = modes // n_bands
                band_logits_list = []
                _static = getattr(self._model, "_static_band_indices", set())
                for i, readout in enumerate(self._model._band_readouts):
                    if i in _static:
                        band_logits_list.append(readout(x_embed).cpu().numpy())
                    else:
                        band_states = states[..., i * band_size : (i + 1) * band_size]
                        band_features = torch.cat([band_states, x_embed], dim=-1)
                        band_logits_list.append(readout(band_features).cpu().numpy())
                result["band_logits"] = band_logits_list
                logits_linear = torch.from_numpy(sum(bl for bl in band_logits_list)).to(self._device)
            else:
                result["band_logits"] = None
                logits_linear = self._model._linear_logits(x)

            # Expert routing (captured during _linear_logits or manual forward)
            route = getattr(self._model.linear_readout, "_last_route", None)
            result["route_weights"] = route.cpu().numpy() if route is not None else None

            # Local path
            if self._model.config.enable_local:
                logits_local = self._model._local_logits(x)
                result["local_logits"] = logits_local.cpu().numpy()
                if self._model.gate_proj is None:
                    gate = self._model.config.local_scale
                else:
                    ent_l, max_l, var_l = self._model._logit_features(logits_linear)
                    ent_r, max_r, var_r = self._model._logit_features(logits_local)
                    features = torch.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], dim=-1)
                    gate = torch.sigmoid(self._model.gate_proj(features)) * self._model.config.local_scale
                logits = logits_linear + gate * logits_local
            else:
                result["local_logits"] = None
                logits = logits_linear

            result["logits"] = logits.cpu().numpy()

        return result

    def weights(self) -> dict[str, np.ndarray]:
        """All learned parameters as numpy arrays."""
        out = {}
        for name, param in self._model.named_parameters():
            out[name] = param.detach().cpu().numpy()
        for name, buf in self._model.named_buffers():
            out[name] = buf.detach().cpu().numpy()
        return out


def _reconstruct_config(result: dict):
    """Rebuild CausalBankConfig from a result JSON."""
    from decepticons.causal_bank import CausalBankConfig

    model = result["model"]
    fields = {}
    fields["max_seq_len"] = result["config"]["train"]["seq_len"]
    fields["init_seed"] = model.get("init_seed", model.get("seed", 42))

    _DIRECT_FIELDS = [
        "linear_modes", "embedding_dim", "linear_readout_kind",
        "linear_readout_depth", "linear_readout_num_experts", "readout_bands",
        "local_window", "local_scale", "mix_mode", "share_embedding",
        "linear_impl", "linear_half_life_min", "linear_half_life_max",
        "oscillatory_frac", "oscillatory_schedule",
        "oscillatory_period_min", "oscillatory_period_max",
        "input_proj_scheme", "substrate_mode", "state_dim", "state_impl",
        "num_heads", "static_bank_gate", "bank_gate_span",
        "num_blocks", "block_mixing_ratio", "num_hemispheres",
        "fast_hemisphere_ratio", "fast_lr_mult", "local_poly_order",
        "substrate_poly_order", "block_stride", "training_noise",
        "adaptive_reg", "trust_routing", "band_experts",
        "magnitude_normalize", "overwrite_gate", "mode_selector",
        "temporal_attention", "temporal_attention_heads", "temporal_attention_head_dim",
    ]
    for f in _DIRECT_FIELDS:
        if f in model and model[f] is not None:
            fields[f] = model[f]

    if "linear_hidden" in model:
        fields["linear_hidden"] = tuple(model["linear_hidden"])
    if "local_hidden" in model:
        fields["local_hidden"] = tuple(model["local_hidden"])
    if "band_experts" in model and model["band_experts"]:
        fields["band_experts"] = tuple(model["band_experts"])

    return CausalBankConfig(**fields)


def _infer_band_experts(state_dict: dict, n_bands: int) -> list[int] | None:
    """Infer per-band expert counts from state dict keys.

    Returns list like [8, 4, 2, 0] or None if uniform.
    Band with only .weight/.bias (no router/experts_in) → 0 (static).
    Non-routed readouts (TiedRecursiveReadout, MLP) → None (skip inference).
    """
    import re

    band_expert_counts = []
    any_asym = False

    for bi in range(n_bands):
        prefix = f"_band_readouts.{bi}"
        band_keys = [k for k in state_dict if k.startswith(prefix)]
        if not band_keys:
            band_expert_counts.append(0)
            any_asym = True
            continue
        # Non-routed readout types — don't infer expert counts, let config handle it
        has_in_proj = any(k.startswith(f"{prefix}.in_proj.") for k in band_keys)
        has_layers = any(k.startswith(f"{prefix}.layers.") for k in band_keys)
        if has_in_proj or has_layers:
            # TiedRecursiveReadout or MLP — band_experts is irrelevant
            return None
        # Check for routed experts
        has_router = any(k.startswith(f"{prefix}.router.") for k in band_keys)
        if not has_router:
            # Static band: only .weight and .bias, no structured readout
            band_expert_counts.append(0)
            any_asym = True
            continue
        # Count experts by finding max expert index in experts_in
        max_idx = -1
        for k in state_dict:
            m = re.match(rf"_band_readouts\.{bi}\.experts_in\.(\d+)\.", k)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        n_exp = max_idx + 1 if max_idx >= 0 else 2
        band_expert_counts.append(n_exp)
        if n_exp != band_expert_counts[0]:
            any_asym = True

    if any_asym or (band_expert_counts and 0 in band_expert_counts):
        return band_expert_counts
    return None


def _infer_band_readout_kind(state_dict: dict) -> str | None:
    """Detect per-band readout type from state dict keys."""
    band0_keys = [k for k in state_dict if k.startswith("_band_readouts.0.")]
    if not band0_keys:
        return None
    if any(".in_proj." in k for k in band0_keys):
        return "tied_recursive"
    if any(".layers." in k for k in band0_keys):
        return "mlp"
    if any(".router." in k for k in band0_keys):
        return "routed_sqrelu_experts"
    # Just .weight/.bias → static (handled by band_experts=(0,...))
    return None


def load_checkpoint(
    path: str,
    *,
    result_json: str | None = None,
    tokenizer_path: str | None = None,
    device: str = "cpu",
) -> CausalBankInference:
    """Load a causal bank checkpoint for inference/analysis.

    Args:
        path: Path to .checkpoint.pt file
        result_json: Path to result .json (auto-detected from checkpoint path if omitted)
        tokenizer_path: Path to sentencepiece .model file (optional)
        device: "cpu", "cuda", or "mps"
    """
    import torch
    from decepticons.models.causal_bank_torch import CausalBankModel

    ckpt_path = Path(path)

    # Auto-detect result JSON
    if result_json is None:
        candidate = ckpt_path.with_name(ckpt_path.name.replace(".checkpoint.pt", ".json"))
        if candidate.exists():
            result_json = str(candidate)
        else:
            raise FileNotFoundError(
                f"No result JSON found. Tried {candidate}. Pass result_json= explicitly."
            )

    result = json.loads(Path(result_json).read_text())
    config = _reconstruct_config(result)
    vocab_size = result["dataset"].get("vocab_size", 1024)

    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)

    # Strip torch.compile _orig_mod. prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    # Infer true linear_modes from state dict (JSON may undercount)
    if "linear_in_proj" in state_dict:
        true_modes = state_dict["linear_in_proj"].shape[1]
        if true_modes != config.linear_modes:
            from dataclasses import replace as dc_replace
            config = dc_replace(config, linear_modes=true_modes)

    # Infer substrate transforms from state dict
    _transform_flags = {
        "_magnitude_normalizer.": "magnitude_normalize",
        "_overwrite_gate.": "overwrite_gate",
        "_mode_selector.": "mode_selector",
        "_temporal_attention.": "temporal_attention",
        "_gated_delta_rotation_proj.": "complex_rotation",
        "_lasso_proj.": "lasso_rotation",
        "_so5_proj.": "so5_rotation",
        "_quat_proj.": "quaternion_rotation",
    }
    # Position signal: detect from gate weight shape (embed_dim + 1)
    _position_signal_detected = False
    gd_retain_key = "_gated_delta_retain_gate.weight"
    if gd_retain_key in state_dict:
        gate_in = state_dict[gd_retain_key].shape[1]
        embed_dim = config.embedding_dim
        if gate_in == embed_dim + 1 and not getattr(config, "position_signal", False):
            _position_signal_detected = True
    transform_updates = {}
    for prefix, flag in _transform_flags.items():
        if any(k.startswith(prefix) for k in state_dict) and not getattr(config, flag, False):
            transform_updates[flag] = True
    if _position_signal_detected:
        transform_updates["position_signal"] = True
    if transform_updates:
        from dataclasses import replace as dc_replace
        config = dc_replace(config, **transform_updates)

    # Infer num_blocks from state dict
    block_indices = set()
    for k in state_dict:
        if k.startswith("_block_layers."):
            idx = k.split(".")[1]
            block_indices.add(int(idx))
    if block_indices:
        inferred_blocks = max(block_indices) + 2  # N block_layers = N+1 total blocks
        if inferred_blocks != getattr(config, "num_blocks", 1):
            from dataclasses import replace as dc_replace
            config = dc_replace(config, num_blocks=inferred_blocks)

    # Infer asymmetric band_experts from state dict if not in config
    n_bands = getattr(config, "readout_bands", 1)
    if n_bands > 1 and not getattr(config, "band_experts", ()):
        band_experts = _infer_band_experts(state_dict, n_bands)
        if band_experts:
            from dataclasses import replace as dc_replace
            config = dc_replace(config, band_experts=tuple(band_experts))

    # Infer per-band readout kind from state dict if it differs from config
    if n_bands > 1:
        band_readout_kind = _infer_band_readout_kind(state_dict)
        if band_readout_kind and band_readout_kind != getattr(config, "linear_readout_kind", "mlp"):
            from dataclasses import replace as dc_replace
            # The main linear_readout uses the config's kind; override only for bands
            # by setting the config kind to match what the bands actually have.
            # Save original for the main readout fixup after load.
            _original_readout_kind = config.linear_readout_kind
            config = dc_replace(config, linear_readout_kind=band_readout_kind)
        else:
            _original_readout_kind = None
    else:
        _original_readout_kind = None

    model = CausalBankModel(vocab_size=vocab_size, config=config).to(device)

    if _original_readout_kind:
        # The main linear_readout was built with the band kind; rebuild it with the original
        # This handles the case where bands use MLP but linear_readout uses routed experts
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    model.eval()

    # Half-lives from the decay buffer
    decays = model.linear_decays.detach().cpu().numpy()
    half_lives = np.log(0.5) / np.log(np.clip(decays, 1e-12, 1.0 - 1e-12))

    # Tokenizer (optional)
    tokenizer = None
    if tokenizer_path:
        try:
            import sentencepiece as spm
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.Load(tokenizer_path)
        except ImportError:
            pass

    # Config dict for external consumers
    config_dict = {
        "model_type": "causal_bank",
        "n_modes": config.linear_modes,
        "n_experts": config.linear_readout_num_experts,
        "n_bands": getattr(config, "readout_bands", 1),
        "embed_dim": config.embedding_dim,
        "hidden_dim": config.linear_hidden[0] if config.linear_hidden else 0,
        "vocab_size": vocab_size,
        "seq_len": config.max_seq_len,
        "readout_kind": config.linear_readout_kind,
        "substrate_mode": config.substrate_mode,
        "half_life_min": config.linear_half_life_min,
        "half_life_max": config.linear_half_life_max,
        "local_window": config.local_window,
        "params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    return CausalBankInference(
        config=config_dict,
        half_lives=half_lives,
        tokenizer=tokenizer,
        _model=model,
        _device=device,
    )
