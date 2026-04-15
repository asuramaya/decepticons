from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from decepticons.causal_bank import (
    CausalBankConfig,
    build_linear_bank,
    learnable_substrate_keys,
    osc_pair_count,
    scale_config,
    validate_config,
)

from .common import _embedding_uniform, _rng_for, _xavier_uniform
from .hash_memory import HashMemory
from .readouts_torch import (
    MLP,
    GRUReadout,
    RoutedSquaredReLUReadout,
    TiedEmbedReadout,
    TiedRecursiveReadout,
    _copy_embedding_,
    _copy_linear_,
)
from .substrate_transforms import (
    MagnitudeNormalizer,
    ModeSelector,
    OverwriteGate,
    SubstrateBankRouter,
    TemporalAttention,
)

__all__ = [
    "CausalBankConfig",
    "CausalBankModel",
    "scale_config",
]

class CausalBankModel(nn.Module):
    def __init__(self, vocab_size: int, config=None):
        super().__init__()
        if config is None:
            config = CausalBankConfig()
        validate_config(config)

        self.vocab_size = vocab_size
        self.config = config
        self.shared_embedding = None
        self.linear_embedding = None
        self.local_embedding = None
        self.linear_readout = None
        self.local_readout = None
        self.gate_proj = None
        self.bank_gate_logits = None
        self.non_osc_modes = 0
        self.osc_mode_count = 0

        if config.share_embedding and config.enable_linear and config.enable_local:
            self.shared_embedding = nn.Embedding(vocab_size, config.embedding_dim)

        if config.enable_linear:
            if self.shared_embedding is None:
                self.linear_embedding = nn.Embedding(vocab_size, config.embedding_dim)
            in_proj, decays, kernel = build_linear_bank(config)
            learnable_keys = learnable_substrate_keys(config)

            if "linear_in_proj" in learnable_keys:
                self.linear_in_proj = nn.Parameter(torch.from_numpy(in_proj))
            else:
                self.register_buffer("linear_in_proj", torch.from_numpy(in_proj))

            if "linear_decays" in learnable_keys:
                self.linear_decays = nn.Parameter(torch.from_numpy(decays.astype(np.float32)))
            else:
                self.register_buffer("linear_decays", torch.from_numpy(decays.astype(np.float32)))

            self._learned_recurrence = "recurrence_gate" in learnable_keys
            if self._learned_recurrence:
                # Per-mode gate: input -> sigmoid gate value
                # W_gate: [embedding_dim, linear_modes], b_gate: [linear_modes]
                self._gate_weight = nn.Parameter(torch.zeros(config.embedding_dim, config.linear_modes))
                self._gate_bias = nn.Parameter(torch.zeros(config.linear_modes))
                # Initialize gate bias so initial gate ≈ decay values (smooth start)
                with torch.no_grad():
                    self._gate_bias.copy_(
                        torch.log(self.linear_decays / (1 - self.linear_decays + 1e-6)).clamp(-5, 5)
                    )
            else:
                self._learned_recurrence = False

            if "linear_decays" in learnable_keys:
                # Kernel will be recomputed in forward() from learnable decays
                self._recompute_kernel = True
                self.register_buffer("_kernel_time_idx", torch.arange(config.max_seq_len))
                self.linear_kernel = None
            elif config.linear_impl == "kernel":
                self._recompute_kernel = False
                self.register_buffer("linear_kernel", torch.from_numpy(kernel))
            else:
                self._recompute_kernel = False
                self.linear_kernel = None
            linear_readout_in_dim = config.linear_modes + config.embedding_dim
            if config.linear_readout_kind == "mlp":
                self.linear_readout = MLP(
                    linear_readout_in_dim,
                    config.linear_hidden,
                    vocab_size,
                )
            elif config.linear_readout_kind == "tied_recursive":
                if len(config.linear_hidden) != 1:
                    raise ValueError(
                        "causal-bank tied_recursive linear readout currently expects exactly one hidden width."
                    )
                self.linear_readout = TiedRecursiveReadout(
                    linear_readout_in_dim,
                    config.linear_hidden[0],
                    vocab_size,
                    config.linear_readout_depth,
                )
            elif config.linear_readout_kind == "recurrent":
                self.linear_readout = GRUReadout(linear_readout_in_dim, vocab_size, config)
            elif config.linear_readout_kind == "tied_embed_readout":
                if len(config.linear_hidden) != 1:
                    raise ValueError(
                        "causal-bank tied_embed_readout expects exactly one hidden width."
                    )
                self.linear_readout = TiedEmbedReadout(
                    linear_readout_in_dim,
                    config.linear_hidden[0],
                    config.embedding_dim,
                    config.linear_readout_num_experts,
                    normalize_embed=getattr(config, 'tied_readout_normalize', False),
                )
            else:
                if len(config.linear_hidden) != 1:
                    raise ValueError(
                        "causal-bank routed_sqrelu_experts linear readout currently expects exactly one hidden width."
                    )
                self.linear_readout = RoutedSquaredReLUReadout(
                    linear_readout_in_dim,
                    config.linear_hidden[0],
                    vocab_size,
                    config.linear_readout_num_experts,
                )
            # Banded readout: split modes by timescale, one readout per band
            n_bands = getattr(config, 'readout_bands', 1)
            if n_bands > 1:
                band_size = config.linear_modes // n_bands
                band_in_dim = band_size + config.embedding_dim
                band_hidden = tuple(h // n_bands for h in config.linear_hidden)
                readout_kind = getattr(config, 'linear_readout_kind', 'mlp')
                # Per-band expert counts: () = uniform, (8,4,2,0) = asymmetric, 0 = static bias
                _band_experts = getattr(config, 'band_experts', ())
                if not _band_experts:
                    _band_experts = tuple([config.linear_readout_num_experts] * n_bands)
                _band_readouts = []
                _static_band_indices = []
                for bi in range(n_bands):
                    n_exp = _band_experts[bi] if bi < len(_band_experts) else config.linear_readout_num_experts
                    if n_exp == 0:
                        # Static band: learnable bias vector, no forward computation
                        _static_band_indices.append(bi)
                        _band_readouts.append(nn.Linear(config.embedding_dim, vocab_size))
                    elif readout_kind == "tied_recursive":
                        _band_readouts.append(
                            TiedRecursiveReadout(band_in_dim, band_hidden[0], vocab_size, config.linear_readout_depth))
                    elif readout_kind == "tied_embed_readout" and n_exp >= 2:
                        _band_readouts.append(
                            TiedEmbedReadout(band_in_dim, band_hidden[0], config.embedding_dim, n_exp,
                                             normalize_embed=getattr(config, 'tied_readout_normalize', False)))
                    elif readout_kind == "routed_sqrelu_experts" and n_exp >= 2:
                        _band_readouts.append(
                            RoutedSquaredReLUReadout(band_in_dim, band_hidden[0], vocab_size, n_exp))
                    else:
                        _band_readouts.append(MLP(band_in_dim, band_hidden, vocab_size))
                self._band_readouts = nn.ModuleList(_band_readouts)
                self._static_band_indices = set(_static_band_indices)

            # Write-side transforms: route and overwrite (before/during substrate)
            self._use_substrate_bank_router = getattr(config, 'substrate_bank_router', False)
            self._substrate_n_banks = getattr(config, 'substrate_n_banks', 4)
            if self._use_substrate_bank_router:
                self._substrate_bank_router = SubstrateBankRouter(
                    config.embedding_dim,
                    self._substrate_n_banks,
                )

            self._use_overwrite_gate = getattr(config, 'overwrite_gate', False)
            if self._use_overwrite_gate:
                self._overwrite_gate = OverwriteGate(
                    config.embedding_dim,
                    config.linear_modes,
                )

            # Read-side transforms: normalize, select, attend (after substrate)
            self._use_magnitude_norm = getattr(config, 'magnitude_normalize', False)
            if self._use_magnitude_norm:
                self._magnitude_normalizer = MagnitudeNormalizer(
                    config.linear_modes,
                    keep_magnitude=getattr(config, 'magnitude_keep', True),
                )

            self._use_mode_selector = getattr(config, 'mode_selector', False)
            if self._use_mode_selector:
                self._mode_selector = ModeSelector(
                    config.embedding_dim,
                    config.linear_modes,
                )

            self._use_temporal_attention = getattr(config, 'temporal_attention', False)
            self._temporal_snapshot_interval = getattr(config, 'temporal_snapshot_interval', 64)
            if self._use_temporal_attention:
                self._temporal_attention = TemporalAttention(
                    config.linear_modes,
                    num_heads=getattr(config, 'temporal_attention_heads', 2),
                    head_dim=getattr(config, 'temporal_attention_head_dim', 32),
                )

            # Hash memory: order-preserving token retrieval
            self._use_hash_memory = getattr(config, 'hash_memory', False)
            if self._use_hash_memory:
                self._hash_memory = HashMemory(
                    config.embedding_dim,
                    memory_dim=getattr(config, 'hash_memory_dim', 64),
                    num_slots=getattr(config, 'hash_memory_slots', 64),
                )

            # Substrate polynomial expansion
            self._substrate_poly = getattr(config, 'substrate_poly_order', 1)
            if self._substrate_poly >= 2 and config.enable_linear:
                # Project modes to smaller dim, then expand
                poly_dim = min(64, config.linear_modes // 4)
                self._substrate_poly_proj = nn.Linear(config.linear_modes, poly_dim)
                quad_dim = poly_dim * (poly_dim + 1) // 2
                extra_dim = quad_dim
                if self._substrate_poly >= 3:
                    extra_dim += poly_dim
                # Adjust the linear readout input dimension
                new_readout_dim = config.linear_modes + config.embedding_dim + extra_dim
                # Recreate linear readout with expanded input
                if config.linear_readout_kind == "mlp":
                    self.linear_readout = MLP(new_readout_dim, config.linear_hidden, vocab_size)
                elif config.linear_readout_kind == "tied_recursive":
                    self.linear_readout = TiedRecursiveReadout(
                        new_readout_dim, config.linear_hidden[0], vocab_size, config.linear_readout_depth,
                    )
                elif config.linear_readout_kind == "recurrent":
                    self.linear_readout = GRUReadout(new_readout_dim, vocab_size, config)
                else:
                    self.linear_readout = RoutedSquaredReLUReadout(
                        new_readout_dim, config.linear_hidden[0], vocab_size, config.linear_readout_num_experts,
                    )

            osc_pairs = osc_pair_count(config)
            self.non_osc_modes = config.linear_modes - 2 * osc_pairs
            self.osc_mode_count = 2 * osc_pairs
            if config.static_bank_gate and self.osc_mode_count > 0:
                self.bank_gate_logits = nn.Parameter(torch.zeros((2,), dtype=torch.float32))

        if config.enable_local:
            if self.shared_embedding is None:
                self.local_embedding = nn.Embedding(vocab_size, config.embedding_dim)
            base_local_dim = config.local_window * config.embedding_dim
            self._poly_order = getattr(config, 'local_poly_order', 1)
            if self._poly_order >= 2:
                poly_proj_dim = min(64, base_local_dim // 4)
                self._poly_proj = nn.Linear(base_local_dim, poly_proj_dim)
                # Quadratic features: poly_proj_dim * (poly_proj_dim + 1) / 2
                quad_dim = poly_proj_dim * (poly_proj_dim + 1) // 2
                expanded_dim = base_local_dim + quad_dim
                if self._poly_order >= 3:
                    # Cubic: element-wise cube of projection (cheap approximation)
                    expanded_dim += poly_proj_dim
                self.local_readout = MLP(expanded_dim, config.local_hidden, vocab_size)
            else:
                self.local_readout = MLP(base_local_dim, config.local_hidden, vocab_size)

        if config.enable_linear and config.enable_local and config.mix_mode == "gated":
            self.gate_proj = nn.Linear(6, 1)

        # Stacked substrate blocks
        self.num_blocks = getattr(config, 'num_blocks', 1)
        if self.num_blocks > 1 and config.enable_linear:
            mixing_dim = max(int(config.linear_modes * getattr(config, 'block_mixing_ratio', 0.25)), 1)
            self._block_layers = nn.ModuleList()
            for _ in range(self.num_blocks - 1):
                self._block_layers.append(nn.Sequential(
                    nn.Linear(config.linear_modes, mixing_dim),
                    nn.ReLU(),
                    nn.Linear(mixing_dim, config.linear_modes),
                ))

        # --- Recurrent state augment / learned state substrate ---
        self._state_impl = getattr(config, "state_impl", "scan")
        self._substrate_mode = getattr(config, "substrate_mode", "frozen")
        self._use_gated_retention_substrate = self._substrate_mode == "gated_retention"
        self._use_gated_delta_substrate = self._substrate_mode == "gated_delta"
        self._use_primary_state_substrate = self._use_gated_retention_substrate or self._use_gated_delta_substrate
        self._use_state_augment = getattr(config, "state_dim", 0) > 0 and not self._use_primary_state_substrate
        self._state_num_heads = max(int(getattr(config, "num_heads", 1)), 1)
        self._num_hemispheres = getattr(config, "num_hemispheres", 1)
        if self._use_state_augment or self._use_primary_state_substrate:
            sd = int(config.state_dim)
            ed = int(config.embedding_dim)
            self._state_head_dim = max(sd // self._state_num_heads, 1)
            self._state_width = self._state_num_heads * self._state_head_dim

            if self._state_impl == "scan":
                self._ssm_A = nn.Parameter(torch.zeros(self._state_num_heads, self._state_head_dim))
                self._ssm_B_proj = nn.Linear(ed, self._state_width, bias=False)
                self._ssm_C_proj = nn.Linear(ed, self._state_width, bias=False)
                self._ssm_D = nn.Parameter(torch.ones(1))
                self._ssm_out_proj = nn.Linear(self._state_width, config.linear_modes, bias=False)

                with torch.no_grad():
                    init_decays = torch.linspace(-0.1, -5.0, sd).view(self._state_num_heads, self._state_head_dim)
                    self._ssm_A.copy_(init_decays)

                if self._num_hemispheres == 2:
                    fast_dim = max(int(sd * getattr(config, "fast_hemisphere_ratio", 0.25)), 1)
                    slow_dim = max(sd - fast_dim, 0)
                    self._fast_dim = fast_dim
                    self._slow_dim = slow_dim
                    with torch.no_grad():
                        flat = self._ssm_A.view(-1)
                        flat[:fast_dim] = torch.linspace(-1.0, -3.0, fast_dim)
                        if slow_dim > 0:
                            flat[fast_dim:] = torch.linspace(-0.01, -0.5, slow_dim)
                if self._use_gated_delta_substrate:
                    # Position signal: gates receive [x_embed, log(1+t)] breaking
                    # shift invariance. The substrate can learn position-dependent
                    # gating without explicit position embeddings.
                    self._use_position_signal = getattr(config, "position_signal", False)
                    gate_in_dim = ed + 1 if self._use_position_signal else ed
                    self._gated_delta_retain_gate = nn.Linear(gate_in_dim, self._state_width)
                    self._gated_delta_write_gate = nn.Linear(gate_in_dim, self._state_width)
                    self._gated_delta_erase_gate = nn.Linear(gate_in_dim, self._state_width)
                    with torch.no_grad():
                        self._gated_delta_retain_gate.weight.zero_()
                        self._gated_delta_write_gate.weight.zero_()
                        self._gated_delta_erase_gate.weight.zero_()
                        self._gated_delta_retain_gate.bias.fill_(2.0)
                        self._gated_delta_write_gate.bias.fill_(-1.0)
                        self._gated_delta_erase_gate.bias.fill_(-2.0)
                    # Complex rotation: input-dependent phase rotation per mode pair.
                    # SO(2) — commutative. Encodes content sensitivity, not order.
                    self._use_complex_rotation = getattr(config, "complex_rotation", False)
                    if self._use_complex_rotation:
                        assert self._state_head_dim % 2 == 0, (
                            f"complex_rotation requires even state_head_dim, got {self._state_head_dim}"
                        )
                        self._rotation_half_dim = self._state_head_dim // 2
                        self._gated_delta_rotation_proj = nn.Linear(ed, self._state_num_heads * self._rotation_half_dim)
                        with torch.no_grad():
                            self._gated_delta_rotation_proj.weight.zero_()
                            self._gated_delta_rotation_proj.bias.zero_()
                    # Lasso rotation: 2x2 matrix transition per mode pair.
                    # Noncommutative — "A B" ≠ "B A" because 2x2 matmul doesn't commute.
                    # Like a lasso: rotation plane tilts based on input, and the composition
                    # of tilts encodes the ORDER of the sequence.
                    self._use_lasso_rotation = getattr(config, "lasso_rotation", False)
                    if self._use_lasso_rotation:
                        assert self._state_head_dim % 2 == 0, (
                            f"lasso_rotation requires even state_head_dim, got {self._state_head_dim}"
                        )
                        self._rotation_half_dim = self._state_head_dim // 2
                        n_pairs = self._state_num_heads * self._rotation_half_dim
                        # 4 matrix elements per pair: [[m00, m01], [m10, m11]]
                        self._lasso_proj = nn.Linear(gate_in_dim, n_pairs * 4)
                        with torch.no_grad():
                            # Init near identity: [[1,0],[0,1]] via bias
                            self._lasso_proj.weight.zero_()
                            bias = torch.zeros(n_pairs * 4)
                            bias[0::4] = 0.5   # m00 → sigmoid(0.5) ≈ 0.62
                            bias[1::4] = -3.0  # m01 → sigmoid(-3) ≈ 0.05 (near zero)
                            bias[2::4] = -3.0  # m10 → sigmoid(-3) ≈ 0.05 (near zero)
                            bias[3::4] = 0.5   # m11 → sigmoid(0.5) ≈ 0.62
                            self._lasso_proj.bias.copy_(bias)
                    # Quaternion rotation: Hamilton product per mode quad.
                    # SO(3) noncommutative, associative → scannable.
                    # head_dim must be 4 (each head IS a quaternion).
                    # Quintic rotation: 5×5 block matrix per head.
                    # SO(5) is non-solvable (Abel-Ruffini). One layer of 5×5 can
                    # express group structures that no stacking of diagonal/complex/
                    # quaternion can.
                    self._use_quintic_rotation = getattr(config, "quintic_rotation", False)
                    if self._use_quintic_rotation:
                        block_size = self._state_head_dim
                        self._quintic_block_size = block_size
                        self._quintic_proj = nn.Linear(gate_in_dim, self._state_num_heads * block_size * block_size)
                        with torch.no_grad():
                            self._quintic_proj.weight.zero_()
                            bias = torch.zeros(self._state_num_heads * block_size * block_size)
                            for h_idx in range(self._state_num_heads):
                                for i in range(block_size):
                                    bias[h_idx * block_size * block_size + i * block_size + i] = 2.0
                            self._quintic_proj.bias.copy_(bias)
                    # SO(5) rotation via Lie algebra exponential.
                    # Learn skew-symmetric matrix (10 params per head), exponentiate
                    # to get guaranteed rotation. Spectral radius = 1 always. Can't diverge.
                    self._use_so5_rotation = getattr(config, "so5_rotation", False)
                    if self._use_so5_rotation:
                        block_size = self._state_head_dim
                        self._so5_block_size = block_size
                        # Lie algebra of SO(n) has n*(n-1)/2 free parameters
                        lie_dim = block_size * (block_size - 1) // 2
                        self._so5_lie_dim = lie_dim
                        self._so5_proj = nn.Linear(gate_in_dim, self._state_num_heads * lie_dim)
                        # Upper triangular indices for building skew-symmetric matrix
                        self._so5_triu_indices = torch.triu_indices(block_size, block_size, offset=1)
                        with torch.no_grad():
                            # Zero init: exp(0) = identity rotation
                            self._so5_proj.weight.zero_()
                            self._so5_proj.bias.zero_()
                    self._use_quaternion_rotation = getattr(config, "quaternion_rotation", False)
                    if self._use_quaternion_rotation:
                        assert self._state_head_dim == 4, (
                            f"quaternion_rotation requires state_head_dim=4, got {self._state_head_dim}. "
                            f"Use --state-dim {self._state_num_heads * 4} --num-heads {self._state_num_heads}"
                        )
                        # Project embedding to unit quaternion per head
                        self._quat_proj = nn.Linear(gate_in_dim, self._state_num_heads * 4)
                        with torch.no_grad():
                            self._quat_proj.weight.zero_()
                            # Init as identity quaternion (1,0,0,0)
                            bias = torch.zeros(self._state_num_heads * 4)
                            bias[0::4] = 1.0  # w=1
                            self._quat_proj.bias.copy_(bias)
            else:
                self._retention_q_proj = nn.Linear(ed, self._state_width, bias=False)
                self._retention_k_proj = nn.Linear(ed, self._state_width, bias=False)
                self._retention_v_proj = nn.Linear(ed, self._state_width, bias=False)
                self._retention_out_proj = nn.Linear(self._state_width, config.linear_modes, bias=False)
                self._retention_logit_decay = nn.Parameter(torch.zeros(self._state_num_heads))
                self._retention_D = nn.Parameter(torch.ones(1))
                with torch.no_grad():
                    if self._num_hemispheres == 2 and self._state_num_heads > 1:
                        fast_heads = max(int(round(self._state_num_heads * getattr(config, "fast_hemisphere_ratio", 0.25))), 1)
                        slow_heads = max(self._state_num_heads - fast_heads, 0)
                        self._retention_logit_decay[:fast_heads] = torch.linspace(-1.5, 0.0, fast_heads)
                        if slow_heads > 0:
                            self._retention_logit_decay[fast_heads:] = torch.linspace(2.0, 4.0, slow_heads)
                    else:
                        self._retention_logit_decay.copy_(torch.linspace(0.5, 3.5, self._state_num_heads))
                if self._use_gated_retention_substrate:
                    self._gated_retention_retain_gate = nn.Linear(ed, self._state_num_heads)
                    self._gated_retention_write_gate = nn.Linear(ed, self._state_num_heads)
                    self._gated_retention_erase_gate = nn.Linear(ed, self._state_num_heads)
                    with torch.no_grad():
                        self._gated_retention_retain_gate.weight.zero_()
                        self._gated_retention_write_gate.weight.zero_()
                        self._gated_retention_erase_gate.weight.zero_()
                        self._gated_retention_retain_gate.bias.fill_(2.0)
                        self._gated_retention_write_gate.bias.fill_(-1.0)
                        self._gated_retention_erase_gate.bias.fill_(-2.0)

        # --- Sticky registers: persistent memory for surprising tokens ---
        self._sticky_registers = getattr(config, "sticky_registers", 0)
        if self._sticky_registers > 0:
            nr = self._sticky_registers
            ed = config.embedding_dim
            # Write gate: embedding → scalar write strength per register
            self._sticky_write_gate = nn.Linear(ed, nr)
            # Write value: embedding → register content
            self._sticky_write_proj = nn.Linear(ed, nr)
            # Read projection: registers → linear_modes (adds to substrate states)
            self._sticky_read_proj = nn.Linear(nr, config.linear_modes, bias=False)
            # Decay: very slow, derived from config half-life
            sticky_hl = getattr(config, "sticky_half_life", 1000.0)
            sticky_decay = float(np.exp(np.log(0.5) / max(sticky_hl, 1.0)))
            self.register_buffer("_sticky_decay", torch.tensor(sticky_decay, dtype=torch.float32))
            with torch.no_grad():
                # Init write gate biased toward NOT writing (most tokens aren't surprising)
                self._sticky_write_gate.bias.fill_(-2.0)
                self._sticky_write_gate.weight.zero_()

        # --- Patch encoding ---
        self._patch_size = getattr(config, 'patch_size', 1)
        self._patch_causal_mode = getattr(config, 'patch_causal_decoder', 'none')
        if self._patch_size > 1:
            # Encoder: patch_size * embed_dim → embed_dim
            self._patch_encoder = nn.Linear(self._patch_size * config.embedding_dim, config.embedding_dim)
            # Decoder: readout features → patch_size * vocab_size
            readout_feat_dim = (
                config.linear_modes + config.embedding_dim
                if config.enable_linear
                else config.embedding_dim
            )
            if self._patch_causal_mode == 'autoregressive':
                # MEGABYTE-style autoregressive byte decoder
                ssm_out_dim = config.linear_modes + config.embedding_dim if config.enable_linear else config.embedding_dim
                decoder_input_dim = ssm_out_dim + config.embedding_dim  # SSM features + prev byte embedding
                self._patch_byte_embed = nn.Embedding(vocab_size, config.embedding_dim)
                self._patch_byte_decoder = nn.GRU(decoder_input_dim, config.embedding_dim, batch_first=True)
                self._patch_byte_output = nn.Linear(config.embedding_dim, vocab_size)
            elif self._patch_causal_mode == 'mlp_factored':
                # Factored MLP: each byte position gets its own output head
                ssm_out_dim = config.linear_modes + config.embedding_dim if config.enable_linear else config.embedding_dim
                self._patch_byte_heads = nn.ModuleList([
                    nn.Linear(ssm_out_dim, vocab_size)
                    for _ in range(self._patch_size)
                ])
            else:
                # Original (illegal) flat decoder
                self._patch_decoder = nn.Linear(readout_feat_dim, self._patch_size * vocab_size)

        # --- Trust-routing mode ---
        self._trust_routing = getattr(config, 'trust_routing', False)
        if self._trust_routing:
            trust_input_dim = config.linear_modes + config.embedding_dim if config.enable_linear else config.embedding_dim
            self._trust_gate = nn.Sequential(
                nn.Linear(trust_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self._fallback_readout = nn.Linear(trust_input_dim, vocab_size)

            self._ngram_table = None  # injected via set_ngram_table()

        # --- Adaptive substrate: minimal recurrence, everything learned ---
        self._use_adaptive_substrate = getattr(config, "adaptive_substrate", False)
        if self._use_adaptive_substrate:
            ed = int(config.embedding_dim)
            n_modes = int(config.linear_modes)
            # 5 projections: decay, omega, write, project_real, project_imag
            self._adaptive_decay_proj = nn.Linear(ed, n_modes)
            self._adaptive_omega_proj = nn.Linear(ed, n_modes)
            self._adaptive_write_proj = nn.Linear(ed, n_modes)
            self._adaptive_proj_real = nn.Linear(ed, n_modes, bias=False)
            self._adaptive_proj_imag = nn.Linear(ed, n_modes, bias=False)
            # Output projection: 2*n_modes (real+imag) + embed_dim → readout_in_dim
            self._adaptive_out_proj = nn.Linear(2 * n_modes + ed, config.linear_modes + ed)
            with torch.no_grad():
                # Init decay near 0.95 (moderate memory)
                self._adaptive_decay_proj.weight.zero_()
                self._adaptive_decay_proj.bias.fill_(3.0)  # sigmoid(3) ≈ 0.95
                # Omega init: zero (pure EMA) or HRR Fourier basis
                self._adaptive_omega_proj.weight.zero_()
                if getattr(config, "hrr_omega_init", False):
                    # HRR binding: uniform Fourier basis [0, 2π).
                    # Each mode gets a unique rotation frequency.
                    # The state IS a discrete Fourier transform of the input.
                    # Readout demodulates by combining real/imag at each frequency.
                    import math
                    self._adaptive_omega_proj.bias.copy_(
                        torch.linspace(0, 2 * math.pi * (1 - 1/n_modes), n_modes)
                    )
                    # Frozen omega: the dynamics provide order FOR FREE.
                    # No optimization needed — the Fourier basis IS the physics.
                    if getattr(config, "freeze_omega", False):
                        self._adaptive_omega_proj.weight.requires_grad_(False)
                        self._adaptive_omega_proj.bias.requires_grad_(False)
                else:
                    self._adaptive_omega_proj.bias.zero_()
                # Init write near 0.1 (conservative writes)
                self._adaptive_write_proj.weight.zero_()
                self._adaptive_write_proj.bias.fill_(-2.0)  # sigmoid(-2) ≈ 0.12

        # Online causal memory (only useful with the linear path)
        self._use_online_memory = (
            getattr(config, "memory_kind", "none") != "none"
            and config.enable_linear
        )
        if self._use_online_memory:
            from decepticons.online_memory import OnlineCausalMemory, OnlineMemoryConfig
            mem_order = {"ngram": 3, "exact_context": 4, "statistical_backoff": 3}.get(config.memory_kind, 3)
            self._online_memory = OnlineCausalMemory(OnlineMemoryConfig(
                max_order=mem_order,
                bucket_count=8192,
                vocabulary_size=vocab_size,
            ))
            # Memory feature projection: 7 features -> readout input dim (additive residual)
            self._memory_proj = nn.Linear(7, linear_readout_in_dim)
            # Gate: scalar learned mix weight, initialised to 0 so memory has no effect at init
            self._memory_gate = nn.Parameter(torch.tensor(0.0))

        # Wire shared embedding weight into TiedEmbedReadout instances
        self._wire_tied_embed_readouts()

        self._reset_trainable_parameters()

    def _wire_tied_embed_readouts(self) -> None:
        """Wire the shared embedding weight into all TiedEmbedReadout modules."""
        embed = self.shared_embedding or self.linear_embedding
        if embed is None:
            return
        weight = embed.weight
        if isinstance(self.linear_readout, TiedEmbedReadout):
            self.linear_readout.set_embedding_weight(weight)
        if hasattr(self, '_band_readouts'):
            for readout in self._band_readouts:
                if isinstance(readout, TiedEmbedReadout):
                    readout.set_embedding_weight(weight)

    def _reset_trainable_parameters(self) -> None:
        seed = int(self.config.init_seed)
        if self.shared_embedding is not None:
            weight = _embedding_uniform(tuple(self.shared_embedding.weight.shape), _rng_for(seed, "shared_embedding.weight"))
            _copy_embedding_(self.shared_embedding, weight)
        if self.linear_embedding is not None:
            weight = _embedding_uniform(tuple(self.linear_embedding.weight.shape), _rng_for(seed, "linear_embedding.weight"))
            _copy_embedding_(self.linear_embedding, weight)
        if self.local_embedding is not None:
            weight = _embedding_uniform(tuple(self.local_embedding.weight.shape), _rng_for(seed, "local_embedding.weight"))
            _copy_embedding_(self.local_embedding, weight)
        if self.linear_readout is not None:
            self.linear_readout.reset_parameters_with_seed(seed, "linear_readout")
        if self.local_readout is not None:
            self.local_readout.reset_parameters_with_seed(seed, "local_readout")
        if self.gate_proj is not None:
            gate_weight = _xavier_uniform(tuple(self.gate_proj.weight.shape), _rng_for(seed, "gate_proj.weight"))
            _copy_linear_(self.gate_proj, gate_weight)
        if self.bank_gate_logits is not None:
            with torch.no_grad():
                self.bank_gate_logits.zero_()

    def param_groups(self, base_lr: float) -> list[dict]:
        """Return parameter groups with per-hemisphere learning rates."""
        if getattr(self, "_num_hemispheres", 1) != 2 or not (
            getattr(self, "_use_state_augment", False)
            or getattr(self, "_use_gated_retention_substrate", False)
            or getattr(self, "_use_gated_delta_substrate", False)
        ):
            return [{"params": list(self.parameters()), "lr": base_lr}]

        fast_params = []
        other_params = []

        fast_mult = getattr(self.config, 'fast_lr_mult', 4.0)

        for name, param in self.named_parameters():
            if (
                "_ssm_A" in name
                or "_ssm_B_proj" in name
                or "_ssm_C_proj" in name
                or "_retention_q_proj" in name
                or "_retention_k_proj" in name
                or "_retention_v_proj" in name
                or "_retention_logit_decay" in name
                or "_gated_retention_retain_gate" in name
                or "_gated_retention_write_gate" in name
                or "_gated_retention_erase_gate" in name
                or "_gated_delta_retain_gate" in name
                or "_gated_delta_write_gate" in name
                or "_gated_delta_erase_gate" in name
            ):
                # Split SSM params conceptually - but they're single tensors
                # Use the fast LR for all SSM params when hemispheres are active
                # (the decay initialization already handles the fast/slow split)
                fast_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if fast_params:
            groups.append({"params": fast_params, "lr": base_lr * fast_mult})
        if other_params:
            groups.append({"params": other_params, "lr": base_lr})
        return groups

    @staticmethod
    def _logit_features(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        max_logit = torch.max(logits, dim=-1).values
        centered = logits - torch.mean(logits, dim=-1, keepdim=True)
        variance = torch.mean(centered * centered, dim=-1)
        return entropy, max_logit, variance

    def _embed_linear(self, chars: torch.Tensor) -> torch.Tensor:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.linear_embedding is None:
            raise RuntimeError("causal-bank linear path has no embedding table.")
        return self.linear_embedding(chars)

    def _embed_local(self, chars: torch.Tensor) -> torch.Tensor:
        if self.shared_embedding is not None:
            return self.shared_embedding(chars)
        if self.local_embedding is None:
            raise RuntimeError("causal-bank local path has no embedding table.")
        return self.local_embedding(chars)

    def _linear_states_fft(self, drive: torch.Tensor, timesteps: int) -> torch.Tensor:
        drive_mb = drive.transpose(1, 2)
        n_fft = 1 << int(math.ceil(math.log2(max(2 * timesteps - 1, 1))))
        time = torch.arange(timesteps, dtype=drive.dtype, device=drive.device)
        decays = self.linear_decays.to(device=drive.device, dtype=drive.dtype)
        kernel = torch.pow(decays[:, None], time[None, :])
        drive_f = torch.fft.rfft(drive_mb, n=n_fft, dim=-1)
        kernel_f = torch.fft.rfft(kernel.unsqueeze(0), n=n_fft, dim=-1)
        states_mb = torch.fft.irfft(drive_f * kernel_f, n=n_fft, dim=-1)[..., :timesteps]
        return states_mb.transpose(1, 2)

    def _apply_mode_gate(self, states: torch.Tensor, mode_gate: torch.Tensor | None) -> torch.Tensor:
        if mode_gate is None:
            return states
        if mode_gate.ndim == 1:
            return states * mode_gate.view(1, 1, -1)
        if mode_gate.ndim == 2:
            return states * mode_gate[:, None, :]
        raise ValueError(f"causal-bank mode_gate must be rank-1 or rank-2, got shape {tuple(mode_gate.shape)}")

    def _static_bank_mode_gate(self) -> torch.Tensor | None:
        if self.bank_gate_logits is None or self.osc_mode_count <= 0:
            return None
        values = 1.0 + self.config.bank_gate_span * torch.tanh(self.bank_gate_logits)
        pieces = []
        if self.non_osc_modes > 0:
            pieces.append(values[0:1].expand(self.non_osc_modes))
        if self.osc_mode_count > 0:
            pieces.append(values[1:2].expand(self.osc_mode_count))
        return torch.cat(pieces, dim=0) if pieces else None

    def _chunked_recurrence_scan(self, gates: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """Chunked parallel scan for recurrences of the form h_t = a_t * h_{t-1} + b_t."""
        batch, seq_len = drive.shape[:2]
        state_shape = drive.shape[2:]
        device = drive.device
        dtype = drive.dtype

        K = min(32, seq_len)
        n_chunks = (seq_len + K - 1) // K
        states = torch.zeros_like(drive)
        h = torch.zeros((batch, *state_shape), device=device, dtype=dtype)

        for c in range(n_chunks):
            start = c * K
            end = min(start + K, seq_len)
            a_chunk = gates[:, start:end, ...]
            b_chunk = drive[:, start:end, ...]

            log_a = torch.log(a_chunk.clamp(min=1e-6))
            log_cum_a = torch.cumsum(log_a, dim=1)
            cum_a = torch.exp(log_cum_a)

            inv_cum_a = 1.0 / cum_a.clamp(min=1e-8)
            weighted_b = b_chunk * inv_cum_a
            cum_wb = torch.cumsum(weighted_b, dim=1)

            chunk_states = cum_a * (h.unsqueeze(1) + cum_wb)
            states[:, start:end, ...] = chunk_states
            h = chunk_states[:, -1, ...]

        return states

    def _complex_rotation_scan(
        self,
        gates: torch.Tensor,
        drive: torch.Tensor,
        cos_omega: torch.Tensor,
        sin_omega: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked scan with input-dependent complex rotation.

        State is paired: dims [2j, 2j+1] form a complex number.
        Update: z_t = decay_t * exp(i*ω_t) * z_{t-1} + drive_t

        Uses native complex tensors: one complex multiply per step
        instead of 6 real ops (2 cos, 2 sin, 2 add). Still sequential
        per position (rotation is nonlinear), but each step is one
        CUDA kernel not six.

        gates: [batch, seq, heads, head_dim] — scalar decay per dim
        drive: [batch, seq, heads, head_dim] — additive input per dim
        cos_omega: [batch, seq, heads, half_dim] — cos of rotation angle per pair
        sin_omega: [batch, seq, heads, half_dim] — sin of rotation angle per pair
        """
        batch, seq_len, n_heads, head_dim = drive.shape
        half = head_dim // 2
        device = drive.device

        # Convert to complex: pair dims (2j, 2j+1) → complex z[j]
        drive_even = drive[..., :half]
        drive_odd = drive[..., half:]
        z_drive = torch.complex(drive_even.float(), drive_odd.float())  # [B, T, H, half]

        # Decay is real, applied to both components equally
        a_even = gates[..., :half].float()  # [B, T, H, half]

        # Rotation as complex multiplier: exp(iω) = cos(ω) + i*sin(ω)
        rot = torch.complex(cos_omega.float(), sin_omega.float())  # [B, T, H, half]

        # Combined: a * exp(iω) per position
        a_rot = a_even * rot  # [B, T, H, half]

        # Parallel prefix scan for complex affine recurrence z_t = a_t * z_{t-1} + b_t
        # Uses torch.roll + torch.where for dynamo-compatible fixed shapes.
        ca = a_rot  # [B, T, H, half] complex
        cb = z_drive  # [B, T, H, half] complex

        pos_idx = torch.arange(seq_len, device=device)
        n_levels = max(1, seq_len.bit_length() if isinstance(seq_len, int) else int(seq_len).bit_length())

        for level in range(n_levels):
            d = 1 << level
            if d >= seq_len:
                break
            mask = (pos_idx >= d).view(1, -1, 1, 1)

            ca_prev = torch.roll(ca, d, dims=1)
            cb_prev = torch.roll(cb, d, dims=1)

            new_ca = ca * ca_prev
            new_cb = ca * cb_prev + cb

            ca = torch.where(mask, new_ca, ca)
            cb = torch.where(mask, new_cb, cb)

        # h_0 = 0, so state = cumulative_b
        states = torch.cat([cb.real, cb.imag], dim=-1).to(drive.dtype)
        return states

    def _lasso_rotation_scan(
        self,
        drive: torch.Tensor,
        m_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Noncommutative 2x2 matrix scan — the lasso.

        Parallel prefix scan (Hillis-Steele) on the affine recurrence
        h_t = M_t @ h_{t-1} + b_t. The combine operator is:
            combine((M2, b2), (M1, b1)) = (M2 @ M1, M2 @ b1 + b2)
        which is associative (matrix multiply is associative).

        Runs in O(log n) parallel steps instead of O(n) sequential.
        Each step does batched element-wise ops on the full sequence.

        drive: [batch, seq, heads, head_dim] — additive input
        m_matrix: [batch, seq, heads, half_dim, 4] — [m00, m01, m10, m11]
        """
        batch, seq_len, n_heads, head_dim = drive.shape
        half = head_dim // 2
        device = drive.device
        dtype = drive.dtype

        # Unpack transition matrix elements: each [B, T, H, half]
        a00 = m_matrix[..., 0]
        a01 = m_matrix[..., 1]
        a10 = m_matrix[..., 2]
        a11 = m_matrix[..., 3]

        # Drive split into even/odd: each [B, T, H, half]
        b0 = drive[..., :half]
        b1 = drive[..., half:]

        # Hillis-Steele inclusive prefix scan with affine combine operator.
        # Uses torch.roll + torch.where for dynamo-compatible fixed shapes.
        # 9 iterations for seq_len=512, 10 for 1024. No dynamic slicing.

        pos_idx = torch.arange(seq_len, device=device)  # [T]
        n_levels = max(1, seq_len.bit_length() if isinstance(seq_len, int) else int(seq_len).bit_length())

        for level in range(n_levels):
            d = 1 << level
            if d >= seq_len:
                break
            mask = (pos_idx >= d).view(1, -1, 1, 1)  # [1, T, 1, 1] broadcast

            # Roll left by d: position i reads from position i-d
            p00 = torch.roll(a00, d, dims=1)
            p01 = torch.roll(a01, d, dims=1)
            p10 = torch.roll(a10, d, dims=1)
            p11 = torch.roll(a11, d, dims=1)
            pb0 = torch.roll(b0, d, dims=1)
            pb1 = torch.roll(b1, d, dims=1)

            # A_new = A_curr @ A_prev (2x2 matmul via element-wise ops)
            n00 = a00 * p00 + a01 * p10
            n01 = a00 * p01 + a01 * p11
            n10 = a10 * p00 + a11 * p10
            n11 = a10 * p01 + a11 * p11

            # b_new = A_curr @ b_prev + b_curr
            nb0 = a00 * pb0 + a01 * pb1 + b0
            nb1 = a10 * pb0 + a11 * pb1 + b1

            # Only update positions >= d; first d positions keep original values
            a00 = torch.where(mask, n00, a00)
            a01 = torch.where(mask, n01, a01)
            a10 = torch.where(mask, n10, a10)
            a11 = torch.where(mask, n11, a11)
            b0 = torch.where(mask, nb0, b0)
            b1 = torch.where(mask, nb1, b1)

        # h_0 = 0, so state_t = cumulative_b_t
        states = torch.cat([b0, b1], dim=-1)
        return states

    @staticmethod
    def _hamilton_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication. a, b: [..., 4] as (w, x, y, z)."""
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    def _quaternion_scan(
        self,
        gates: torch.Tensor,
        drive: torch.Tensor,
        q_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel prefix scan with quaternion rotation.

        Recurrence: h_t = q_rot_t ⊗ (gate_t * h_{t-1}) + drive_t

        Prefix operation (associative):
            (g1, d1, q1) ∘ (g2, d2, q2) = (g1*g2, q2 ⊗ (g2*d1) + d2, q2 ⊗ q1)

        gates: [B, T, H, 4] — scalar decay broadcast to all 4 components
        drive: [B, T, H, 4] — additive input
        q_rot: [B, T, H, 4] — unit quaternion rotation per position
        """
        batch, seq_len, n_heads, _ = drive.shape
        device = drive.device

        # Scan state: (g, d, q) where g is scalar, d is quaternion, q is quaternion
        # g: cumulative decay, d: cumulative drive, q: cumulative rotation
        g = gates[..., :1]       # [B, T, H, 1] — scalar decay (take first component, all equal)
        d = drive                # [B, T, H, 4]
        q = q_rot                # [B, T, H, 4]

        pos_idx = torch.arange(seq_len, device=device)
        n_levels = max(1, int(seq_len).bit_length())

        for level in range(n_levels):
            stride = 1 << level
            if stride >= seq_len:
                break
            mask = (pos_idx >= stride).view(1, -1, 1, 1)  # [1, T, 1, 1]

            g_prev = torch.roll(g, stride, dims=1)
            d_prev = torch.roll(d, stride, dims=1)
            q_prev = torch.roll(q, stride, dims=1)

            # Combine: (g, d, q) ∘ (g_prev, d_prev, q_prev)
            new_g = g * g_prev
            new_d = self._hamilton_product(q, g * d_prev) + d
            new_q = self._hamilton_product(q, q_prev)

            g = torch.where(mask, new_g, g)
            d = torch.where(mask, new_d, d)
            q = torch.where(mask, new_q, q)

        # h_0 = 0, so state = cumulative d
        return d

    def _adaptive_substrate_states(self, x_embed: torch.Tensor) -> torch.Tensor:
        """Minimal recurrence: 5 complex projections, parallel scan.

        z_k(t) = a_k(x_t) * z_k(t-1) + b_k(x_t)
        a_k = decay_k(x) * exp(i * omega_k(x))
        b_k = write_k(x) * (proj_real(x) + i * proj_imag(x))

        No heads, no bands, no blocks, no local path.
        Returns features for the readout: [real(z), imag(z), x_embed].
        """
        batch, seq_len, _ = x_embed.shape
        device = x_embed.device
        dtype = x_embed.dtype

        # 5 projections
        decay = torch.sigmoid(self._adaptive_decay_proj(x_embed))     # [B, T, modes]
        omega = self._adaptive_omega_proj(x_embed)                     # [B, T, modes] unbounded
        write = torch.sigmoid(self._adaptive_write_proj(x_embed))     # [B, T, modes]
        proj_r = self._adaptive_proj_real(x_embed)                     # [B, T, modes]
        proj_i = self._adaptive_proj_imag(x_embed)                     # [B, T, modes]

        # Complex transition: a = decay * exp(i*omega)
        a_real = decay * torch.cos(omega)
        a_imag = decay * torch.sin(omega)
        a = torch.complex(a_real.float(), a_imag.float())  # [B, T, modes]

        # Complex drive: b = write * (proj_r + i*proj_i)
        b_real = write * proj_r
        b_imag = write * proj_i
        b = torch.complex(b_real.float(), b_imag.float())  # [B, T, modes]

        # Parallel prefix scan: z_t = a_t * z_{t-1} + b_t
        ca = a
        cb = b
        pos_idx = torch.arange(seq_len, device=device)
        n_levels = max(1, int(seq_len).bit_length())
        for level in range(n_levels):
            d = 1 << level
            if d >= seq_len:
                break
            mask = (pos_idx >= d).view(1, -1, 1)
            ca_prev = torch.roll(ca, d, dims=1)
            cb_prev = torch.roll(cb, d, dims=1)
            new_ca = ca * ca_prev
            new_cb = ca * cb_prev + cb
            ca = torch.where(mask, new_ca, ca)
            cb = torch.where(mask, new_cb, cb)

        # State = cumulative b (h_0 = 0)
        z = cb  # [B, T, modes] complex

        # Output: concat real, imag, embedding → project to readout dim
        features = torch.cat([z.real.to(dtype), z.imag.to(dtype), x_embed], dim=-1)
        out = self._adaptive_out_proj(features)

        # Store for forensics
        self._last_adaptive_decay = decay.detach()
        self._last_adaptive_omega = omega.detach()
        self._last_adaptive_write = write.detach()

        return out

    def _block_matrix_scan(
        self,
        drive: torch.Tensor,
        m_matrix: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Parallel prefix scan with d×d block matrix transitions.

        Recurrence: h_t = M_t @ h_{t-1} + drive_t
        Prefix: (M2, d2) ∘ (M1, d1) = (M2 @ M1, M2 @ d1 + d2)

        Generalizes the lasso (d=2), quaternion (d=4), and quintic (d=5).
        Uses torch.roll + torch.where for dynamo compatibility.

        drive: [B, T, H, block_size]
        m_matrix: [B, T, H, block_size, block_size]
        """
        batch, seq_len, n_heads, bs = drive.shape
        device = drive.device

        # Scan elements: M (cumulative matrix), d (cumulative drive)
        M = m_matrix  # [B, T, H, bs, bs]
        d = drive     # [B, T, H, bs]

        pos_idx = torch.arange(seq_len, device=device)
        n_levels = max(1, int(seq_len).bit_length())

        for level in range(n_levels):
            stride = 1 << level
            if stride >= seq_len:
                break
            mask_mat = (pos_idx >= stride).view(1, -1, 1, 1, 1)  # [1, T, 1, 1, 1]
            mask_vec = (pos_idx >= stride).view(1, -1, 1, 1)     # [1, T, 1, 1]

            M_prev = torch.roll(M, stride, dims=1)
            d_prev = torch.roll(d, stride, dims=1)

            # M_new = M_curr @ M_prev
            new_M = torch.einsum("...ij,...jk->...ik", M, M_prev)
            # d_new = M_curr @ d_prev + d_curr
            new_d = torch.einsum("...ij,...j->...i", M, d_prev) + d

            M = torch.where(mask_mat, new_M, M)
            d = torch.where(mask_vec, new_d, d)

        # h_0 = 0, so state = cumulative d
        return d

    def _linear_states_recurrent(self, drive: torch.Tensor, x_embed: torch.Tensor) -> torch.Tensor:
        """Learned recurrence: state[t] = gate * state[t-1] + (1-gate) * drive[t]

        gate = sigmoid(x_embed[t] @ W_gate + b_gate)  per-mode, input-dependent

        drive: [batch, seq, modes] (projected input)
        x_embed: [batch, seq, embed_dim] (raw embedding, for gate computation)
        Returns: [batch, seq, modes]
        """
        batch, seq_len, modes = drive.shape
        dtype = drive.dtype

        # Compute gates for all positions at once: [batch, seq, modes]
        gates = torch.sigmoid(
            torch.matmul(x_embed, self._gate_weight.to(dtype=dtype)) + self._gate_bias.to(dtype=dtype)
        )

        return self._chunked_recurrence_scan(gates, (1.0 - gates) * drive)

    def _state_augment(self, x_embed: torch.Tensor) -> torch.Tensor:
        if self._state_impl == "scan":
            return self._selective_scan(x_embed)
        return self._retention_states(x_embed)

    def _selective_scan(self, x_embed: torch.Tensor) -> torch.Tensor:
        """Head-factored selective scan over the content-dependent augment state."""
        batch, seq_len, _ = x_embed.shape
        device = x_embed.device
        dtype = x_embed.dtype

        B = self._ssm_B_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
        C = self._ssm_C_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)

        K = min(32, seq_len)
        n_chunks = (seq_len + K - 1) // K

        y = torch.zeros(
            batch,
            seq_len,
            self._state_num_heads,
            self._state_head_dim,
            device=device,
            dtype=dtype,
        )
        h = torch.zeros(batch, self._state_num_heads, self._state_head_dim, device=device, dtype=dtype)

        for c_idx in range(n_chunks):
            start = c_idx * K
            end = min(start + K, seq_len)

            B_chunk = B[:, start:end, :, :]
            C_chunk = C[:, start:end, :, :]
            drive = B_chunk

            log_a = self._ssm_A.to(dtype=dtype).unsqueeze(0).unsqueeze(0).expand(batch, end - start, -1, -1)
            log_a = log_a.clamp(max=-1e-6)
            log_cum_a = torch.cumsum(log_a, dim=1)
            cum_a = torch.exp(log_cum_a)

            inv_cum_a = 1.0 / cum_a.clamp(min=1e-8)
            weighted_drive = drive * inv_cum_a
            cum_wd = torch.cumsum(weighted_drive, dim=1)

            chunk_states = cum_a * (h.unsqueeze(1) + cum_wd)
            chunk_y = chunk_states * C_chunk
            y[:, start:end, :, :] = chunk_y
            h = chunk_states[:, -1, :, :]

        out = self._ssm_out_proj(y.reshape(batch, seq_len, self._state_width))
        linear_in_proj = self.linear_in_proj.to(device=device, dtype=dtype)
        return out + self._ssm_D * torch.matmul(x_embed, linear_in_proj)

    def _gated_delta_states(self, x_embed: torch.Tensor) -> torch.Tensor:
        """Primary learned scan substrate with token-conditioned retain/write/erase."""
        batch, seq_len, _ = x_embed.shape
        device = x_embed.device
        dtype = x_embed.dtype

        value = self._ssm_B_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
        read = self._ssm_C_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)

        # Gate input: x_embed or [x_embed, log(1+t)] if position_signal enabled
        if getattr(self, "_use_position_signal", False):
            pos = torch.log1p(torch.arange(seq_len, device=device, dtype=dtype))  # [T]
            pos = pos.view(1, seq_len, 1).expand(batch, -1, -1)  # [B, T, 1]
            gate_input = torch.cat([x_embed, pos], dim=-1)  # [B, T, ed+1]
        else:
            gate_input = x_embed

        base_decay = torch.exp(self._ssm_A.to(dtype=dtype).clamp(max=-1e-6)).unsqueeze(0).unsqueeze(0)
        retain = torch.sigmoid(self._gated_delta_retain_gate(gate_input)).to(dtype=dtype).reshape(
            batch,
            seq_len,
            self._state_num_heads,
            self._state_head_dim,
        )
        write = torch.sigmoid(self._gated_delta_write_gate(gate_input)).to(dtype=dtype).reshape(
            batch,
            seq_len,
            self._state_num_heads,
            self._state_head_dim,
        )
        erase = torch.sigmoid(self._gated_delta_erase_gate(gate_input)).to(dtype=dtype).reshape(
            batch,
            seq_len,
            self._state_num_heads,
            self._state_head_dim,
        )

        keep = (base_decay * retain * (1.0 - erase) * (1.0 - write)).clamp(min=1e-6, max=1.0 - 1e-6)
        drive = write * value

        if getattr(self, "_use_so5_rotation", False):
            bs = self._so5_block_size
            lie_dim = self._so5_lie_dim
            # Project to Lie algebra parameters
            lie_params = self._so5_proj(gate_input).reshape(
                batch, seq_len, self._state_num_heads, lie_dim
            ).to(dtype=dtype)
            # Build skew-symmetric matrices and exponentiate to SO(n)
            idx_i, idx_j = self._so5_triu_indices
            A = torch.zeros(batch, seq_len, self._state_num_heads, bs, bs,
                            device=device, dtype=dtype)
            A[..., idx_i, idx_j] = lie_params
            A = A - A.transpose(-1, -2)  # skew-symmetric: A = -A^T
            R = torch.matrix_exp(A)      # guaranteed SO(n), det=1, R^T R = I
            # Scale by decay, apply rotation
            keep_scalar = keep.reshape(batch, seq_len, self._state_num_heads, bs, 1)
            M = R * keep_scalar  # contractive rotation
            drive_r = drive.reshape(batch, seq_len, self._state_num_heads, bs)
            states_r = self._block_matrix_scan(drive_r, M, bs)
            states = states_r.reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
            self._last_so5_rotation = R.mean(dim=2).detach()
        elif getattr(self, "_use_quintic_rotation", False):
            bs = self._quintic_block_size
            m_raw = self._quintic_proj(gate_input).reshape(
                batch, seq_len, self._state_num_heads, bs, bs
            ).to(dtype=dtype)
            m_matrix = torch.sigmoid(m_raw)
            keep_scalar = keep.reshape(batch, seq_len, self._state_num_heads, bs, 1)
            m_matrix = m_matrix * keep_scalar
            drive_r = drive.reshape(batch, seq_len, self._state_num_heads, bs)
            states_r = self._block_matrix_scan(drive_r, m_matrix, bs)
            states = states_r.reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
            self._last_quintic_matrix = m_matrix.mean(dim=2).detach()
        elif getattr(self, "_use_quaternion_rotation", False):
            # Quaternion Hamilton product rotation per head (head_dim=4)
            q_raw = self._quat_proj(gate_input).reshape(
                batch, seq_len, self._state_num_heads, 4
            ).to(dtype=dtype)
            # Normalize to unit quaternion
            q_rot = q_raw / (q_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            states = self._quaternion_scan(keep, drive, q_rot)
            self._last_quat_rotation = q_rot.mean(dim=2).detach()
        elif getattr(self, "_use_lasso_rotation", False):
            # Noncommutative 2x2 matrix transition per mode pair
            half = self._state_head_dim // 2
            m_raw = self._lasso_proj(gate_input).reshape(
                batch, seq_len, self._state_num_heads, half, 4
            ).to(dtype=dtype)
            # Sigmoid bounds elements to (0,1) for stability
            m_matrix = torch.sigmoid(m_raw)
            # Scale by keep (decay) — the matrix shrinks the state overall
            keep_even = keep[..., :half].unsqueeze(-1)   # [B, T, H, half, 1]
            m_matrix = m_matrix * keep_even
            states = self._lasso_rotation_scan(drive, m_matrix)
            self._last_lasso_matrix = m_matrix.mean(dim=2).detach()
        elif getattr(self, "_use_complex_rotation", False):
            # Input-dependent rotation angle per mode pair per head (SO(2), commutative)
            omega_raw = self._gated_delta_rotation_proj(x_embed).reshape(
                batch, seq_len, self._state_num_heads, self._rotation_half_dim
            ).to(dtype=dtype)
            omega = torch.tanh(omega_raw) * 3.14159265
            cos_omega = torch.cos(omega)
            sin_omega = torch.sin(omega)
            states = self._complex_rotation_scan(keep, drive, cos_omega, sin_omega)
            self._last_gated_delta_omega = omega.mean(dim=2).detach()
        else:
            states = self._chunked_recurrence_scan(keep, drive)

        outputs = states * read

        # Store gate values for forensics (mean across heads for compact storage)
        self._last_gated_delta_write = write.mean(dim=2).detach()   # [batch, seq, head_dim]
        self._last_gated_delta_retain = retain.mean(dim=2).detach()
        self._last_gated_delta_erase = erase.mean(dim=2).detach()

        out = self._ssm_out_proj(outputs.reshape(batch, seq_len, self._state_width))
        linear_in_proj = self.linear_in_proj.to(device=device, dtype=dtype)
        return out + self._ssm_D * torch.matmul(x_embed, linear_in_proj)

    def _retention_states(self, x_embed: torch.Tensor) -> torch.Tensor:
        """RetNet-style multi-head matrix memory with exponential decay."""
        batch, seq_len, _ = x_embed.shape
        device = x_embed.device
        dtype = x_embed.dtype

        q = self._retention_q_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
        k = self._retention_k_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
        v = self._retention_v_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)

        flat_dim = self._state_head_dim * self._state_head_dim
        decay = torch.sigmoid(self._retention_logit_decay).to(device=device, dtype=dtype).view(
            1, 1, self._state_num_heads, 1
        )
        scale = 1.0 / math.sqrt(float(self._state_head_dim))
        drive = torch.einsum("bthd,bthe->bthde", k, v).reshape(batch, seq_len, self._state_num_heads, flat_dim)
        gates = decay.expand(batch, seq_len, self._state_num_heads, flat_dim)
        memory = self._chunked_recurrence_scan(gates, drive).reshape(
            batch,
            seq_len,
            self._state_num_heads,
            self._state_head_dim,
            self._state_head_dim,
        )
        outputs = torch.einsum("bthd,bthde->bthe", q * scale, memory)

        out = self._retention_out_proj(outputs.reshape(batch, seq_len, self._state_width))
        linear_in_proj = self.linear_in_proj.to(device=device, dtype=dtype)
        return out + self._retention_D * torch.matmul(x_embed, linear_in_proj)

    def _gated_retention_states(self, x_embed: torch.Tensor) -> torch.Tensor:
        """Primary learned substrate: gated multi-head matrix memory."""
        batch, seq_len, _ = x_embed.shape
        device = x_embed.device
        dtype = x_embed.dtype

        q = self._retention_q_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
        k = self._retention_k_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)
        v = self._retention_v_proj(x_embed).reshape(batch, seq_len, self._state_num_heads, self._state_head_dim)

        flat_dim = self._state_head_dim * self._state_head_dim
        base_decay = torch.sigmoid(self._retention_logit_decay).to(device=device, dtype=dtype).view(
            1, 1, self._state_num_heads, 1
        )
        retain = torch.sigmoid(self._gated_retention_retain_gate(x_embed)).to(dtype=dtype).unsqueeze(-1)
        write = torch.sigmoid(self._gated_retention_write_gate(x_embed)).to(dtype=dtype).unsqueeze(-1)
        erase = torch.sigmoid(self._gated_retention_erase_gate(x_embed)).to(dtype=dtype).unsqueeze(-1)

        scale = 1.0 / math.sqrt(float(self._state_head_dim))
        keep = base_decay * retain * (1.0 - erase)
        drive = write * torch.einsum("bthd,bthe->bthde", k, v).reshape(
            batch,
            seq_len,
            self._state_num_heads,
            flat_dim,
        )
        memory = self._chunked_recurrence_scan(keep.expand(batch, seq_len, self._state_num_heads, flat_dim), drive)
        memory = memory.reshape(
            batch,
            seq_len,
            self._state_num_heads,
            self._state_head_dim,
            self._state_head_dim,
        )
        outputs = torch.einsum("bthd,bthde->bthe", q * scale, memory)

        out = self._retention_out_proj(outputs.reshape(batch, seq_len, self._state_width))
        linear_in_proj = self.linear_in_proj.to(device=device, dtype=dtype)
        drive = torch.matmul(x_embed, linear_in_proj)
        return out + self._retention_D * drive

    def _apply_substrate(
        self,
        drive: torch.Tensor,
        x: torch.Tensor,
        timesteps: int,
        kernels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the substrate (recurrence, kernel, or FFT) to *drive*.

        Returns states [batch, seq, modes].
        """
        if self._use_gated_retention_substrate:
            return self._gated_retention_states(x)
        if self._use_gated_delta_substrate:
            return self._gated_delta_states(x)
        if self._learned_recurrence:
            return self._linear_states_recurrent(drive, x)
        if self.config.linear_impl == "scan":
            # O(n) chunked parallel scan — same as kernel but without materializing the full matrix
            decays = self.linear_decays.to(device=drive.device, dtype=drive.dtype)
            gates = decays.unsqueeze(0).unsqueeze(0).expand(drive.shape[0], timesteps, -1)
            return self._chunked_recurrence_scan(gates, drive)
        if self.config.linear_impl == "kernel":
            if kernels is None:
                raise RuntimeError("causal-bank kernel path called without kernels.")
            drive_mb = drive.permute(2, 0, 1)
            states_mb = torch.matmul(drive_mb, kernels.transpose(1, 2))
            result = states_mb.permute(1, 2, 0)
            if torch.isfinite(result).all():
                return result
            # Kernel path produced NaN/Inf (numerical instability on CPU
            # with large hl_max). Fall back to scan path.
            import warnings
            warnings.warn(
                "Kernel path produced non-finite values, falling back to scan path",
                RuntimeWarning, stacklevel=2,
            )
            decays = self.linear_decays.to(device=drive.device, dtype=drive.dtype)
            gates = decays.unsqueeze(0).unsqueeze(0).expand(drive.shape[0], timesteps, -1)
            return self._chunked_recurrence_scan(gates, drive)
        return self._linear_states_fft(drive, timesteps)

    def _linear_states(self, chars: torch.Tensor, mode_gate: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _, timesteps = chars.shape
        if timesteps > self.config.max_seq_len:
            raise ValueError(
                f"causal-bank max_seq_len={self.config.max_seq_len} is smaller than input timesteps={timesteps}"
            )
        if self.linear_readout is None:
            raise RuntimeError("causal-bank linear path is disabled.")
        x = self._embed_linear(chars)

        linear_in_proj = self.linear_in_proj.to(device=x.device, dtype=x.dtype)
        drive = torch.matmul(x, linear_in_proj)

        # Prepare kernels once (used by kernel and learnable_decays paths)
        kernels = None
        if not self._learned_recurrence and not self._use_primary_state_substrate and self.config.linear_impl == "kernel":
            if self._recompute_kernel:
                # Recompute kernel from learnable decays
                time_idx = self._kernel_time_idx[:timesteps].to(device=x.device)
                delta = time_idx[:, None] - time_idx[None, :]
                mask = delta >= 0
                safe_delta = torch.where(mask, delta, torch.zeros_like(delta)).float()
                decays = self.linear_decays.to(device=x.device, dtype=x.dtype)
                kernel = decays[None, None, :] ** safe_delta[..., None]
                kernel = kernel * mask[..., None].float()
                # kernel shape: (T, T, modes) -> transpose to (modes, T, T)
                kernels = kernel.permute(2, 0, 1)
            elif self.linear_kernel is None:
                raise RuntimeError("causal-bank kernel path is missing its materialized kernel.")
            else:
                kernels = self.linear_kernel[:, :timesteps, :timesteps].to(device=x.device, dtype=x.dtype)

        static_mode_gate = self._static_bank_mode_gate()

        # Write-side: route drive signal to substrate banks before EMA
        if getattr(self, '_use_substrate_bank_router', False):
            drive = self._substrate_bank_router(drive, x)

        # First substrate application
        states = self._apply_substrate(drive, x, timesteps, kernels)

        # Write-side: overwrite stale modes after EMA
        if getattr(self, '_use_overwrite_gate', False):
            states = self._overwrite_gate(states, drive, x)

        # Stacked blocks: mix then re-apply substrate
        block_stride = getattr(self.config, 'block_stride', 1)
        if self.num_blocks > 1 and hasattr(self, '_block_layers'):
            for idx, block_layer in enumerate(self._block_layers):
                if block_stride > 1 and idx > 0:
                    # Subsample: take every stride-th position
                    stride = block_stride ** idx  # exponential: stride 1, 4, 16...
                    if stride < states.shape[1]:
                        strided = states[:, ::stride, :]
                        mixed_strided = strided + block_layer(strided)
                        strided_states = self._apply_substrate(mixed_strided, x[:, ::stride, :], strided.shape[1], kernels)
                        # Upsample back: repeat
                        states_upsampled = strided_states.repeat_interleave(stride, dim=1)[:, :timesteps, :]
                        states = states + states_upsampled  # residual add
                    else:
                        mixed = states + block_layer(states)
                        states = self._apply_substrate(mixed, x, timesteps, kernels)
                else:
                    mixed = states + block_layer(states)
                    states = self._apply_substrate(mixed, x, timesteps, kernels)
                states = self._apply_mode_gate(states, static_mode_gate)

        states = self._apply_mode_gate(states, static_mode_gate)
        states = self._apply_mode_gate(states, mode_gate)

        # Augment: recurrent state path adds content-dependent signal on top of the frozen bank
        if self._use_state_augment:
            states = states + self._state_augment(x)

        # Sticky registers: persistent memory for surprising tokens
        if self._sticky_registers > 0:
            states = states + self._sticky_register_forward(x)

        return states, x

    def _sticky_register_forward(self, x_embed: torch.Tensor) -> torch.Tensor:
        """Run sticky registers: slow-decay memory with surprise-gated writes.

        Uses the chunked parallel scan (same as gated_delta/selective_scan)
        for O(n) compute with vectorized chunks instead of a Python per-position loop.

        Recurrence: reg_t = keep_t * reg_{t-1} + write_t * value_t
          where keep_t = decay * (1 - write_strength_t)

        Returns: [batch, seq, linear_modes] contribution to add to states.
        """
        decay = self._sticky_decay.to(device=x_embed.device, dtype=x_embed.dtype)

        # Write strength: sigmoid(embed → nr), biased toward NOT writing
        write_strength = torch.sigmoid(self._sticky_write_gate(x_embed))  # [batch, seq, nr]
        write_value = self._sticky_write_proj(x_embed)  # [batch, seq, nr]

        # Map to chunked_recurrence_scan form: h_t = a_t * h_{t-1} + b_t
        gates = (decay * (1.0 - write_strength)).clamp(min=1e-6, max=1.0 - 1e-6)  # [batch, seq, nr]
        drive = write_strength * write_value  # [batch, seq, nr]

        reg_states = self._chunked_recurrence_scan(gates, drive)  # [batch, seq, nr]
        return self._sticky_read_proj(reg_states)  # [batch, seq, linear_modes]

    def _compute_online_memory_features(self, chars: torch.Tensor) -> torch.Tensor:
        """Process sequence through the online causal memory, returning projected features.

        Returns tensor of shape [batch, seq_len, readout_in_dim].
        """
        device = chars.device
        batch_size, seq_len = chars.shape
        mem_features_list = []
        for b in range(batch_size):
            self._online_memory.reset()
            batch_features = []
            for t in range(seq_len):
                f = self._online_memory.query_features()
                batch_features.append(torch.from_numpy(f))
                self._online_memory.update(int(chars[b, t].item()))
            mem_features_list.append(torch.stack(batch_features))
        mem_features = torch.stack(mem_features_list).to(device=device)  # [batch, seq, 7]
        return torch.sigmoid(self._memory_gate) * self._memory_proj(mem_features)

    def _linear_logits(self, chars: torch.Tensor, mode_gate: torch.Tensor | None = None) -> torch.Tensor:
        # Adaptive substrate: bypass all transforms, bands, local path.
        # The recurrence IS the model. The readout reads the result directly.
        if getattr(self, '_use_adaptive_substrate', False):
            x_embed = self._embed_linear(chars)
            features = self._adaptive_substrate_states(x_embed)
            return self.linear_readout(features)

        states, x = self._linear_states(chars, mode_gate=mode_gate)
        noise_sigma = getattr(self.config, 'training_noise', 0.0)
        if noise_sigma > 0 and self.training:
            states = states + torch.randn_like(states) * noise_sigma

        # Apply substrate transforms
        if getattr(self, '_use_magnitude_norm', False):
            states = self._magnitude_normalizer(states)
        if getattr(self, '_use_mode_selector', False):
            states = self._mode_selector(states, x)
        if getattr(self, '_use_temporal_attention', False):
            bank = TemporalAttention.build_bank(states, self._temporal_snapshot_interval)
            temporal_ctx = self._temporal_attention(states, bank, snapshot_interval=self._temporal_snapshot_interval)
            states = states + temporal_ctx

        # Hash memory: add retrieved past token context to embedding
        if getattr(self, '_use_hash_memory', False):
            hash_ctx = self._hash_memory(x)
            x = x + hash_ctx

        n_bands = getattr(self.config, 'readout_bands', 1)
        if n_bands > 1 and hasattr(self, '_band_readouts'):
            # Grouped prediction: split modes by timescale, separate readout per band
            modes = states.shape[-1]
            band_size = modes // n_bands
            band_logits = []
            _static = getattr(self, '_static_band_indices', set())
            for i, readout in enumerate(self._band_readouts):
                if i in _static:
                    # Static band: embedding-only lookup, no substrate
                    band_logits.append(readout(x))
                else:
                    band_states = states[..., i * band_size : (i + 1) * band_size]
                    band_features = torch.cat([band_states, x], dim=-1)
                    band_logits.append(readout(band_features))
            return sum(band_logits)

        features = torch.cat([states, x], dim=-1)
        if getattr(self, '_substrate_poly', 1) >= 2:
            proj = self._substrate_poly_proj(states)
            d = proj.shape[-1]
            i, j = torch.triu_indices(d, d, device=proj.device)
            quad = proj[..., i] * proj[..., j]
            poly_parts = [quad]
            if self._substrate_poly >= 3:
                poly_parts.append(proj ** 3)
            features = torch.cat([features] + poly_parts, dim=-1)
        if self._use_online_memory:
            features = features + self._compute_online_memory_features(chars)
        return self.linear_readout(features)

    def _local_window_stack(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, dim = x.shape
        window = self.config.local_window
        if window == 1:
            return x
        pad = torch.zeros((batch, window - 1, dim), dtype=x.dtype, device=x.device)
        padded = torch.cat([pad, x], dim=1)
        windows = padded.unfold(dimension=1, size=window, step=1)
        return windows.permute(0, 1, 3, 2).reshape(batch, timesteps, window * dim)

    def _expand_poly_features(self, local_features: torch.Tensor) -> torch.Tensor:
        """Expand local window features with polynomial terms (NVAR-style)."""
        if self._poly_order < 2:
            return local_features
        proj = self._poly_proj(local_features)  # [batch, seq, poly_proj_dim]
        # Quadratic: outer product (upper triangle)
        d = proj.shape[-1]
        i, j = torch.triu_indices(d, d, device=proj.device)
        quad = proj[..., i] * proj[..., j]  # [batch, seq, d*(d+1)/2]
        parts = [local_features, quad]
        if self._poly_order >= 3:
            # Cubic: element-wise cube of projection (cheap approximation)
            parts.append(proj ** 3)
        return torch.cat(parts, dim=-1)

    def _local_logits(self, chars: torch.Tensor) -> torch.Tensor:
        if self.local_readout is None:
            raise RuntimeError("causal-bank local path is disabled.")
        x = self._embed_local(chars)
        stacked = self._local_window_stack(x)
        if self._poly_order >= 2:
            stacked = self._expand_poly_features(stacked)
        return self.local_readout(stacked)

    def set_ngram_table(self, table) -> None:
        """Inject an n-gram table for trust-routing mode."""
        self._ngram_table = table

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        if self._trust_routing:
            return self._forward_trust_routing(chars)
        if self._patch_size > 1 and self._patch_causal_mode == "hybrid":
            return self._forward_hybrid(chars)
        if self._patch_size > 1:
            return self._forward_patched(chars)
        return self._forward_raw(chars)

    def _forward_trust_routing(self, chars: torch.Tensor) -> torch.Tensor:
        """Trust-routing: table predicts, neural calibrates."""
        batch, seq_len = chars.shape

        # Get substrate features (same as normal forward)
        if self.config.enable_linear:
            states, embed = self._linear_states(chars)
            features = torch.cat([states, embed], dim=-1)
        else:
            features = self._embed_local(chars)

        # Trust gate: [batch, seq, 1]
        trust = torch.sigmoid(self._trust_gate(features))

        # Fallback: cheap linear readout
        fallback_logits = self._fallback_readout(features)

        # Vectorized table lookup — no Python loop, pure numpy
        chars_np = chars.detach().cpu().numpy()
        table_log_probs_np = self._ngram_table.batch_lookup_log_probs(chars_np)
        table_logits = torch.from_numpy(table_log_probs_np).to(features.device)

        # Mix: trust * table + (1-trust) * fallback
        # Both in log-probability space
        fallback_log_probs = fallback_logits - torch.logsumexp(fallback_logits, dim=-1, keepdim=True)
        mixed_log_probs = trust * table_logits + (1 - trust) * fallback_log_probs

        return mixed_log_probs

    def _forward_hybrid(self, chars: torch.Tensor) -> torch.Tensor:
        """Hybrid patch: global SSM on patches, local window on raw bytes.

        The global/linear path encodes bytes into patches (reducing sequence
        length), runs the substrate, then upsamples back to per-byte features.
        The local path operates on raw bytes as usual.
        The readout merges both paths — the global context from patches plus
        the fine-grained byte context from the local window.
        """
        batch, seq_len = chars.shape
        P = self._patch_size
        n_patches = seq_len // P

        # === GLOBAL PATH: patch-level SSM ===
        if self.config.enable_linear:
            embed = self._embed_linear(chars)  # [batch, seq, embed_dim]
            # Group into patches and encode
            patch_embeds = embed.view(batch, n_patches, P * self.config.embedding_dim)
            encoded = self._patch_encoder(patch_embeds)  # [batch, n_patches, embed_dim]

            # Run substrate on patch sequence
            linear_in_proj = self.linear_in_proj.to(device=encoded.device, dtype=encoded.dtype)
            drive = torch.matmul(encoded, linear_in_proj)

            kernels = None
            if not self._learned_recurrence and not self._use_primary_state_substrate and self.config.linear_impl == "kernel":
                if self._recompute_kernel:
                    time_idx = self._kernel_time_idx[:n_patches].to(device=encoded.device)
                    delta = time_idx[:, None] - time_idx[None, :]
                    mask = delta >= 0
                    safe_delta = torch.where(mask, delta, torch.zeros_like(delta)).float()
                    decays = self.linear_decays.to(device=encoded.device, dtype=encoded.dtype)
                    kernel = decays[None, None, :] ** safe_delta[..., None]
                    kernel = kernel * mask[..., None].float()
                    kernels = kernel.permute(2, 0, 1)
                elif self.linear_kernel is not None:
                    kernels = self.linear_kernel[:, :n_patches, :n_patches].to(device=encoded.device, dtype=encoded.dtype)

            static_mode_gate = self._static_bank_mode_gate()
            states = self._apply_substrate(drive, encoded, n_patches, kernels)

            if self.num_blocks > 1 and hasattr(self, '_block_layers'):
                for block_layer in self._block_layers:
                    mixed = states + block_layer(states)
                    states = self._apply_substrate(mixed, encoded, n_patches, kernels)
                    states = self._apply_mode_gate(states, static_mode_gate)

            states = self._apply_mode_gate(states, static_mode_gate)
            if self._use_state_augment:
                states = states + self._state_augment(encoded)

            # Upsample: repeat each patch state for P byte positions
            # Shift by 1 patch: states[t] provides context for bytes in patch[t+1]
            # Pad with zeros at the start for the first patch
            shifted = torch.cat([
                torch.zeros(batch, 1, states.shape[-1], device=states.device, dtype=states.dtype),
                states[:, :-1, :],
            ], dim=1)  # [batch, n_patches, modes]
            upsampled_states = shifted.repeat_interleave(P, dim=1)  # [batch, seq, modes]

            # Patch-level features per byte position
            features = torch.cat([upsampled_states, embed], dim=-1)
            logits_linear = self.linear_readout(features)

            # Training noise
            noise_sigma = getattr(self.config, 'training_noise', 0.0)
            if noise_sigma > 0 and self.training:
                logits_linear = logits_linear + torch.randn_like(logits_linear) * noise_sigma
        else:
            logits_linear = None

        # === LOCAL PATH: raw byte-level (unchanged) ===
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        # === MERGE ===
        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            feat = torch.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], dim=-1)
            gate = torch.sigmoid(self.gate_proj(feat)) * self.config.local_scale

        return logits_linear + gate * logits_local

    def _forward_raw(self, chars: torch.Tensor) -> torch.Tensor:
        # Adaptive substrate: the recurrence IS the entire model. No local path.
        if getattr(self, '_use_adaptive_substrate', False):
            return self._linear_logits(chars).clone()

        logits_linear = self._linear_logits(chars) if self.config.enable_linear else None
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            features = torch.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], dim=-1)
            gate = torch.sigmoid(self.gate_proj(features)) * self.config.local_scale

        # Clone to avoid CUDA graph aliasing error with torch.compile
        return (logits_linear + gate * logits_local).clone()

    def _forward_patched(self, chars: torch.Tensor) -> torch.Tensor:
        """Patch-encoded forward: group bytes into patches, run substrate, decode back to bytes."""
        batch, seq_len = chars.shape
        P = self._patch_size
        # Pad sequence length to be divisible by patch_size
        if seq_len % P != 0:
            pad_len = P - (seq_len % P)
            chars = torch.nn.functional.pad(chars, (0, pad_len), value=0)
            seq_len = chars.shape[1]
            padded = True
        else:
            pad_len = 0
            padded = False

        n_patches = seq_len // P

        # Embed all bytes
        x = self._embed_linear(chars)  # [batch, seq, embed_dim]

        # Reshape into patches: [batch, n_patches, P * embed_dim]
        x_patched = x.reshape(batch, n_patches, P * self.config.embedding_dim)

        # Encode patches: [batch, n_patches, embed_dim]
        x_enc = self._patch_encoder(x_patched)

        # Run the substrate on the patch sequence
        static_mode_gate = self._static_bank_mode_gate()
        if self._use_state_augment:
            states = self._state_augment(x_enc)
            states = self._apply_mode_gate(states, static_mode_gate)
        else:
            linear_in_proj = self.linear_in_proj.to(device=x_enc.device, dtype=x_enc.dtype)
            drive = torch.matmul(x_enc, linear_in_proj)
            # Use timesteps = n_patches for substrate
            kernels = None
            if not self._learned_recurrence and not self._use_primary_state_substrate and self.config.linear_impl == "kernel":
                if self._recompute_kernel:
                    time_idx = self._kernel_time_idx[:n_patches].to(device=x_enc.device)
                    delta = time_idx[:, None] - time_idx[None, :]
                    mask = delta >= 0
                    safe_delta = torch.where(mask, delta, torch.zeros_like(delta)).float()
                    decays = self.linear_decays.to(device=x_enc.device, dtype=x_enc.dtype)
                    kernel = decays[None, None, :] ** safe_delta[..., None]
                    kernel = kernel * mask[..., None].float()
                    kernels = kernel.permute(2, 0, 1)
                elif self.linear_kernel is None:
                    raise RuntimeError("causal-bank kernel path is missing its materialized kernel.")
                else:
                    kernels = self.linear_kernel[:, :n_patches, :n_patches].to(device=x_enc.device, dtype=x_enc.dtype)

            states = self._apply_substrate(drive, x_enc, n_patches, kernels)
            states = self._apply_mode_gate(states, static_mode_gate)

        # Readout features: [batch, n_patches, modes + embed_dim]
        readout_input = torch.cat([states, x_enc], dim=-1)

        # === CAUSAL DECODER ===
        if self._patch_causal_mode == 'autoregressive':
            logits = self._decode_autoregressive(readout_input, chars, batch, seq_len, n_patches, P)
        elif self._patch_causal_mode == 'mlp_factored':
            logits = self._decode_factored(readout_input, batch, seq_len, n_patches, P)
        else:
            # Original (illegal) flat decoder
            patch_logits = self._patch_decoder(readout_input)
            logits = patch_logits.reshape(batch, seq_len, self.vocab_size)

        # Remove padding
        if padded:
            logits = logits[:, :seq_len - pad_len, :]

        return logits

    def _decode_autoregressive(self, ssm_features, chars, batch, seq_len, n_patches, P):
        """MEGABYTE-style: predict next patch's bytes autoregressively.

        For patch t, we predict the bytes of patch t+1:
          b0 from ssm_features[t]
          b1 from ssm_features[t] + embed(b0)
          b2 from ssm_features[t] + embed(b0) + embed(b1)  [via GRU hidden state]
          b3 from ssm_features[t] + embed(b2) + ... [via GRU hidden state]
        """
        vocab_size = self.vocab_size
        device = ssm_features.device
        dtype = ssm_features.dtype

        all_logits = torch.zeros(batch, seq_len, vocab_size, device=device, dtype=dtype)
        patches = chars.view(batch, n_patches, P)

        # For each patch position, generate P bytes autoregressively
        for t in range(n_patches):
            # Context: ssm_features for PREVIOUS patches (shifted by 1)
            if t == 0:
                ctx = torch.zeros(batch, ssm_features.shape[-1], device=device, dtype=dtype)
            else:
                ctx = ssm_features[:, t - 1, :]  # [batch, features]

            # GRU hidden state starts from zero
            h = torch.zeros(1, batch, self.config.embedding_dim, device=device, dtype=dtype)

            for b_idx in range(P):
                global_pos = t * P + b_idx
                if b_idx == 0:
                    prev_byte_embed = torch.zeros(batch, self.config.embedding_dim, device=device, dtype=dtype)
                else:
                    # Use the ACTUAL previous byte (teacher forcing during training)
                    prev_byte = patches[:, t, b_idx - 1]
                    prev_byte_embed = self._patch_byte_embed(prev_byte)

                # GRU input: [SSM context, previous byte embedding]
                gru_input = torch.cat([ctx, prev_byte_embed], dim=-1).unsqueeze(1)  # [batch, 1, dim]
                gru_out, h = self._patch_byte_decoder(gru_input, h)
                byte_logits = self._patch_byte_output(gru_out.squeeze(1))  # [batch, vocab]
                all_logits[:, global_pos, :] = byte_logits

        return all_logits

    def _decode_factored(self, ssm_features, batch, seq_len, n_patches, P):
        """Factored MLP: each byte position has its own head.

        No within-patch autoregression. Each byte is predicted independently
        from the SSM features of the PREVIOUS patch.
        """
        vocab_size = self.vocab_size
        device = ssm_features.device
        dtype = ssm_features.dtype

        all_logits = torch.zeros(batch, seq_len, vocab_size, device=device, dtype=dtype)

        for t in range(n_patches):
            if t == 0:
                ctx = torch.zeros(batch, ssm_features.shape[-1], device=device, dtype=dtype)
            else:
                ctx = ssm_features[:, t - 1, :]

            for b_idx in range(P):
                global_pos = t * P + b_idx
                byte_logits = self._patch_byte_heads[b_idx](ctx)
                all_logits[:, global_pos, :] = byte_logits

        return all_logits

    def substrate_regularization(self, step: int = 0) -> torch.Tensor:
        """Regularization for learnable substrate parameters.

        Returns a scalar loss term that should be added to the training loss.
        Returns 0 for frozen substrate.  When ``adaptive_reg=True`` on the
        config the strength grows with ``sqrt(step / 1000)``.
        """
        if not getattr(self, '_recompute_kernel', False) and not getattr(self, '_learned_recurrence', False):
            return torch.tensor(0.0, device=next(self.parameters()).device)

        dev = next(self.parameters()).device

        # --- Adaptive scale ---
        adaptive = getattr(self.config, 'adaptive_reg', False)
        scale = 1.0 + (step / 1000.0) ** 0.5 if adaptive else 1.0

        if not getattr(self, '_recompute_kernel', False):
            # Learned recurrence with no learnable decays: only adaptive scaling
            # applies to a small L2 penalty on gate weights if present.
            if self._learned_recurrence:
                reg = self._gate_weight.pow(2).sum() * 1e-4
                return reg * scale
            return torch.tensor(0.0, device=dev)

        decays = self.linear_decays
        reg = torch.tensor(0.0, device=decays.device)

        # 1. Keep decays in valid range (0, 1) via soft penalty
        # Decays outside (0.01, 0.9999) get penalized
        reg = reg + torch.relu(-decays + 0.01).sum() * 10.0  # penalty for < 0.01
        reg = reg + torch.relu(decays - 0.9999).sum() * 10.0  # penalty for > 0.9999

        # 2. Diversity penalty: penalize when modes become too similar
        # Use pairwise distance between decay values
        if decays.numel() > 1:
            flat = decays.flatten()
            diffs = flat.unsqueeze(0) - flat.unsqueeze(1)
            # Penalize small differences (modes collapsing together)
            similarity = torch.exp(-diffs.pow(2) / 0.01)
            # Zero out diagonal
            mask = 1.0 - torch.eye(flat.shape[0], device=decays.device)
            diversity_loss = (similarity * mask).sum() / max(mask.sum(), 1.0)
            reg = reg + diversity_loss * 0.1

        return reg * scale

    def band_balance_loss(self) -> torch.Tensor:
        """Sum balance_loss() from all band readouts that are RoutedSquaredReLUReadout.

        Returns 0 if no expert bands exist.
        """
        if not hasattr(self, '_band_readouts'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for readout in self._band_readouts:
            if isinstance(readout, RoutedSquaredReLUReadout):
                total = total + readout.balance_loss()
        return total

    def forward_with_mode_gate(self, chars: torch.Tensor, mode_gate: torch.Tensor | None) -> torch.Tensor:
        logits_linear = self._linear_logits(chars, mode_gate=mode_gate) if self.config.enable_linear else None
        logits_local = self._local_logits(chars) if self.config.enable_local else None

        if logits_linear is None:
            return logits_local
        if logits_local is None:
            return logits_linear

        if self.gate_proj is None:
            gate = self.config.local_scale
        else:
            ent_l, max_l, var_l = self._logit_features(logits_linear)
            ent_r, max_r, var_r = self._logit_features(logits_local)
            features = torch.stack([ent_l, ent_r, max_l, max_r, var_l, var_r], dim=-1)
            gate = torch.sigmoid(self.gate_proj(features)) * self.config.local_scale

        return logits_linear + gate * logits_local
