from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np


CAUSAL_BANK_FAMILY_ID = "causal-bank"
CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED = 42
CAUSAL_BANK_VARIANTS = (
    "base",
    "linear_only",
    "local_only",
    "gated",
    "window4",
    "window16",
    "shared_embedding",
)
CAUSAL_BANK_OSCILLATORY_SCHEDULES = (
    "logspace",
    "mincorr_greedy",
    "period_bucket_greedy",
)
CAUSAL_BANK_READOUT_KINDS = ("mlp", "tied_recursive", "routed_sqrelu_experts", "recurrent")
CAUSAL_BANK_INPUT_PROJ_SCHEMES = ("random", "orthogonal_rows", "split_banks", "kernel_energy")


def _logspace_half_lives(start: float, end: float, count: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(start), np.log(end), count, dtype=np.float32))


def _decays_from_half_lives(half_lives: np.ndarray) -> np.ndarray:
    return np.exp(np.log(0.5, dtype=np.float32) / half_lives.astype(np.float32, copy=False))


@dataclass(frozen=True)
class CausalBankConfig:
    embedding_dim: int = 32
    linear_modes: int = 256
    max_seq_len: int = 256
    linear_half_life_min: float = 1.5
    linear_half_life_max: float = 512.0
    linear_hidden: tuple[int, ...] = (128,)
    linear_readout_kind: str = "mlp"
    linear_readout_depth: int = 1
    linear_readout_num_experts: int = 4
    local_window: int = 8
    local_hidden: tuple[int, ...] = (128,)
    local_scale: float = 0.25
    mix_mode: str = "additive"
    share_embedding: bool = False
    linear_impl: str = "kernel"
    enable_linear: bool = True
    enable_local: bool = True
    oscillatory_frac: float = 0.0
    oscillatory_period_min: float = 4.0
    oscillatory_period_max: float = 64.0
    oscillatory_schedule: str = "logspace"
    oscillatory_candidate_period_count: int = 200
    oscillatory_candidate_half_life_count: int = 20
    static_bank_gate: bool = False
    bank_gate_span: float = 0.5
    input_proj_scheme: str = "random"
    init_seed: int = 42
    memory_kind: str = "none"
    substrate_mode: str = "frozen"  # "frozen", "learnable_decays", "learnable_mixing"
    num_blocks: int = 1
    block_mixing_ratio: float = 0.25  # bottleneck ratio for inter-block mixing
    state_dim: int = 0  # selective scan state dim (0 = use linear_modes, >0 = compressed state)
    num_heads: int = 1  # multi-head state (each head runs independent scan)
    patch_size: int = 1  # byte-to-patch grouping (1 = raw bytes, >1 = patch encoding)
    num_hemispheres: int = 1  # 1=uniform, 2=fast/slow split
    fast_hemisphere_ratio: float = 0.25  # fraction of state_dim for fast hemisphere
    fast_lr_mult: float = 4.0  # learning rate multiplier for fast hemisphere params


@dataclass(frozen=True)
class CausalBankFamilySpec:
    family_id: str = CAUSAL_BANK_FAMILY_ID
    deterministic_substrate_seed: int = CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED
    variants: tuple[str, ...] = CAUSAL_BANK_VARIANTS
    oscillatory_schedules: tuple[str, ...] = CAUSAL_BANK_OSCILLATORY_SCHEDULES
    linear_readout_kinds: tuple[str, ...] = CAUSAL_BANK_READOUT_KINDS
    input_proj_schemes: tuple[str, ...] = CAUSAL_BANK_INPUT_PROJ_SCHEMES


CAUSAL_BANK_FAMILY = CausalBankFamilySpec()


def validate_config(config: CausalBankConfig) -> None:
    if not config.enable_linear and not config.enable_local:
        raise ValueError("causal-bank must enable at least one path.")
    if config.mix_mode not in {"additive", "gated"}:
        raise ValueError(f"Unknown causal-bank mix_mode: {config.mix_mode}")
    if config.linear_impl not in {"kernel", "fft"}:
        raise ValueError(f"Unknown causal-bank linear_impl: {config.linear_impl}")
    if config.input_proj_scheme not in CAUSAL_BANK_INPUT_PROJ_SCHEMES:
        raise ValueError(f"Unknown causal-bank input_proj_scheme: {config.input_proj_scheme}")
    if config.linear_readout_kind not in CAUSAL_BANK_READOUT_KINDS:
        raise ValueError(f"Unknown causal-bank linear_readout_kind: {config.linear_readout_kind}")
    if config.linear_readout_depth < 1:
        raise ValueError("causal-bank linear_readout_depth must be >= 1.")
    if config.linear_readout_num_experts < 2:
        raise ValueError("causal-bank linear_readout_num_experts must be >= 2.")
    if config.oscillatory_frac < 0.0 or config.oscillatory_frac >= 1.0:
        raise ValueError("causal-bank oscillatory_frac must be in [0, 1).")
    if config.oscillatory_frac > 0.0 and config.linear_impl != "kernel":
        raise ValueError("causal-bank oscillatory modes currently require linear_impl='kernel'.")
    if config.oscillatory_schedule not in CAUSAL_BANK_OSCILLATORY_SCHEDULES:
        raise ValueError(f"Unknown causal-bank oscillatory_schedule: {config.oscillatory_schedule}")
    if config.oscillatory_candidate_period_count < 1 or config.oscillatory_candidate_half_life_count < 1:
        raise ValueError("causal-bank oscillatory candidate grid counts must be positive.")
    if config.local_window < 1:
        raise ValueError("causal-bank local_window must be >= 1.")
    if config.linear_half_life_min <= 0:
        raise ValueError("causal-bank linear_half_life_min must be positive.")
    if config.linear_half_life_max <= config.linear_half_life_min:
        raise ValueError("causal-bank linear_half_life_max must be > linear_half_life_min.")
    if config.oscillatory_frac > 0:
        if config.oscillatory_period_min <= 0:
            raise ValueError("causal-bank oscillatory_period_min must be positive.")
        if config.oscillatory_period_max <= config.oscillatory_period_min:
            raise ValueError("causal-bank oscillatory_period_max must be > oscillatory_period_min.")
    if config.substrate_mode not in ("frozen", "learnable_decays", "learnable_mixing", "learned_recurrence"):
        raise ValueError(f"Unknown causal-bank substrate_mode: {config.substrate_mode!r}")
    from open_predictive_coder.memory_protocol import MEMORY_KINDS
    if config.memory_kind not in MEMORY_KINDS:
        raise ValueError(f"Unknown causal-bank memory_kind: {config.memory_kind!r}; expected one of {MEMORY_KINDS}")
    if config.num_blocks < 1:
        raise ValueError("causal-bank num_blocks must be >= 1.")
    if config.block_mixing_ratio <= 0 or config.block_mixing_ratio > 1:
        raise ValueError("causal-bank block_mixing_ratio must be in (0, 1].")
    if config.state_dim < 0:
        raise ValueError("causal-bank state_dim must be >= 0.")
    if config.num_heads < 1:
        raise ValueError("causal-bank num_heads must be >= 1.")
    if config.state_dim > 0 and config.state_dim % config.num_heads != 0:
        raise ValueError("causal-bank state_dim must be divisible by num_heads.")
    if config.patch_size < 1:
        raise ValueError("causal-bank patch_size must be >= 1.")
    if config.num_hemispheres not in (1, 2):
        raise ValueError("causal-bank num_hemispheres must be 1 or 2.")
    if config.fast_hemisphere_ratio <= 0 or config.fast_hemisphere_ratio >= 1:
        raise ValueError("causal-bank fast_hemisphere_ratio must be in (0, 1).")
    if config.fast_lr_mult <= 0:
        raise ValueError("causal-bank fast_lr_mult must be positive.")


def learnable_substrate_keys(config: CausalBankConfig) -> tuple[str, ...]:
    """Return which substrate tensor names should be trainable parameters.

    Returns empty tuple for frozen mode. Used by downstream Torch/MLX models
    to decide register_buffer vs nn.Parameter.
    """
    if config.substrate_mode == "frozen":
        return ()
    if config.substrate_mode == "learnable_decays":
        return ("linear_decays",)
    if config.substrate_mode == "learnable_mixing":
        return ("linear_in_proj",)
    if config.substrate_mode == "learned_recurrence":
        return ("linear_in_proj", "linear_decays", "recurrence_gate")
    return ()


def osc_pair_count(config: CausalBankConfig) -> int:
    osc_pairs = int((config.linear_modes * config.oscillatory_frac) // 2)
    return max(min(osc_pairs, config.linear_modes // 2), 0)


def _random_in_proj(rng: np.random.Generator, embedding_dim: int, mode_count: int) -> np.ndarray:
    scale = 1.0 / math.sqrt(embedding_dim)
    return rng.standard_normal((embedding_dim, mode_count), dtype=np.float32) * scale


def _orthogonal_rows_in_proj(rng: np.random.Generator, embedding_dim: int, mode_count: int) -> np.ndarray:
    if mode_count <= 0:
        return np.zeros((embedding_dim, 0), dtype=np.float32)
    # When mode_count < embedding_dim, reduced QR on (mode_count, embedding_dim)
    # gives q of shape (mode_count, mode_count) — too few rows.  Use the
    # transposed factorisation so q always has embedding_dim rows.
    k = max(mode_count, embedding_dim)
    mat = rng.standard_normal((k, embedding_dim), dtype=np.float32)
    q, _ = np.linalg.qr(mat, mode="reduced")
    # q is (k, embedding_dim) → take first mode_count columns of q.T
    proj = q.T[:, :mode_count].astype(np.float32, copy=False)
    proj *= np.float32(math.sqrt(mode_count / embedding_dim))
    return proj


def _split_bank_in_proj(
    rng: np.random.Generator,
    embedding_dim: int,
    non_osc_modes: int,
    osc_mode_count: int,
) -> np.ndarray:
    in_proj = np.zeros((embedding_dim, non_osc_modes + osc_mode_count), dtype=np.float32)
    if non_osc_modes <= 0:
        in_proj[:, :] = _orthogonal_rows_in_proj(rng, embedding_dim, osc_mode_count)
        return in_proj
    if osc_mode_count <= 0:
        in_proj[:, :] = _orthogonal_rows_in_proj(rng, embedding_dim, non_osc_modes)
        return in_proj
    non_osc_dim = max(embedding_dim // 2, 1)
    osc_dim = max(embedding_dim - non_osc_dim, 1)
    in_proj[:non_osc_dim, :non_osc_modes] = _orthogonal_rows_in_proj(
        rng, non_osc_dim, non_osc_modes
    )
    osc_block = np.zeros((osc_dim, osc_mode_count), dtype=np.float32)
    osc_pairs = osc_mode_count // 2
    for idx in range(osc_pairs):
        base = _orthogonal_rows_in_proj(rng, osc_dim, 1).reshape(-1)
        start = 2 * idx
        osc_block[:, start] = base
        osc_block[:, start + 1] = base
    if osc_mode_count % 2 == 1:
        osc_block[:, -1] = _orthogonal_rows_in_proj(rng, osc_dim, 1).reshape(-1)
    in_proj[embedding_dim - osc_dim :, non_osc_modes:] = osc_block
    return in_proj


def _kernel_from_decays(decays: np.ndarray, max_seq_len: int) -> np.ndarray:
    time_idx = np.arange(max_seq_len, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    mask = delta >= 0
    safe_delta = np.where(mask, delta, 0).astype(np.float32)
    kernel = np.power(decays[None, None, :], safe_delta[..., None], dtype=np.float32)
    kernel = np.where(mask[..., None], kernel, 0.0).astype(np.float32)
    return np.transpose(kernel, (2, 0, 1))


def _kernel_from_damped_oscillators(
    decays: np.ndarray,
    periods: np.ndarray,
    max_seq_len: int,
) -> np.ndarray:
    time_idx = np.arange(max_seq_len, dtype=np.int32)
    delta = time_idx[:, None] - time_idx[None, :]
    mask = delta >= 0
    lags = np.where(mask, delta, 0).astype(np.float32)
    envelope = np.power(decays[:, None, None], lags[None, :, :], dtype=np.float32)
    omega = (2.0 * np.pi / periods.astype(np.float32, copy=False))[:, None, None]
    cos_kernel = np.where(
        mask[None, :, :], envelope * np.cos(omega * lags[None, :, :]), 0.0
    ).astype(np.float32)
    sin_kernel = np.where(
        mask[None, :, :], envelope * np.sin(omega * lags[None, :, :]), 0.0
    ).astype(np.float32)
    kernel = np.empty((decays.shape[0] * 2, max_seq_len, max_seq_len), dtype=np.float32)
    kernel[0::2] = cos_kernel
    kernel[1::2] = sin_kernel
    return kernel


def _oscillatory_half_lives(config: CausalBankConfig, count: int) -> np.ndarray:
    return _logspace_half_lives(
        max(config.linear_half_life_min, 2.0),
        config.linear_half_life_max,
        count,
    )


def _normalized_damped_oscillator_pair(period: float, decay: float, length: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    envelope = np.power(np.float32(decay), t, dtype=np.float32)
    omega = np.float32(2.0 * np.pi / period)
    pair = np.concatenate(
        [
            envelope * np.cos(omega * t),
            envelope * np.sin(omega * t),
        ]
    ).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(pair))
    if norm > 0.0:
        pair /= np.float32(norm)
    return pair


def _build_mincorr_greedy_schedule(
    config: CausalBankConfig, osc_pairs: int
) -> tuple[np.ndarray, np.ndarray]:
    if osc_pairs <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    candidate_periods = np.linspace(
        config.oscillatory_period_min,
        config.oscillatory_period_max,
        config.oscillatory_candidate_period_count,
        dtype=np.float32,
    )
    candidate_half_lives = _oscillatory_half_lives(
        config, config.oscillatory_candidate_half_life_count
    )
    grid_periods, grid_half_lives = np.meshgrid(
        candidate_periods, candidate_half_lives, indexing="xy"
    )
    flat_periods = grid_periods.reshape(-1)
    flat_half_lives = grid_half_lives.reshape(-1)
    if osc_pairs > flat_periods.shape[0]:
        raise ValueError(
            "causal-bank mincorr_greedy needs at least as many candidate (period, half-life) pairs as oscillatory pairs."
        )

    decays = _decays_from_half_lives(flat_half_lives)
    basis = np.stack(
        [
            _normalized_damped_oscillator_pair(
                float(period), float(decay), config.max_seq_len
            )
            for period, decay in zip(flat_periods, decays)
        ],
        axis=0,
    )
    gram = basis @ basis.T
    abs_gram = np.abs(gram).astype(np.float32, copy=False)
    initial_scores = (np.sum(abs_gram, axis=1, dtype=np.float32) - 1.0) / max(
        abs_gram.shape[0] - 1, 1
    )
    first = int(np.argmin(initial_scores))
    selected = np.zeros((abs_gram.shape[0],), dtype=bool)
    selected[first] = True
    best_corr = abs_gram[first].copy()
    best_corr[selected] = np.inf

    while int(np.sum(selected)) < osc_pairs:
        next_idx = int(np.argmin(best_corr))
        selected[next_idx] = True
        best_corr = np.maximum(best_corr, abs_gram[next_idx])
        best_corr[selected] = np.inf

    chosen = np.flatnonzero(selected)
    return (
        flat_periods[chosen].astype(np.float32, copy=False),
        flat_half_lives[chosen].astype(np.float32, copy=False),
    )


def _greedy_periods_for_decay(
    candidate_periods: np.ndarray,
    decay: float,
    count: int,
    length: int,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    if count > candidate_periods.shape[0]:
        raise ValueError(
            "causal-bank period_bucket_greedy needs at least as many candidate periods as bucket slots."
        )
    basis = np.stack(
        [
            _normalized_damped_oscillator_pair(float(period), float(decay), length)
            for period in candidate_periods
        ],
        axis=0,
    )
    abs_gram = np.abs(basis @ basis.T).astype(np.float32, copy=False)
    initial_scores = (np.sum(abs_gram, axis=1, dtype=np.float32) - 1.0) / max(
        abs_gram.shape[0] - 1, 1
    )
    first = int(np.argmin(initial_scores))
    selected = np.zeros((abs_gram.shape[0],), dtype=bool)
    selected[first] = True
    best_corr = abs_gram[first].copy()
    best_corr[selected] = np.inf

    while int(np.sum(selected)) < count:
        next_idx = int(np.argmin(best_corr))
        selected[next_idx] = True
        best_corr = np.maximum(best_corr, abs_gram[next_idx])
        best_corr[selected] = np.inf
    return candidate_periods[np.flatnonzero(selected)].astype(np.float32, copy=False)


def _build_period_bucket_greedy_schedule(
    config: CausalBankConfig, osc_pairs: int
) -> tuple[np.ndarray, np.ndarray]:
    if osc_pairs <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    half_lives = _oscillatory_half_lives(config, osc_pairs)
    candidate_periods = np.linspace(
        config.oscillatory_period_min,
        config.oscillatory_period_max,
        config.oscillatory_candidate_period_count,
        dtype=np.float32,
    )
    bucket_count = min(config.oscillatory_candidate_half_life_count, osc_pairs)
    edges = np.linspace(0, osc_pairs, bucket_count + 1, dtype=np.int32)
    periods = np.empty((osc_pairs,), dtype=np.float32)
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        bucket_half_lives = half_lives[start:end]
        rep_half_life = float(
            np.exp(np.mean(np.log(bucket_half_lives, dtype=np.float32), dtype=np.float32))
        )
        rep_decay = float(_decays_from_half_lives(np.array([rep_half_life], dtype=np.float32))[0])
        chosen = _greedy_periods_for_decay(
            candidate_periods,
            rep_decay,
            end - start,
            config.max_seq_len,
        )
        periods[start:end] = np.sort(chosen, kind="stable")
    return periods.astype(np.float32, copy=False), half_lives.astype(np.float32, copy=False)


def _build_oscillatory_schedule(
    config: CausalBankConfig, osc_pairs: int
) -> tuple[np.ndarray, np.ndarray]:
    if osc_pairs <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if config.oscillatory_schedule == "logspace":
        periods = _logspace_half_lives(
            config.oscillatory_period_min, config.oscillatory_period_max, osc_pairs
        )
        half_lives = _oscillatory_half_lives(config, osc_pairs)
        return periods.astype(np.float32, copy=False), half_lives.astype(np.float32, copy=False)
    if config.oscillatory_schedule == "mincorr_greedy":
        return _build_mincorr_greedy_schedule(config, osc_pairs)
    if config.oscillatory_schedule == "period_bucket_greedy":
        return _build_period_bucket_greedy_schedule(config, osc_pairs)
    raise ValueError(f"Unknown causal-bank oscillatory_schedule: {config.oscillatory_schedule}")


def build_linear_bank(config: CausalBankConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(CAUSAL_BANK_DETERMINISTIC_SUBSTRATE_SEED)
    osc_pairs = osc_pair_count(config)
    non_osc_modes = config.linear_modes - 2 * osc_pairs
    osc_mode_count = 2 * osc_pairs

    if config.input_proj_scheme == "random":
        in_proj = _random_in_proj(rng, config.embedding_dim, config.linear_modes)
    elif config.input_proj_scheme == "orthogonal_rows":
        in_proj = _orthogonal_rows_in_proj(rng, config.embedding_dim, config.linear_modes)
    elif config.input_proj_scheme == "split_banks":
        in_proj = _split_bank_in_proj(rng, config.embedding_dim, non_osc_modes, osc_mode_count)
    elif config.input_proj_scheme == "kernel_energy":
        in_proj = _random_in_proj(rng, config.embedding_dim, config.linear_modes)
    else:
        raise ValueError(f"Unknown causal-bank input_proj_scheme: {config.input_proj_scheme}")

    kernels: list[np.ndarray] = []
    decay_parts: list[np.ndarray] = []

    if non_osc_modes > 0:
        half_lives = _logspace_half_lives(
            config.linear_half_life_min, config.linear_half_life_max, non_osc_modes
        )
        decays = _decays_from_half_lives(half_lives)
        kernels.append(_kernel_from_decays(decays, config.max_seq_len))
        decay_parts.append(decays)

    if osc_pairs > 0:
        periods, half_lives = _build_oscillatory_schedule(config, osc_pairs)
        decays = _decays_from_half_lives(half_lives)
        if config.input_proj_scheme not in {"split_banks"}:
            for idx in range(osc_pairs):
                start = non_osc_modes + 2 * idx
                base = _random_in_proj(rng, config.embedding_dim, 1).reshape(-1)
                if config.input_proj_scheme == "orthogonal_rows":
                    base = _orthogonal_rows_in_proj(rng, config.embedding_dim, 1).reshape(-1)
                in_proj[:, start] = base
                in_proj[:, start + 1] = base
        kernels.append(_kernel_from_damped_oscillators(decays, periods, config.max_seq_len))
        decay_parts.append(np.repeat(decays, 2))

    if not kernels:
        raise ValueError("causal-bank linear bank must contain at least one mode.")

    kernel = np.concatenate(kernels, axis=0).astype(np.float32, copy=False)
    decays_full = np.concatenate(decay_parts, axis=0).astype(np.float32, copy=False)
    if config.input_proj_scheme == "kernel_energy":
        mode_energy = np.sqrt(np.mean(kernel * kernel, axis=(1, 2), dtype=np.float32)).astype(
            np.float32, copy=False
        )
        mode_energy = mode_energy / max(float(np.mean(mode_energy)), 1e-6)
        in_proj = in_proj * mode_energy[None, :]
    return in_proj.astype(np.float32, copy=False), decays_full, kernel


def apply_variant(config: CausalBankConfig, variant: str) -> CausalBankConfig:
    if variant == "base":
        return config
    if variant == "linear_only":
        return replace(config, enable_local=False)
    if variant == "local_only":
        return replace(config, enable_linear=False)
    if variant == "gated":
        return replace(config, mix_mode="gated")
    if variant == "window4":
        return replace(config, local_window=4)
    if variant == "window16":
        return replace(config, local_window=16)
    if variant == "shared_embedding":
        return replace(config, share_embedding=True)
    raise ValueError(f"Unknown causal-bank variant: {variant}")


def scale_config(config: CausalBankConfig, scale: float) -> CausalBankConfig:
    if scale == 1.0:
        return config
    return replace(
        config,
        embedding_dim=max(int(round(config.embedding_dim * scale)), 1),
        linear_modes=max(int(round(config.linear_modes * scale)), 1),
        linear_hidden=tuple(max(int(round(width * scale)), 1) for width in config.linear_hidden),
        local_hidden=tuple(max(int(round(width * scale)), 1) for width in config.local_hidden),
    )
