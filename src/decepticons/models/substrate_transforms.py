"""Substrate transform primitives for the causal bank architecture.

These modules sit between the substrate (EMA/gated-delta) output and the
readout. They address the manifold waste problem: 43% position counter,
47% ghosts, 6% useful content.

Four primitives:
  - MagnitudeNormalizer: kills the position counter (0-1 params)
  - ModeSelector: per-token soft attention over modes (~130k params)
  - SubstrateBank: splits modes into independent banks by token type (~2k params)
  - TemporalAttention: recovers order from substrate snapshots (~50k params)
"""

from __future__ import annotations

import torch
from torch import nn


class OverwriteGate(nn.Module):
    """Per-mode gate that interpolates between EMA decay and full overwrite.

    The EMA accumulates forever — old tokens fade but never vanish. The
    OverwriteGate gives each mode the ability to REPLACE its state when
    the current token makes the old state stale.

    state[m] = gate * drive + (1 - gate) * decay * state[m]

    gate ≈ 0: normal EMA (remember)
    gate ≈ 1: overwrite with current drive (forget and replace)

    Bias initialized near 0 (default: remember). Training pushes toward 1
    for modes that benefit from forgetting.

    ~n_modes * embed_dim params.
    """

    def __init__(self, embed_dim: int, n_modes: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, n_modes)
        # Initialize bias to -3.0 so sigmoid ≈ 0.047 (strong default toward remember)
        with torch.no_grad():
            self.gate_proj.bias.fill_(-3.0)

    def forward(
        self,
        states: torch.Tensor,
        drive: torch.Tensor,
        embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        states: [..., n_modes] — current substrate state (after EMA decay)
        drive:  [..., n_modes] — current drive signal (pre-decay input)
        embed:  [..., embed_dim] — current token embedding
        returns: [..., n_modes] — gated state
        """
        gate = torch.sigmoid(self.gate_proj(embed))  # [..., n_modes]
        self._last_gate_values = gate.detach()
        return gate * drive + (1.0 - gate) * states


class SubstrateBankRouter(nn.Module):
    """Route tokens to substrate banks before the EMA runs.

    Instead of every token driving every mode, the router assigns each
    token to a subset of modes. Content tokens drive content modes.
    Structure tokens drive structure modes. No cross-contamination.

    The router produces per-token, per-bank soft weights. The drive signal
    is scaled by these weights before entering the substrate. Each bank's
    modes only accumulate signal from tokens routed to that bank.

    Split by half-life quartile at init (bank 0 = fastest modes, etc.)
    or let the router learn the assignment.

    ~embed_dim * n_banks params for the router.
    """

    def __init__(self, embed_dim: int, n_banks: int):
        super().__init__()
        self.router = nn.Linear(embed_dim, n_banks)
        self.n_banks = n_banks

    def forward(self, drive: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """Scale drive signal per bank.

        drive: [..., n_modes] — full drive signal for all modes
        embed: [..., embed_dim] — current token embedding
        returns: [..., n_modes] — bank-routed drive signal
        """
        n_modes = drive.shape[-1]
        bank_size = n_modes // self.n_banks

        weights = torch.softmax(self.router(embed), dim=-1)  # [..., n_banks]

        # Expand weights to per-mode: each bank's modes get that bank's weight
        mode_weights = torch.zeros_like(drive)
        for b in range(self.n_banks):
            start = b * bank_size
            end = start + bank_size if b < self.n_banks - 1 else n_modes
            mode_weights[..., start:end] = weights[..., b:b + 1]

        return drive * mode_weights


class MagnitudeNormalizer(nn.Module):
    """Normalize substrate magnitude to kill the position counter.

    PC1 of the substrate correlates r=0.91 with position — 43% of the
    manifold is a clock. Dividing by L2 norm makes the substrate a unit
    sphere where direction encodes content, magnitude encodes nothing.

    Optionally preserves log-magnitude as a side feature (1 param for
    the learned gate that controls how much magnitude info to keep).
    """

    def __init__(self, n_modes: int, *, keep_magnitude: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.keep_magnitude = keep_magnitude
        if keep_magnitude:
            self._mag_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: [..., n_modes] -> [..., n_modes] (+ optional magnitude channel)"""
        norm = states.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        self._last_pre_norm = norm.detach()  # [..., 1]
        normalized = states / norm
        if self.keep_magnitude:
            mag_weight = torch.sigmoid(self._mag_gate)
            log_mag = torch.log(norm + self.eps)
            return normalized + mag_weight * log_mag * (states / norm)
        return normalized


class ModeSelector(nn.Module):
    """Per-token soft attention over substrate modes.

    Current readout reads ALL modes at every position. ModeSelector produces
    a per-token mask that weights which modes are relevant to the current
    token. The position-counter mode gets masked out at late positions.
    Ghost modes get masked out when they carry stale content.

    selector(embedding) -> weights[n_modes]
    output = states * weights
    """

    def __init__(self, embed_dim: int, n_modes: int, *, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(embed_dim, n_modes)
        self.temperature = temperature

    def forward(self, states: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        states: [..., n_modes] — substrate output
        embed:  [..., embed_dim] — current token embedding
        returns: [..., n_modes] — mode-selected substrate
        """
        weights = torch.sigmoid(self.proj(embed) / self.temperature)  # [..., n_modes]
        self._last_weights = weights.detach()
        return states * weights


class SubstrateRouter(nn.Module):
    """Route tokens to independent substrate banks by learned type.

    Instead of one substrate driven by all tokens, split into N banks
    each driven by a subset. The router produces soft assignment weights
    per token. Each bank accumulates a cleaner signal because it's not
    contaminated by irrelevant token types.

    This module doesn't own the substrates — it produces the routing
    weights. The substrate forward pass uses these weights to scale
    the drive signal per bank.

    router(embedding) -> weights[n_banks]  (soft assignment, sums to 1)
    """

    def __init__(self, embed_dim: int, n_banks: int):
        super().__init__()
        self.router = nn.Linear(embed_dim, n_banks)
        self.n_banks = n_banks

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """embed: [..., embed_dim] -> [..., n_banks] soft routing weights."""
        return torch.softmax(self.router(embed), dim=-1)


class TemporalAttention(nn.Module):
    """Lightweight cross-attention over substrate snapshots.

    Every K positions, snapshot the substrate state into a fixed-size
    memory bank. At readout time, attend to the bank to recover order
    information the EMA centroid destroys.

    Cost: O(n/K * M) where M is bank size. At K=64, M=8: ~1% of full attention.

    The bank is managed externally (by the model's forward pass).
    This module only does the attention read.
    """

    def __init__(self, state_dim: int, *, num_heads: int = 2, head_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(state_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(state_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(state_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, state_dim, bias=False)
        self._scale = head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        bank: torch.Tensor,
        snapshot_interval: int | None = None,
    ) -> torch.Tensor:
        """
        query: [batch, seq, state_dim] — current substrate states
        bank:  [batch, M, state_dim]   — snapshot memory bank
        snapshot_interval: spacing between snapshots (for causal masking)
        returns: [batch, seq, state_dim] — attended context
        """
        batch, seq, _ = query.shape
        M = bank.shape[1]
        if M == 0:
            return torch.zeros_like(query)

        # Project
        q = self.q_proj(query).reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(bank).reshape(batch, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(bank).reshape(batch, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: [batch, heads, seq, M]
        attn = (q @ k.transpose(-2, -1)) * self._scale

        # Causal mask: position t can only attend to snapshots taken at positions <= t
        if snapshot_interval is not None and snapshot_interval > 0:
            snapshot_positions = torch.arange(0, seq, snapshot_interval, device=query.device)[:M]
            query_positions = torch.arange(seq, device=query.device)
            causal_mask = query_positions[:, None] >= snapshot_positions[None, :]  # [seq, M]
            attn = attn.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        self._last_attn_weights = attn.detach()  # [batch, heads, seq, M]

        # Aggregate: [batch, heads, seq, head_dim]
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        result = self.out_proj(out)
        self._last_output = result.detach()  # [batch, seq, state_dim]
        return result

    @staticmethod
    def build_bank(
        states: torch.Tensor,
        snapshot_interval: int = 64,
    ) -> torch.Tensor:
        """Build a memory bank by snapshotting substrate states at intervals.

        states: [batch, seq, state_dim]
        returns: [batch, n_snapshots, state_dim]
        """
        indices = list(range(0, states.shape[1], snapshot_interval))
        if not indices:
            return states[:, :0, :]  # empty bank
        return states[:, indices, :]
