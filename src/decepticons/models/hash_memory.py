"""Hash-indexed token memory for the causal bank architecture.

Stores specific token embeddings at position-hashed addresses.
Retrieves by content similarity (dot-product attention over M slots).
**O(n·M·D) per forward** (no python loops — fully vectorized causal gather).
Preserves token identity, legal (past-only writes).

Unlike the EMA which produces a decayed average of all past tokens,
this memory stores EXACT embeddings of specific past tokens. The hash
collision is the forgetting mechanism — newer writes overwrite older
ones at the same slot.
"""

from __future__ import annotations

import torch
from torch import nn


class HashMemory(nn.Module):
    """Fixed-size hash-indexed memory with content-based retrieval.

    Slot assignment: position t writes to slot (t mod M).
    At read time, slot s contains the most recent write from any
    past position p < t with p mod M == s. The memory state at each
    timestep is reconstructed via a vectorized gather — no autograd-
    breaking in-place writes, no python loop in T.
    """

    def __init__(self, embed_dim: int, memory_dim: int, num_slots: int = 64):
        super().__init__()
        self.num_slots = int(num_slots)
        self.memory_dim = int(memory_dim)
        self.write_proj = nn.Linear(embed_dim, memory_dim)
        self.read_query = nn.Linear(embed_dim, memory_dim)
        self.read_out = nn.Linear(memory_dim, embed_dim)
        self._scale = memory_dim ** -0.5

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, T, embed_dim]
        returns:    [B, T, embed_dim] — retrieved memory context per position
        """
        B, T, _ = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype
        M = self.num_slots

        write_vals = self.write_proj(embeddings)   # [B, T, memory_dim]
        read_queries = self.read_query(embeddings)  # [B, T, memory_dim]

        # Build per-position slot read-index table on int64.
        # For (t, s), read the most recent past write to slot s strictly before t:
        #   diff[t, s] = t - 1 - s    (largest p ≤ t-1 with p mod M == s == p when adjusted)
        #   if diff < 0: slot has no valid write yet (mask out).
        #   else:        p = (diff // M) * M + s
        t_idx = torch.arange(T, device=device, dtype=torch.long)
        s_idx = torch.arange(M, device=device, dtype=torch.long)
        diff = t_idx[:, None] - 1 - s_idx[None, :]   # [T, M]
        valid = diff >= 0                              # [T, M]
        p = (diff // M) * M + s_idx[None, :]           # [T, M]
        p = p.clamp(min=0)                             # safe gather index

        # Vectorized causal gather: memory[b, t, s, d] = write_vals[b, p[t,s], d]
        # Advanced indexing on dim=1 with shape [T, M] expands that dim → [B, T, M, memory_dim].
        memory = write_vals[:, p, :]                   # [B, T, M, memory_dim]

        # Per-position dot-product attention over slots.
        q = read_queries                               # [B, T, memory_dim]
        sim = (q.unsqueeze(2) * memory).sum(-1) * self._scale   # [B, T, M]
        sim = sim.masked_fill(~valid.unsqueeze(0), float("-inf"))
        attn = torch.softmax(sim, dim=-1)
        # Position 0 has no valid slots → softmax is all-NaN; replace with 0.
        attn = attn.nan_to_num(0.0)

        retrieved = (attn.unsqueeze(-1) * memory).sum(dim=2)    # [B, T, memory_dim]
        out = self.read_out(retrieved).to(dtype)                 # [B, T, embed_dim]
        return out
