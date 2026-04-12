"""Hash-indexed token memory for the causal bank architecture.

Stores specific token embeddings at position-hashed addresses.
Retrieves by content similarity (dot product attention over M slots).
O(M) per position, preserves token identity, legal (past-only writes).

Unlike the EMA which produces a decayed average of all past tokens,
this memory stores EXACT embeddings of specific past tokens. The hash
collision is the forgetting mechanism — newer writes overwrite older ones.
"""

from __future__ import annotations

import torch
from torch import nn


class HashMemory(nn.Module):
    """Fixed-size hash-indexed memory with content-based retrieval.

    Write: memory[hash(position % M)] = projection(embedding)
    Read: attend over all M slots with current embedding as query

    The memory is NOT a parameter — it's a runtime buffer that gets
    written during the forward pass. Each sequence starts with a
    cleared memory.
    """

    def __init__(self, embed_dim: int, memory_dim: int, num_slots: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.write_proj = nn.Linear(embed_dim, memory_dim)
        self.read_query = nn.Linear(embed_dim, memory_dim)
        self.read_out = nn.Linear(memory_dim, embed_dim)
        self._scale = memory_dim ** -0.5

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [batch, seq, embed_dim] — token embeddings
        returns: [batch, seq, embed_dim] — retrieved memory context
        """
        batch, seq, embed_dim = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype

        # Project all embeddings for writing and querying upfront
        write_vals = self.write_proj(embeddings)  # [batch, seq, memory_dim]
        read_queries = self.read_query(embeddings)  # [batch, seq, memory_dim]

        # Build memory states without in-place ops (autograd-safe)
        # For each position t, memory contains write_vals from positions that hash to each slot
        # and were written before t
        output_list: list[torch.Tensor] = []

        # Pre-compute which positions map to which slots
        slot_assignments = [t % self.num_slots for t in range(seq)]

        for t in range(seq):
            # Build memory snapshot at time t: contains all writes from positions < t
            if t == 0:
                # No past writes — output zero
                output_list.append(torch.zeros(batch, embed_dim, device=device, dtype=dtype))
                continue

            # Gather the most recent write for each slot from positions < t
            slot_contents = []
            slot_valid = []
            for s in range(self.num_slots):
                # Find the latest position < t that maps to slot s
                latest = -1
                for past_t in range(t - 1, -1, -1):
                    if slot_assignments[past_t] == s:
                        latest = past_t
                        break
                if latest >= 0:
                    slot_contents.append(write_vals[:, latest, :])
                    slot_valid.append(True)
                else:
                    slot_contents.append(torch.zeros(batch, self.memory_dim, device=device, dtype=dtype))
                    slot_valid.append(False)

            memory = torch.stack(slot_contents, dim=1)  # [batch, num_slots, memory_dim]
            valid_mask = torch.tensor(slot_valid, device=device, dtype=torch.bool)

            # READ: attend over memory
            q = read_queries[:, t, :]  # [batch, memory_dim]
            sim = torch.bmm(
                q.unsqueeze(1),
                memory.transpose(1, 2),
            ).squeeze(1) * self._scale

            sim = sim.masked_fill(~valid_mask.unsqueeze(0).expand(batch, -1), float("-inf"))
            attn = torch.softmax(sim, dim=-1)
            attn = attn.nan_to_num(0.0)

            retrieved = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)
            output_list.append(self.read_out(retrieved))

        return torch.stack(output_list, dim=1)
