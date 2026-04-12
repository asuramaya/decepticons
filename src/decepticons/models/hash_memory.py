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

        # Initialize empty memory per sequence
        memory = torch.zeros(batch, self.num_slots, self.memory_dim, device=device, dtype=dtype)
        # Track which slots have been written to (for masking empty slots)
        written = torch.zeros(batch, self.num_slots, device=device, dtype=torch.bool)

        outputs = torch.zeros(batch, seq, embed_dim, device=device, dtype=dtype)

        # Project all embeddings for writing and querying upfront
        write_vals = self.write_proj(embeddings)  # [batch, seq, memory_dim]
        read_queries = self.read_query(embeddings)  # [batch, seq, memory_dim]

        for t in range(seq):
            # READ: attend over memory with current query
            q = read_queries[:, t, :]  # [batch, memory_dim]
            # Similarity with all slots: [batch, num_slots]
            sim = torch.bmm(
                q.unsqueeze(1),  # [batch, 1, memory_dim]
                memory.transpose(1, 2),  # [batch, memory_dim, num_slots]
            ).squeeze(1) * self._scale  # [batch, num_slots]

            # Mask unwritten slots
            sim = sim.masked_fill(~written, float("-inf"))

            # Softmax attention (all -inf → uniform zeros after softmax, which is fine)
            attn = torch.softmax(sim, dim=-1)  # [batch, num_slots]
            # Handle all-masked case (first position, no writes yet)
            attn = attn.nan_to_num(0.0)

            # Retrieve: [batch, memory_dim]
            retrieved = torch.bmm(
                attn.unsqueeze(1),  # [batch, 1, num_slots]
                memory,  # [batch, num_slots, memory_dim]
            ).squeeze(1)

            outputs[:, t, :] = self.read_out(retrieved)

            # WRITE: store current token at hashed address
            addr = t % self.num_slots
            memory[:, addr, :] = write_vals[:, t, :]
            written[:, addr] = True

        return outputs
