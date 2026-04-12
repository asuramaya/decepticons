"""Per-byte prediction difficulty measurement.

The general primitive for prediction-aware tokenizer construction.
Any model that produces per-token loss can produce a difficulty array.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def byte_difficulty(
    per_token_loss: np.ndarray,
    token_ids: np.ndarray,
    token_byte_lengths: np.ndarray,
) -> np.ndarray:
    """Distribute per-token loss across the bytes each token covers.

    Args:
        per_token_loss: float32 [num_tokens] -- cross-entropy per token (nats).
        token_ids: int32 [num_tokens] -- token IDs from the tokenizer.
        token_byte_lengths: int32 [num_tokens] -- number of bytes each token covers.

    Returns:
        float32 [total_bytes] -- per-byte difficulty.
        Each byte inherits the loss of the token that covers it,
        divided by the number of bytes in that token.
        Easy tokens (low loss, many bytes) produce low per-byte difficulty.
        Hard tokens (high loss, few bytes) produce high per-byte difficulty.
    """
    per_token_loss = np.asarray(per_token_loss, dtype=np.float32)
    token_byte_lengths = np.asarray(token_byte_lengths, dtype=np.int32)

    total_bytes = int(token_byte_lengths.sum())
    result = np.empty(total_bytes, dtype=np.float32)

    pos = 0
    for i in range(len(per_token_loss)):
        n_bytes = int(token_byte_lengths[i])
        if n_bytes > 0:
            result[pos : pos + n_bytes] = per_token_loss[i] / n_bytes
        pos += n_bytes

    return result


def byte_difficulty_from_model(
    model: Any,
    dataset: Any,
    *,
    num_sequences: int = 200,
    seq_len: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Run a model on data and return per-byte difficulty.

    This is the convenience wrapper. It handles the forward pass,
    loss computation, and byte-length lookup. The model must support
    forward(chars) -> logits [batch, seq, vocab].

    Args:
        model: a torch model with forward(chars) -> logits.
        dataset: a TokenShardDataset with test_stream and vocab_size.
        num_sequences: how many sequences to run.
        seq_len: sequence length.
        device: torch device.

    Returns:
        float32 [num_sequences * seq_len * avg_bytes_per_token] -- per-byte difficulty.
    """
    import torch

    model.eval()
    all_losses: list[np.ndarray] = []
    all_byte_lengths: list[np.ndarray] = []

    sp = None
    try:
        import sentencepiece as spm
        tokenizer_path = getattr(dataset, 'tokenizer_path', None)
        if tokenizer_path:
            sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    except Exception:  # noqa: S110
        pass  # sentencepiece unavailable — fall back to estimated byte lengths

    with torch.no_grad():
        for _ in range(num_sequences):
            tokens = dataset.test_stream.take(seq_len + 1)
            input_ids = torch.tensor(tokens[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
            target_ids = torch.tensor(tokens[1 : seq_len + 1], dtype=torch.long, device=device)

            logits = model(input_ids).squeeze(0)

            loss = torch.nn.functional.cross_entropy(
                logits, target_ids, reduction="none"
            )
            all_losses.append(loss.cpu().numpy())

            if sp is not None:
                byte_lens = np.array([
                    len(sp.id_to_piece(int(tid)).encode("utf-8").replace(b"\xe2\x96\x81", b" "))
                    if not sp.is_byte(int(tid)) else 1
                    for tid in tokens[1 : seq_len + 1]
                ], dtype=np.int32)
            else:
                bpt = getattr(dataset, 'test_bytes_per_token', 2.436)
                byte_lens = np.full(seq_len, max(1, int(round(bpt))), dtype=np.int32)
            all_byte_lengths.append(byte_lens)

    losses = np.concatenate(all_losses)
    byte_lens = np.concatenate(all_byte_lengths)
    token_ids = np.zeros(len(losses), dtype=np.int32)

    return byte_difficulty(losses, token_ids, byte_lens)


def embedding_difficulty(embedding_weight: np.ndarray) -> np.ndarray:
    """Extract per-token difficulty from embedding norms.

    Heinrich found embedding norm correlates r=0.99 with substrate
    displacement. The model already encodes difficulty in its embeddings.
    This is instant (one weight read) vs byte_difficulty_from_model
    (200 forward passes).

    Args:
        embedding_weight: float32 [vocab_size, embed_dim] — the embedding matrix.

    Returns:
        float32 [vocab_size] — per-token difficulty (L2 norm of embedding).
    """
    w = np.asarray(embedding_weight, dtype=np.float32)
    return np.linalg.norm(w, axis=-1)
