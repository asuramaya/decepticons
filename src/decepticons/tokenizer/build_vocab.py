"""Build a prediction-aware vocabulary from per-byte difficulty scores.

Standard BPE merges the most frequent byte pairs. This builder merges
the EASIEST byte pairs — sequences the model already predicts well.
Hard sequences stay split so the substrate gets more steps on them.

Usage:
    difficulty = byte_difficulty_from_model(model, dataset)
    vocab = build_vocab(difficulty, text, vocab_size=8192)
"""

from __future__ import annotations

import numpy as np


def _bigram_difficulty(difficulty: np.ndarray, text: bytes) -> np.ndarray:
    """Aggregate per-byte difficulty into per-bigram difficulty.

    Returns: float32 [256, 256] — mean difficulty of byte b given preceding byte a.
    """
    counts = np.zeros((256, 256), dtype=np.int64)
    totals = np.zeros((256, 256), dtype=np.float64)

    n = min(len(difficulty), len(text))
    for i in range(1, n):
        a = text[i - 1]
        b = text[i]
        counts[a, b] += 1
        totals[a, b] += difficulty[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(counts > 0, totals / counts, 0.0)
    return mean.astype(np.float32)


def score_piece(piece: bytes, bigram_diff: np.ndarray) -> float:
    """Score a candidate vocabulary piece by the difficulty of its internal transitions.

    Low score = easy internal transitions = safe to merge (model predicts them anyway).
    High score = hard internal transitions = should stay split.

    For a single byte, score is 0 (no internal transitions).
    For multi-byte pieces, score is the mean bigram difficulty of consecutive bytes.
    """
    if len(piece) <= 1:
        return 0.0
    total = 0.0
    for i in range(1, len(piece)):
        total += float(bigram_diff[piece[i - 1], piece[i]])
    return total / (len(piece) - 1)


def build_vocab(
    difficulty: np.ndarray,
    text: bytes,
    *,
    vocab_size: int = 8192,
    candidate_vocab_size: int = 32768,
    sentencepiece_model_path: str | None = None,
    difficulty_weight: float = 0.7,
) -> list[bytes]:
    """Build a prediction-aware vocabulary.

    Strategy:
    1. Train sentencepiece at candidate_vocab_size (large pool of candidates).
    2. Score each candidate piece by internal bigram difficulty.
    3. Keep the vocab_size pieces with lowest difficulty (easiest internal transitions).

    The result: fat tokens for easy byte sequences (model predicts them anyway),
    fine tokens for hard byte sequences (substrate gets more steps).

    Args:
        difficulty: float32 per-byte difficulty array (from byte_difficulty).
        text: raw byte stream (same text the difficulty was measured on).
        vocab_size: target vocabulary size.
        candidate_vocab_size: size of the sentencepiece candidate pool.
        sentencepiece_model_path: optional pre-trained sentencepiece model to use
            as the candidate pool instead of training a new one.
        difficulty_weight: blend between difficulty (1.0) and frequency (0.0)
            for the final scoring. 0.7 = 70% difficulty, 30% frequency.

    Returns:
        List of byte sequences (the vocabulary pieces), sorted by score.
    """
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise ImportError("sentencepiece is required for build_vocab") from exc

    bigram_diff = _bigram_difficulty(difficulty, text)

    # Load or identify candidate pieces
    if sentencepiece_model_path is not None:
        sp = spm.SentencePieceProcessor(model_file=sentencepiece_model_path)
    else:
        raise ValueError(
            "Training sentencepiece from scratch requires sentencepiece_model_path. "
            "Train a large sentencepiece model first, then pass the .model path here."
        )

    # Extract all pieces and their frequencies
    candidates: list[tuple[bytes, float, float]] = []  # (piece_bytes, freq_score, diff_score)
    sp_vocab_size = sp.vocab_size()
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        piece = sp.id_to_piece(token_id)
        # Convert piece to raw bytes (remove sentencepiece's space marker)
        piece_bytes = piece.encode("utf-8").replace(b"\xe2\x96\x81", b" ")
        if sp.is_byte(token_id):
            piece_bytes = bytes([sp.piece_to_id(piece) - 3]) if len(piece_bytes) == 1 else piece_bytes

        freq_score = sp.get_score(token_id)  # log-probability (higher = more frequent)
        diff_score = score_piece(piece_bytes, bigram_diff)
        candidates.append((piece_bytes, freq_score, diff_score))

    if not candidates:
        return []

    # Normalize scores to [0, 1] range
    freq_scores = np.array([c[1] for c in candidates], dtype=np.float32)
    diff_scores = np.array([c[2] for c in candidates], dtype=np.float32)

    # Frequency: higher is better (more common = more useful to merge)
    freq_min, freq_max = freq_scores.min(), freq_scores.max()
    freq_range = freq_max - freq_min
    freq_norm = (freq_scores - freq_min) / freq_range if freq_range > 0 else np.ones_like(freq_scores)

    # Difficulty: lower is better (easier internal transitions = safer to merge)
    diff_min, diff_max = diff_scores.min(), diff_scores.max()
    diff_range = diff_max - diff_min
    diff_norm = 1.0 - (diff_scores - diff_min) / diff_range if diff_range > 0 else np.ones_like(diff_scores)

    # Combined score: higher is better
    combined = (1.0 - difficulty_weight) * freq_norm + difficulty_weight * diff_norm

    # Sort by combined score (best first), keep top vocab_size
    # Always keep byte tokens (first 256) regardless of score
    byte_pieces = []
    merge_pieces = []
    for i, (piece_bytes, _, _) in enumerate(candidates):
        if len(piece_bytes) == 1:
            byte_pieces.append((piece_bytes, float("inf")))  # always keep
        else:
            merge_pieces.append((piece_bytes, float(combined[i])))

    merge_pieces.sort(key=lambda x: x[1], reverse=True)

    # Take all byte tokens + top merge tokens up to vocab_size
    result = [p for p, _ in byte_pieces]
    remaining = vocab_size - len(result)
    result.extend(p for p, _ in merge_pieces[:remaining])

    return result
