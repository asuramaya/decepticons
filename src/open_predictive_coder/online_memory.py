"""Online causal memory: accumulates n-gram statistics during sequential processing.

This is a runtime data structure, not a learned parameter. It builds up
token count statistics as it processes a sequence, and provides empirical
distributions for prediction. Strictly causal: only uses tokens already seen.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class OnlineMemoryConfig:
    """Configuration for online causal memory."""
    max_order: int = 3          # maximum n-gram order (2=bigram, 3=trigram, etc.)
    bucket_count: int = 8192    # hash table size for orders >= 3
    vocabulary_size: int = 256

    def __post_init__(self) -> None:
        if self.max_order < 2:
            raise ValueError("online memory max_order must be >= 2")
        if self.bucket_count < 1:
            raise ValueError("online memory bucket_count must be positive")
        if self.vocabulary_size < 1:
            raise ValueError("online memory vocabulary_size must be positive")


ONLINE_MEMORY_HASH_MULTIPLIERS = (2654435761, 2246822519, 3266489917, 2028178513)


class OnlineCausalMemory:
    """Accumulates n-gram counts during sequential token processing.

    Usage:
        mem = OnlineCausalMemory(config)
        for token in sequence:
            features = mem.query_features()  # get current prediction features
            mem.update(token)                # add token to history
        mem.reset()                          # clear for next sequence
    """

    def __init__(self, config: OnlineMemoryConfig) -> None:
        self.config = config
        self.vocab = config.vocabulary_size
        self.max_order = config.max_order
        self.buckets = config.bucket_count

        # Unigram counts: [vocab]
        self.unigram = np.zeros(self.vocab, dtype=np.float32)
        # Bigram counts: [vocab, vocab] (prev -> current)
        self.bigram = np.zeros((self.vocab, self.vocab), dtype=np.float32)
        # Higher-order: hashed buckets [buckets, vocab]
        self.higher = np.zeros((self.buckets, self.vocab), dtype=np.float32)

        # Context ring buffer
        self._context = np.zeros(self.max_order, dtype=np.int64)
        self._pos = 0  # total tokens seen
        self._total = 0.0

    def reset(self) -> None:
        """Clear all counts for the next sequence."""
        self.unigram[:] = 0
        self.bigram[:] = 0
        self.higher[:] = 0
        self._context[:] = 0
        self._pos = 0
        self._total = 0.0

    def update(self, token_id: int) -> None:
        """Add a token to the memory. Call AFTER using query_features for this position."""
        self.unigram[token_id] += 1.0
        self._total += 1.0

        if self._pos >= 1:
            prev = int(self._context[(self._pos - 1) % self.max_order])
            self.bigram[prev, token_id] += 1.0

        if self._pos >= 2:
            # Hash the context for trigram and higher
            for order in range(3, min(self.max_order + 1, self._pos + 2)):
                h = 0
                for i in range(order - 1):
                    ctx_idx = (self._pos - (order - 1) + i) % self.max_order
                    h = h ^ (int(self._context[ctx_idx]) * ONLINE_MEMORY_HASH_MULTIPLIERS[i % len(ONLINE_MEMORY_HASH_MULTIPLIERS)])
                bucket = h % self.buckets
                self.higher[bucket, token_id] += 1.0

        self._context[self._pos % self.max_order] = token_id
        self._pos += 1

    def query_features(self) -> np.ndarray:
        """Return memory prediction features for the current position.

        Returns a feature vector of shape [num_features] where:
        - features[0] = unigram entropy
        - features[1] = unigram peak probability
        - features[2] = bigram entropy (conditioned on prev token)
        - features[3] = bigram peak probability
        - features[4] = trigram peak probability (from hash table)
        - features[5] = total tokens seen (log scale)
        - features[6] = bigram count for prev token (log scale, confidence signal)
        """
        features = np.zeros(7, dtype=np.float32)

        # Feature 5: position info
        features[5] = np.log1p(self._total)

        if self._total < 1:
            return features

        # Unigram features
        p_uni = self.unigram / self._total
        p_uni = np.clip(p_uni, 1e-10, None)
        features[0] = -np.sum(p_uni * np.log(p_uni))  # entropy
        features[1] = np.max(p_uni)  # peak

        if self._pos >= 1:
            prev = int(self._context[(self._pos - 1) % self.max_order])
            bigram_row = self.bigram[prev]
            row_total = bigram_row.sum()
            if row_total > 0:
                p_bi = bigram_row / row_total
                p_bi = np.clip(p_bi, 1e-10, None)
                features[2] = -np.sum(p_bi * np.log(p_bi))
                features[3] = np.max(p_bi)
                features[6] = np.log1p(row_total)

        if self._pos >= 2:
            h = 0
            for i in range(min(2, self._pos)):
                ctx_idx = (self._pos - 2 + i) % self.max_order
                h = h ^ (int(self._context[ctx_idx]) * ONLINE_MEMORY_HASH_MULTIPLIERS[i % len(ONLINE_MEMORY_HASH_MULTIPLIERS)])
            bucket = h % self.buckets
            tri_row = self.higher[bucket]
            tri_total = tri_row.sum()
            if tri_total > 0:
                features[4] = np.max(tri_row / tri_total)

        return features
