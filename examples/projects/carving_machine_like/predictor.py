from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class HierarchicalPredictorConfig:
    controller_view_dim: int = 32
    controller_width: int = 64
    prediction_l2: float = 1e-3
    seed: int = 7


class RidgeRegressor:
    def __init__(self, *, input_dim: int, output_dim: int, alpha: float = 1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.weights: np.ndarray | None = None

    @staticmethod
    def _augment(features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        bias = np.ones((features.shape[0], 1), dtype=np.float64)
        return np.concatenate([features, bias], axis=1)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = self._augment(features)
        y = np.asarray(targets, dtype=np.float64)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and targets must have the same number of rows")
        regularizer = np.eye(x.shape[1], dtype=np.float64) * self.alpha
        regularizer[-1, -1] = 0.0
        gram = x.T @ x + regularizer
        rhs = x.T @ y
        self.weights = np.linalg.solve(gram, rhs)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros((np.asarray(features).shape[0], self.output_dim), dtype=np.float64)
        return self._augment(features) @ self.weights


class HierarchicalPredictor:
    def __init__(
        self,
        *,
        fast_size: int,
        mid_size: int,
        slow_size: int,
        config: HierarchicalPredictorConfig | None = None,
    ):
        self.config = config or HierarchicalPredictorConfig()
        self.fast_size = fast_size
        self.mid_size = mid_size
        self.slow_size = slow_size
        self.rng = np.random.default_rng(self.config.seed)

        self.slow_projection = self._make_projection(slow_size, self.config.controller_view_dim)
        self.mid_projection = self._make_projection(mid_size, self.config.controller_view_dim)
        hidden_in = self.config.controller_view_dim * 2
        self.w1 = self._make_projection(hidden_in, self.config.controller_width)
        self.w2 = self._make_projection(self.config.controller_width, self.config.controller_width)
        self.readout = RidgeRegressor(
            input_dim=self.config.controller_width,
            output_dim=fast_size,
            alpha=self.config.prediction_l2,
        )

    def _make_projection(self, rows: int, cols: int) -> np.ndarray:
        return self.rng.standard_normal((rows, cols)).astype(np.float64) / np.sqrt(max(rows, 1))

    def _encode(self, slow: np.ndarray, mid: np.ndarray) -> np.ndarray:
        slow_view = slow @ self.slow_projection
        mid_view = mid @ self.mid_projection
        hidden = np.concatenate([slow_view, mid_view], axis=-1)
        hidden = _gelu(hidden @ self.w1)
        hidden = _gelu(hidden @ self.w2)
        return hidden

    def fit(self, slow: np.ndarray, mid: np.ndarray, fast: np.ndarray) -> None:
        hidden = self._encode(slow, mid)
        self.readout.fit(hidden, np.asarray(fast, dtype=np.float64))

    def predict(self, slow: np.ndarray, mid: np.ndarray) -> np.ndarray:
        hidden = self._encode(slow, mid)
        return _sigmoid(self.readout.predict(hidden))


__all__ = [
    "HierarchicalPredictor",
    "HierarchicalPredictorConfig",
]
