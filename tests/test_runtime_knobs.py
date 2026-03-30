from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PROJECTS_ROOT = REPO_ROOT / "examples" / "projects"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from open_predictive_coder import HierarchicalSubstrate, TrainModeConfig, hierarchical_small  # noqa: E402
from conker_shared import build_hierarchical_stability_expert  # noqa: E402


class TrainModeIntegrationTests(unittest.TestCase):
    def test_ancestor_style_hierarchical_surface_matches_detached_stride_knob(self) -> None:
        config = hierarchical_small()
        substrate = HierarchicalSubstrate(config.hierarchical)
        train_mode = TrainModeConfig(
            state_mode="detached",
            slow_update_stride=config.hierarchical.slow_update_stride,
            rollout_checkpoints=(2, 4),
            rollout_checkpoint_stride=2,
        )

        self.assertTrue(train_mode.uses_detached_state)
        self.assertFalse(train_mode.uses_through_state)
        self.assertTrue(train_mode.uses_sparse_slow_updates)
        self.assertEqual(config.hierarchical.slow_update_stride, 2)
        self.assertEqual([train_mode.should_update_slow(step) for step in range(4)], [False, True, False, True])
        self.assertEqual(train_mode.resolve_rollout_checkpoints(4), (2, 4))

        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        state = substrate.initial_state()
        checkpoint_states: dict[int, np.ndarray] = {}
        for step_index, token in enumerate(tokens, start=1):
            state = substrate.step(state, int(token))
            if step_index in train_mode.resolve_rollout_checkpoints(tokens.size):
                checkpoint_states[step_index] = state.copy()

        self.assertEqual(tuple(checkpoint_states), (2, 4))
        self.assertEqual(checkpoint_states[2].shape[0], substrate.state_dim)

    def test_conker_style_hierarchical_expert_matches_through_state_stride_knob(self) -> None:
        expert = build_hierarchical_stability_expert(name="stability")
        train_mode = TrainModeConfig(
            state_mode="through_state",
            slow_update_stride=expert.substrate.config.slow_update_stride,
            rollout_checkpoints=(1, 3),
            rollout_checkpoint_stride=2,
        )

        self.assertTrue(train_mode.uses_through_state)
        self.assertFalse(train_mode.uses_detached_state)
        self.assertTrue(train_mode.uses_sparse_slow_updates)
        self.assertEqual(expert.substrate.config.slow_update_stride, 2)
        self.assertEqual([train_mode.should_update_slow(step) for step in range(4)], [False, True, False, True])
        self.assertEqual(train_mode.resolve_rollout_checkpoints(4), (1, 2, 3, 4))

        tokens = np.array([5, 6, 7, 8], dtype=np.int64)
        fit_report = expert.fit(tokens)
        self.assertGreater(fit_report.bits_per_byte, 0.0)
        probabilities, targets = expert.sequence_probabilities(tokens)
        self.assertEqual(probabilities.shape[0], targets.shape[0])
        self.assertEqual(probabilities.shape[1], expert.vocabulary_size)

    def test_checkpoint_resolution_can_drive_example_trace_selection(self) -> None:
        train_mode = TrainModeConfig(
            state_mode="detached",
            slow_update_stride=3,
            rollout_checkpoints=(2, 4),
            rollout_checkpoint_stride=2,
        )

        checkpoints = train_mode.resolve_rollout_checkpoints(5)
        self.assertEqual(checkpoints, (2, 4, 5))

        config = hierarchical_small()
        substrate = HierarchicalSubstrate(config.hierarchical)
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        state = substrate.initial_state()
        captured_steps: list[int] = []
        for step_index, token in enumerate(tokens, start=1):
            state = substrate.step(state, int(token))
            if step_index in checkpoints:
                captured_steps.append(step_index)

        self.assertEqual(captured_steps, list(checkpoints))
        self.assertEqual(state.shape[0], substrate.state_dim)


if __name__ == "__main__":
    unittest.main()
