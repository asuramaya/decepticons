from __future__ import annotations

import unittest

from open_predictive_coder.train_modes import TrainModeConfig


class TrainModeConfigTests(unittest.TestCase):
    def test_default_config_describes_detached_state_and_final_checkpoint(self) -> None:
        config = TrainModeConfig()

        self.assertTrue(config.uses_detached_state)
        self.assertFalse(config.uses_through_state)
        self.assertFalse(config.uses_sparse_slow_updates)
        self.assertTrue(config.should_update_slow(0))
        self.assertTrue(config.should_update_slow(1))
        self.assertEqual(config.resolve_rollout_checkpoints(5), (5,))

    def test_through_state_sparse_updates_and_checkpoint_stride_are_resolved(self) -> None:
        config = TrainModeConfig(
            state_mode="through_state",
            slow_update_stride=3,
            rollout_checkpoints=(2, 4),
            rollout_checkpoint_stride=2,
        )

        self.assertFalse(config.uses_detached_state)
        self.assertTrue(config.uses_through_state)
        self.assertTrue(config.uses_sparse_slow_updates)
        self.assertEqual([config.should_update_slow(step) for step in range(6)], [False, False, True, False, False, True])
        self.assertEqual(config.resolve_rollout_checkpoints(6), (2, 4, 6))

    def test_resolve_rollout_checkpoints_deduplicates_and_keeps_final_step(self) -> None:
        config = TrainModeConfig(
            rollout_checkpoints=(1, 3, 3, 5),
            rollout_checkpoint_stride=2,
        )

        self.assertEqual(config.resolve_rollout_checkpoints(5), (1, 2, 3, 4, 5))

    def test_invalid_mode_and_stride_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            TrainModeConfig(state_mode="invalid")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            TrainModeConfig(slow_update_stride=0)

        with self.assertRaises(ValueError):
            TrainModeConfig(rollout_checkpoint_stride=0)

        with self.assertRaises(ValueError):
            TrainModeConfig(rollout_checkpoints=(0,))

        with self.assertRaises(ValueError):
            TrainModeConfig().resolve_rollout_checkpoints(0)

        with self.assertRaises(ValueError):
            TrainModeConfig(rollout_checkpoints=(6,)).resolve_rollout_checkpoints(5)


if __name__ == "__main__":
    unittest.main()
