from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "oracle" / "bidirectional_analysis"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import BidirectionalAnalysisConfig, BidirectionalAnalysisModel  # noqa: E402

from decepticons import ByteCodec, OracleAnalysisAdapter, TrainModeConfig  # noqa: E402


class BidirectionalAnalysisExampleTests(unittest.TestCase):
    def test_analyze_uses_train_mode_checkpoints_and_route_coverage(self) -> None:
        config = BidirectionalAnalysisConfig()
        model = BidirectionalAnalysisModel(config)
        tokens = ByteCodec.encode_text("oracle analysis compares prefix and suffix state repeatedly.")
        report = model.analyze(tokens)

        self.assertIsInstance(model, OracleAnalysisAdapter)
        self.assertEqual(report.checkpoints, config.train_mode.resolve_rollout_checkpoints(len(tokens)))
        self.assertEqual(len(report.points), len(report.checkpoints))
        self.assertGreaterEqual(report.oracle_preference_rate, 0.0)
        self.assertLessEqual(report.oracle_preference_rate, 1.0)
        self.assertTrue(all(point.selected_route in {"causal", "oracle"} for point in report.points))
        self.assertTrue(any(point.slow_update_active for point in report.points))

    def test_detached_mode_still_analyzes_but_changes_state_usage(self) -> None:
        detached = BidirectionalAnalysisConfig(
            train_mode=TrainModeConfig(
                state_mode="detached",
                slow_update_stride=3,
                rollout_checkpoints=(4, 8),
            )
        )
        model = BidirectionalAnalysisModel(detached)
        tokens = ByteCodec.encode_text("oracle analysis compares forward and reverse scans.")
        report = model.analyze(tokens)

        self.assertEqual(report.checkpoints, detached.train_mode.resolve_rollout_checkpoints(len(tokens)))
        self.assertEqual(len(report.points), len(report.checkpoints))
        self.assertTrue(detached.train_mode.uses_detached_state)
        self.assertTrue(all(point.route_weights.shape[0] == 2 for point in report.points))
        self.assertTrue(all(point.selected_route in {"causal", "oracle"} for point in report.points))

    def test_short_sequence_rejected(self) -> None:
        model = BidirectionalAnalysisModel()
        with self.assertRaises(ValueError):
            model.analyze(ByteCodec.encode_text("x"))


if __name__ == "__main__":
    unittest.main()
