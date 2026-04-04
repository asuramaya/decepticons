from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.tools.diagnostics import (  # noqa: E402
    SnapshotSeries,
    capture_snapshot,
    compare_ablation,
    compare_ablation_map,
    decompose_two_factor,
    format_alignment_summary,
    format_binary_mask_summary,
    format_snapshot_record,
    summarize_alignment,
    summarize_binary_mask,
    summarize_snapshot_series,
)


class DiagnosticsToolTests(unittest.TestCase):
    def test_binary_mask_summary_matches_basic_counts(self) -> None:
        mask = np.array([[0.0, 0.2, 0.7], [0.9, 1.0, 0.4]], dtype=np.float64)
        summary = summarize_binary_mask(mask, name="mask")

        self.assertEqual(summary.name, "mask")
        self.assertEqual(summary.shape, (2, 3))
        self.assertAlmostEqual(summary.mean, mask.mean())
        self.assertAlmostEqual(summary.gt05, 3 / 6)
        self.assertAlmostEqual(summary.lt01, 1 / 6)
        self.assertEqual(summary.active_dims, 1)
        self.assertEqual(summary.strong_active_dims, 0)
        self.assertIn("mask:", format_binary_mask_summary(summary))

    def test_alignment_summary_reports_pairwise_relationship(self) -> None:
        source = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        target = np.array([1.0, 2.0, 5.0], dtype=np.float64)
        summary = summarize_alignment(source, target, source_name="mask", target_name="gate")

        self.assertEqual(summary.source_name, "mask")
        self.assertEqual(summary.target_name, "gate")
        self.assertEqual(summary.shape, (3,))
        self.assertGreater(summary.pearson, 0.0)
        self.assertGreater(summary.cosine, 0.0)
        self.assertAlmostEqual(summary.mae, np.mean(np.abs(source - target)))
        self.assertIn("mask->gate", format_alignment_summary(summary))

    def test_snapshot_series_captures_named_signals(self) -> None:
        record_1 = capture_snapshot(
            10,
            mask=np.array([0.1, 0.2, 0.3]),
            gate=np.array([0.8, 0.7, 0.6]),
        )
        record_2 = capture_snapshot(
            20,
            mask=np.array([0.2, 0.3, 0.4]),
            gate=np.array([0.7, 0.6, 0.5]),
        )
        series = SnapshotSeries((record_1, record_2))
        summary = summarize_snapshot_series(series, signal_name="mask")

        self.assertEqual(record_1.signal_names(), ("mask", "gate"))
        self.assertEqual(series.signal_names(), ("mask", "gate"))
        self.assertEqual(summary["signal"], "mask")
        self.assertAlmostEqual(summary["delta_mean"], record_2.get("mask").mean - record_1.get("mask").mean)
        self.assertIn("step=10", format_snapshot_record(record_1))

    def test_ablation_helpers_compute_delta_and_interaction(self) -> None:
        comparison = compare_ablation("baseline", 1.0, "no_mask", 1.2)
        self.assertEqual(comparison.baseline_name, "baseline")
        self.assertEqual(comparison.variant_name, "no_mask")
        self.assertAlmostEqual(comparison.delta, 0.2)
        self.assertAlmostEqual(comparison.relative_change, 20.0)

        comparisons = compare_ablation_map("baseline", 1.0, {"no_mask": 1.2, "no_suppress": 0.9})
        self.assertEqual(len(comparisons), 2)

        decomposition = decompose_two_factor(
            "baseline",
            10.0,
            "remove_mask",
            11.0,
            "remove_suppress",
            12.0,
            "remove_both",
            14.0,
        )
        self.assertAlmostEqual(decomposition.first_effect, 1.0)
        self.assertAlmostEqual(decomposition.second_effect, 2.0)
        self.assertAlmostEqual(decomposition.interaction, 1.0)


if __name__ == "__main__":
    unittest.main()
