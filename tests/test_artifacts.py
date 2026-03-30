from __future__ import annotations

import unittest

from open_predictive_coder.artifacts import ArtifactAccounting, ArtifactMetadata, ReplaySpan


class ArtifactMetadataTests(unittest.TestCase):
    def test_from_mapping_round_trips_and_merges(self) -> None:
        metadata = ArtifactMetadata.from_mapping(
            {
                "dataset": "toy",
                "step": 12,
                "flags": ("causal", True, 0.5),
            }
        )

        self.assertEqual(metadata.get("dataset"), "toy")
        self.assertEqual(metadata.get("step"), 12)
        self.assertEqual(
            metadata.to_dict(),
            {
                "dataset": "toy",
                "step": 12,
                "flags": ("causal", True, 0.5),
            },
        )

        merged = metadata.merged(step=13, note="replay")
        self.assertEqual(merged.get("step"), 13)
        self.assertEqual(merged.get("note"), "replay")
        self.assertEqual(metadata.get("step"), 12)

    def test_invalid_metadata_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactMetadata(items=(("", 1),))

        with self.assertRaises(TypeError):
            ArtifactMetadata.from_mapping({"bad": object()})


class ReplaySpanTests(unittest.TestCase):
    def test_span_length_and_empty_state(self) -> None:
        span = ReplaySpan(4, 10, label="window")

        self.assertEqual(span.length, 6)
        self.assertFalse(span.is_empty)
        self.assertEqual(span.label, "window")

        empty = ReplaySpan(2, 2)
        self.assertTrue(empty.is_empty)
        self.assertEqual(empty.length, 0)

    def test_invalid_span_bounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            ReplaySpan(-1, 2)

        with self.assertRaises(ValueError):
            ReplaySpan(5, 4)


class ArtifactAccountingTests(unittest.TestCase):
    def test_accounting_summarizes_spans_and_ratios(self) -> None:
        spans = (
            ReplaySpan(0, 4, label="prefix"),
            ReplaySpan(8, 11, label="suffix"),
        )
        accounting = ArtifactAccounting(
            artifact_name="conker",
            artifact_bytes=20,
            replay_bytes=12,
            replay_spans=spans,
            metadata=ArtifactMetadata.from_mapping({"mode": "causal"}),
        )

        self.assertEqual(accounting.replay_span_count, 2)
        self.assertEqual(accounting.replay_span_length, 7)
        self.assertAlmostEqual(accounting.coverage_ratio, 0.6)
        self.assertEqual(accounting.artifact_gap_bytes, 8)
        self.assertEqual(accounting.metadata.get("mode"), "causal")

    def test_accounting_validation_rejects_bad_values(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactAccounting(artifact_name="", artifact_bytes=1, replay_bytes=1)

        with self.assertRaises(ValueError):
            ArtifactAccounting(artifact_name="x", artifact_bytes=-1, replay_bytes=1)

        with self.assertRaises(ValueError):
            ArtifactAccounting(artifact_name="x", artifact_bytes=1, replay_bytes=-1)

        with self.assertRaises(TypeError):
            ArtifactAccounting(
                artifact_name="x",
                artifact_bytes=1,
                replay_bytes=1,
                replay_spans=(object(),),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
