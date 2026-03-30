from __future__ import annotations

import unittest

from open_predictive_coder.artifacts import make_artifact_accounting, make_replay_span
from open_predictive_coder.artifacts_audits import audit_artifact, summarize_artifact_audits


class ArtifactAuditTests(unittest.TestCase):
    def test_audit_record_derives_ratios_and_metadata(self) -> None:
        accounting = make_artifact_accounting(
            "teacher_export",
            artifact_bytes=120,
            replay_bytes=45,
            replay_spans=(
                make_replay_span(2, 12, label="clean"),
                make_replay_span(20, 35, label="repair"),
            ),
        )

        record = audit_artifact(
            accounting,
            side_data_count=3,
            side_data_bytes=30,
            metadata={"channel": "bridge", "mode": "audit"},
        )

        self.assertEqual(record.artifact_name, "teacher_export")
        self.assertEqual(record.replay_span_count, 2)
        self.assertEqual(record.replay_span_length, 25)
        self.assertEqual(record.payload_bytes, 150)
        self.assertAlmostEqual(record.coverage_ratio, 45 / 120)
        self.assertAlmostEqual(record.payload_coverage_ratio, 45 / 150)
        self.assertAlmostEqual(record.side_data_ratio, 30 / 150)
        self.assertEqual(record.metadata.get("channel"), "bridge")
        self.assertEqual(record.metadata.get("mode"), "audit")

    def test_summary_aggregates_records(self) -> None:
        first = audit_artifact(
            make_artifact_accounting(
                "noncausal_export",
                artifact_bytes=80,
                replay_bytes=20,
                replay_spans=(make_replay_span(0, 5, label="a"),),
            ),
            side_data_count=1,
            side_data_bytes=10,
            payload_bytes=100,
        )
        second = audit_artifact(
            make_artifact_accounting(
                "causal_export",
                artifact_bytes=40,
                replay_bytes=10,
                replay_spans=(make_replay_span(5, 9, label="b"), make_replay_span(15, 18, label="c")),
            ),
            side_data_count=2,
            side_data_bytes=4,
            payload_bytes=48,
        )

        summary = summarize_artifact_audits((first, second), metadata={"family": "frontier"})

        self.assertEqual(summary.record_count, 2)
        self.assertEqual(summary.artifact_bytes, 120)
        self.assertEqual(summary.replay_bytes, 30)
        self.assertEqual(summary.payload_bytes, 148)
        self.assertEqual(summary.side_data_bytes, 14)
        self.assertEqual(summary.side_data_count, 3)
        self.assertEqual(summary.replay_span_count, 3)
        self.assertEqual(summary.replay_span_length, 12)
        self.assertAlmostEqual(summary.coverage_ratio, 30 / 120)
        self.assertAlmostEqual(summary.payload_coverage_ratio, 30 / 148)
        self.assertAlmostEqual(summary.side_data_ratio, 14 / 148)
        self.assertEqual(summary.metadata.get("family"), "frontier")

    def test_validation_rejects_negative_sizes(self) -> None:
        accounting = make_artifact_accounting("payload", artifact_bytes=10, replay_bytes=2)

        with self.assertRaises(ValueError):
            audit_artifact(accounting, side_data_count=-1)

        with self.assertRaises(ValueError):
            audit_artifact(accounting, side_data_bytes=-1)

        with self.assertRaises(ValueError):
            audit_artifact(accounting, payload_bytes=-1)

    def test_zero_payload_ratios_clamp_to_zero(self) -> None:
        record = audit_artifact(
            make_artifact_accounting("empty", artifact_bytes=0, replay_bytes=0),
            side_data_bytes=0,
            payload_bytes=0,
        )

        self.assertEqual(record.coverage_ratio, 0.0)
        self.assertEqual(record.payload_coverage_ratio, 0.0)
        self.assertEqual(record.side_data_ratio, 0.0)

    def test_summary_gap_and_empty_summary_are_well_defined(self) -> None:
        summary = summarize_artifact_audits(())
        self.assertEqual(summary.record_count, 0)
        self.assertEqual(summary.artifact_gap_bytes, 0)
        self.assertEqual(summary.coverage_ratio, 0.0)
        self.assertEqual(summary.payload_coverage_ratio, 0.0)

        first = audit_artifact(make_artifact_accounting("a", artifact_bytes=10, replay_bytes=9))
        second = audit_artifact(make_artifact_accounting("b", artifact_bytes=15, replay_bytes=7))
        combined = summarize_artifact_audits((first, second))
        self.assertEqual(combined.artifact_gap_bytes, 9)


if __name__ == "__main__":
    unittest.main()
