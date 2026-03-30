from __future__ import annotations

import unittest

from open_predictive_coder.artifacts import ArtifactMetadata, make_artifact_accounting, make_replay_span
from open_predictive_coder.artifacts_audits import audit_artifact, summarize_artifact_audits


class RuntimeArtifactAccountingTests(unittest.TestCase):
    def test_replay_spans_and_payload_accounting_round_trip(self) -> None:
        accounting = make_artifact_accounting(
            "cache_repair",
            artifact_bytes=64,
            replay_bytes=18,
            replay_spans=(
                make_replay_span(2, 8, label="repair", channel="causal"),
                make_replay_span(10, 14, label="support", channel="bridge"),
            ),
            metadata=ArtifactMetadata.from_mapping({"mode": "runtime"}),
            tokens=24,
        )
        audit = audit_artifact(
            accounting,
            side_data_count=2,
            side_data_bytes=12,
        )

        self.assertEqual(audit.replay_span_count, 2)
        self.assertEqual(audit.replay_span_length, 10)
        self.assertEqual(audit.payload_bytes, 76)
        self.assertEqual(audit.metadata.get("mode"), "runtime")
        self.assertAlmostEqual(audit.payload_coverage_ratio, 18 / 76)
        self.assertAlmostEqual(audit.side_data_ratio, 12 / 76)

    def test_summary_accumulates_runtime_records(self) -> None:
        first = audit_artifact(
            make_artifact_accounting(
                "teacher_export",
                artifact_bytes=80,
                replay_bytes=20,
                replay_spans=(make_replay_span(0, 6, label="teacher"),),
            ),
            side_data_count=1,
            side_data_bytes=16,
        )
        second = audit_artifact(
            make_artifact_accounting(
                "replay_fields",
                artifact_bytes=120,
                replay_bytes=50,
                replay_spans=(make_replay_span(8, 18, label="field"),),
            ),
            side_data_count=2,
            side_data_bytes=8,
        )

        summary = summarize_artifact_audits((first, second), metadata={"family": "runtime"})

        self.assertEqual(summary.record_count, 2)
        self.assertEqual(summary.replay_span_count, 2)
        self.assertEqual(summary.replay_span_length, 16)
        self.assertEqual(summary.side_data_count, 3)
        self.assertEqual(summary.side_data_bytes, 24)
        self.assertEqual(summary.metadata.get("family"), "runtime")
        self.assertAlmostEqual(summary.coverage_ratio, 70 / 200)
        self.assertAlmostEqual(summary.payload_coverage_ratio, 70 / (96 + 128))


if __name__ == "__main__":
    unittest.main()
