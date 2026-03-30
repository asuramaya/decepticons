from __future__ import annotations

import unittest

from examples.tools.diagnostics import (
    diagnose_carving_machine_like,
    diagnose_conker_1_like,
    diagnose_conker_2_like,
    diagnose_conker_3_like,
    diagnose_conker_early_like,
    format_example_diagnostics,
)


class DiagnosticsExampleTests(unittest.TestCase):
    def test_carving_machine_like_report_includes_trace_and_series(self) -> None:
        report = diagnose_carving_machine_like(
            corpus=(
                "carving machine keeps fast, mid, and slow state aligned.\n"
                "the sampled readout should stay text-first and diagnostic.\n"
            )
            * 2
        )

        self.assertEqual(report.project, "carving_machine_like")
        self.assertIsNotNone(report.snapshot)
        self.assertIsNotNone(report.series)
        self.assertIn("features", report.snapshot.signal_names())
        self.assertIn("gates", report.snapshot.signal_names())
        self.assertIn("fit vs score", format_example_diagnostics(report))
        self.assertIn("hierarchical_trace", format_example_diagnostics(report))

    def test_conker_early_like_report_includes_ablation_summary(self) -> None:
        report = diagnose_conker_early_like(
            corpus=(
                "exact history should help when the local suffix is stable.\n"
                "support should matter more than lore and less than evidence.\n"
            )
            * 2
        )

        self.assertEqual(report.project, "conker_early_like")
        self.assertTrue(report.ablations)
        self.assertIn("base vs mixed", format_example_diagnostics(report))
        self.assertIn("exact_context_repair", format_example_diagnostics(report))

    def test_conker_replica_helpers_are_supported(self) -> None:
        conker_1 = diagnose_conker_1_like(
            corpus=(
                "memory should beat stability when the suffix is narrow.\n"
                "stability should win when the substrate is already clean.\n"
            )
            * 2
        )
        conker_2 = diagnose_conker_2_like(
            corpus=(
                "linear memory carries the main path while local correction stays smaller.\n"
                "the correction expert should only matter when the base path misses detail.\n"
            )
            * 2
        )
        conker_3 = diagnose_conker_3_like(
            corpus=(
                "local residual repair should stay narrow and selective.\n"
                "the base path should remain responsible for most of the distribution.\n"
            )
            * 2
        )

        self.assertEqual(conker_1.project, "conker_1_like")
        self.assertEqual(conker_2.project, "conker_2_like")
        self.assertEqual(conker_3.project, "conker_3_like")
        self.assertIn("conker_replica", format_example_diagnostics(conker_1))
        self.assertIn("conker_replica", format_example_diagnostics(conker_2))
        self.assertIn("conker_replica", format_example_diagnostics(conker_3))
        self.assertTrue(conker_1.ablations)
        self.assertTrue(conker_2.ablations)
        self.assertTrue(conker_3.ablations)


if __name__ == "__main__":
    unittest.main()
