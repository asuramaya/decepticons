from __future__ import annotations

import unittest

import numpy as np

from decepticons.metrics import bits_per_byte_from_probabilities
from decepticons.probability_diagnostics import probability_diagnostics
from decepticons.teacher_export import (
    TeacherExportAdapter,
    TeacherExportConfig,
    TeacherExportRecord,
    TeacherExportReport,
)


class TeacherExportAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.teacher = np.asarray(
            [
                [3.0, 1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.student = np.asarray(
            [
                [1.0, 3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.targets = np.asarray([0, 1], dtype=np.int64)

    def test_record_exposes_normalized_pair_and_labels(self) -> None:
        adapter = TeacherExportAdapter(TeacherExportConfig(vocabulary_size=4, source_names=("teacher", "student")))

        record = adapter.record(self.teacher, self.student)

        self.assertIsInstance(record, TeacherExportRecord)
        self.assertEqual(record.tokens, 2)
        self.assertEqual(record.steps, 2)
        self.assertEqual(record.source_names, ("teacher", "student"))
        self.assertEqual(record.teacher_probs.shape, (2, 4))
        self.assertEqual(record.student_probs.shape, (2, 4))
        self.assertEqual(record.teacher_labels.shape, (2,))
        self.assertEqual(record.student_labels.shape, (2,))
        self.assertEqual(record.teacher_probs.dtype, np.float64)
        self.assertEqual(record.student_probs.dtype, np.float64)
        self.assertEqual(record.teacher_labels.dtype, np.int64)
        self.assertEqual(record.student_labels.dtype, np.int64)
        self.assertTrue(np.allclose(record.teacher_probs.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(record.student_probs.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(record.diagnostics.as_dict()["entropy"], probability_diagnostics(record.teacher_probs, record.student_probs).entropy))

    def test_export_produces_bits_and_agreement_metrics(self) -> None:
        adapter = TeacherExportAdapter(TeacherExportConfig(vocabulary_size=4))

        report = adapter.export(self.teacher, self.student, targets=self.targets)

        self.assertIsInstance(report, TeacherExportReport)
        self.assertAlmostEqual(
            report.teacher_bits_per_byte,
            bits_per_byte_from_probabilities(report.record.teacher_probs, self.targets),
            places=12,
        )
        self.assertAlmostEqual(
            report.student_bits_per_byte,
            bits_per_byte_from_probabilities(report.record.student_probs, self.targets),
            places=12,
        )
        self.assertAlmostEqual(
            report.mean_bits_per_byte,
            0.5 * (report.teacher_bits_per_byte + report.student_bits_per_byte),
            places=12,
        )
        self.assertAlmostEqual(report.label_flip_rate, 1.0, places=12)
        self.assertAlmostEqual(report.label_agreement_rate, 0.0, places=12)
        self.assertGreaterEqual(report.mean_entropy, 0.0)
        self.assertLessEqual(report.mean_peak, 1.0)
        self.assertGreaterEqual(report.mean_overlap, 0.0)

    def test_normalization_safe_and_shape_validation(self) -> None:
        adapter = TeacherExportAdapter()
        zero_teacher = np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        zero_student = np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64)

        record = adapter.record(zero_teacher, zero_student, source_names=("oracle", "student"))

        self.assertTrue(np.all(np.isfinite(record.diagnostics.entropy)))
        self.assertTrue(np.all(np.isfinite(record.diagnostics.peak)))
        self.assertTrue(np.all(np.isfinite(record.diagnostics.top_k_mass)))
        self.assertTrue(np.all(np.isfinite(record.diagnostics.overlap)))
        self.assertTrue(np.all(np.isfinite(record.diagnostics.shared_top_k_mass)))
        self.assertTrue(np.all(np.isfinite(record.diagnostics.top2_margin)))
        self.assertTrue(np.allclose(record.teacher_probs.sum(axis=-1), 1.0))
        self.assertTrue(np.allclose(record.student_probs.sum(axis=-1), 1.0))
        self.assertEqual(record.source_names, ("oracle", "student"))

        with self.assertRaises(ValueError):
            adapter.record(self.teacher[:, :3], self.student)
        with self.assertRaises(ValueError):
            adapter.export(self.teacher, self.student, targets=np.asarray([0], dtype=np.int64))

    def test_fit_is_a_score_alias(self) -> None:
        adapter = TeacherExportAdapter()

        fit_report = adapter.fit(self.teacher, self.student, targets=self.targets)

        self.assertIsInstance(fit_report, TeacherExportReport)
        self.assertEqual(fit_report.tokens, 2)
        self.assertEqual(fit_report.record.steps, 2)


if __name__ == "__main__":
    unittest.main()
