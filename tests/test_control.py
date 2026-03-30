from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.control import (
    ControllerSummaryBuilder,
    ControllerSummaryConfig,
    stack_summaries,
)


class ControlTests(unittest.TestCase):
    def test_summary_builder_mean_abs_reduces_sequence(self) -> None:
        builder = ControllerSummaryBuilder(ControllerSummaryConfig(reduction="mean_abs"))
        signal = np.asarray([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float64)

        summary = builder.encode(signal, name="surprise")

        self.assertEqual(summary.name, "surprise")
        self.assertTrue(np.allclose(summary.values, np.asarray([2.0, 3.0])))

    def test_summary_builder_normalizes_when_requested(self) -> None:
        builder = ControllerSummaryBuilder(ControllerSummaryConfig(reduction="identity", normalize=True))

        summary = builder.encode(np.asarray([3.0, 4.0]))

        self.assertTrue(np.allclose(summary.values, np.asarray([0.6, 0.8])))

    def test_stack_summaries_requires_matching_dim(self) -> None:
        builder = ControllerSummaryBuilder()
        left = builder.encode(np.asarray([1.0, 2.0]))
        right = builder.encode(np.asarray([3.0, 4.0]))

        stacked = stack_summaries([left, right])

        self.assertEqual(stacked.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
