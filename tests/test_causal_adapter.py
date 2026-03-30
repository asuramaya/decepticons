from __future__ import annotations

import importlib
import unittest

import numpy as np

from open_predictive_coder.artifacts import ArtifactAccounting
from open_predictive_coder.exact_context import ExactContextMemory, SupportWeightedMixer


MODULE_NAME = "open_predictive_coder.causal_predictive"
CLASS_NAMES = ("CausalPredictiveAdapter", "CausalPredictiveModel")


def load_causal_module():
    return importlib.import_module(MODULE_NAME)


def load_causal_class():
    module = load_causal_module()
    for name in CLASS_NAMES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"{MODULE_NAME} does not expose any of {CLASS_NAMES}")


class CausalAdapterSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.adapter_cls = load_causal_class()
        except (ModuleNotFoundError, AttributeError):
            self.skipTest(
                "planned causal adapter is not present yet; expected one of "
                f"{CLASS_NAMES} in {MODULE_NAME}"
            )

    def _make_adapter(self) -> object:
        causal_cls = self.adapter_cls
        try:
            return causal_cls()
        except TypeError:
            try:
                return causal_cls(vocabulary_size=256)
            except TypeError as exc:  # pragma: no cover - implementation-specific
                self.skipTest(f"causal adapter constructor signature is not yet settled: {exc}")

    def test_adapter_surface_exposes_causal_methods(self) -> None:
        adapter = self._make_adapter()
        for name in ("fit", "predict_proba", "score"):
            self.assertTrue(hasattr(adapter, name), f"missing expected method {name}")

    def test_fit_predicts_and_scores_causally(self) -> None:
        adapter = self._make_adapter()
        corpus = ("abababababab", "abcabcabcabc")

        fit_report = adapter.fit(corpus)
        probabilities = adapter.predict_proba("ab")
        score = adapter.score("abab")

        self.assertIsNotNone(fit_report)
        self.assertEqual(probabilities.ndim, 1)
        self.assertGreater(probabilities.size, 1)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=6)
        self.assertGreater(score.tokens, 0)
        self.assertGreater(score.bits_per_byte, 0.0)

    def test_adapter_reports_artifact_accounting_or_adjacent_runtime(self) -> None:
        adapter = self._make_adapter()
        adapter.fit(("abababab", "abcabc"))

        accounting = None
        for name in ("artifact_accounting", "artifact_accounting_for", "accounting"):
            if hasattr(adapter, name):
                value = getattr(adapter, name)
                accounting = value() if callable(value) else value
                break

        if accounting is None:
            self.skipTest("causal adapter does not expose artifact accounting yet")

        self.assertIsInstance(accounting, ArtifactAccounting)
        self.assertGreaterEqual(accounting.artifact_bytes, 0)
        self.assertGreaterEqual(accounting.replay_bytes, 0)


class CausalKernelSplitTests(unittest.TestCase):
    def test_kernel_exports_causal_primitives_but_not_conker_policy(self) -> None:
        import open_predictive_coder as opc

        self.assertTrue(hasattr(opc, "ExactContextMemory"))
        self.assertTrue(hasattr(opc, "SupportWeightedMixer"))
        self.assertTrue(hasattr(opc, "ArtifactAccounting"))

        self.assertFalse(hasattr(opc, "ConkerMixerPolicy"))
        self.assertFalse(hasattr(opc, "ConkerResidualPolicy"))
        self.assertFalse(hasattr(opc, "ConkerCausalAdapter"))

    def test_exact_context_and_mixer_remain_project_independent(self) -> None:
        memory = ExactContextMemory()
        mixer = SupportWeightedMixer()
        self.assertTrue(hasattr(memory, "fit"))
        self.assertTrue(hasattr(memory, "predictive_distribution"))
        self.assertTrue(hasattr(mixer, "mix"))


if __name__ == "__main__":
    unittest.main()
