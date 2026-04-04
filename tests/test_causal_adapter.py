from __future__ import annotations

import importlib
import unittest

import numpy as np

from decepticons.artifacts import ArtifactAccounting
from decepticons.exact_context import (
    ExactContextConfig,
    ExactContextMemory,
    SupportMixConfig,
    SupportWeightedMixer,
)
from decepticons.ngram_memory import NgramMemory, NgramMemoryConfig

MODULE_NAME = "decepticons.causal_predictive"
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

    def _make_adapter(self, **kwargs) -> object:
        causal_cls = self.adapter_cls
        try:
            return causal_cls(**kwargs)
        except TypeError:
            if kwargs:
                raise
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

    def test_disabled_ngram_path_preserves_backward_compatibility(self) -> None:
        corpus = ("abababababab", "abcabcabcabc")
        baseline = self._make_adapter()
        explicit_disabled = self._make_adapter(ngram_memory=None)

        baseline.fit(corpus)
        explicit_disabled.fit(corpus)

        baseline_probs = baseline.predict_proba("ab")
        disabled_probs = explicit_disabled.predict_proba("ab")
        baseline_score = baseline.score("abcabc")
        disabled_score = explicit_disabled.score("abcabc")

        np.testing.assert_allclose(baseline_probs, disabled_probs)
        self.assertAlmostEqual(baseline_score.bits_per_byte, disabled_score.bits_per_byte, places=12)
        self.assertAlmostEqual(
            baseline_score.exact_bits_per_byte,
            disabled_score.exact_bits_per_byte,
            places=12,
        )
        self.assertIsNone(baseline_score.ngram_bits_per_byte)
        self.assertIsNone(disabled_score.ngram_bits_per_byte)

    def test_ngram_path_changes_blended_probabilities_in_controlled_case(self) -> None:
        exact_config = ExactContextConfig(max_order=1, alpha=0.0)
        ngram_config = NgramMemoryConfig(vocabulary_size=256, bigram_alpha=0.0, trigram_alpha=0.0)
        mixer = SupportWeightedMixer(SupportMixConfig(base_bias=0.0, expert_bias=0.0, support_scale=0.0))
        corpus = ("abcabcabcabc", "ebdebdebde")

        baseline = self._make_adapter(
            exact_context=ExactContextMemory(exact_config),
            mixer=mixer,
        )
        with_ngram = self._make_adapter(
            exact_context=ExactContextMemory(exact_config),
            ngram_memory=NgramMemory(ngram_config),
            mixer=mixer,
        )

        baseline_fit = baseline.fit(corpus)
        ngram_fit = with_ngram.fit(corpus)

        baseline_probs = baseline.predict_proba("ab")
        ngram_probs = with_ngram.predict_proba("ab")
        baseline_score = baseline.score("abcabc")
        ngram_score = with_ngram.score("abcabc")

        self.assertIsNone(baseline_fit.ngram_fit)
        self.assertIsNotNone(ngram_fit.ngram_fit)
        self.assertGreater(ngram_probs[ord("c")], baseline_probs[ord("c")])
        self.assertFalse(np.allclose(baseline_probs, ngram_probs))
        self.assertLess(ngram_score.bits_per_byte, baseline_score.bits_per_byte)
        self.assertIsNotNone(ngram_score.ngram_bits_per_byte)


class CausalKernelSplitTests(unittest.TestCase):
    def test_kernel_exports_causal_primitives_but_not_descendant_policy(self) -> None:
        import decepticons as opc

        self.assertTrue(hasattr(opc, "ExactContextMemory"))
        self.assertTrue(hasattr(opc, "SupportWeightedMixer"))
        self.assertTrue(hasattr(opc, "ArtifactAccounting"))

        self.assertFalse(hasattr(opc, "MemoryStabilityModel"))
        self.assertFalse(hasattr(opc, "ResidualRepairModel"))
        self.assertFalse(hasattr(opc, "CausalVariantPolicy"))

    def test_exact_context_and_mixer_remain_project_independent(self) -> None:
        memory = ExactContextMemory()
        mixer = SupportWeightedMixer()
        self.assertTrue(hasattr(memory, "fit"))
        self.assertTrue(hasattr(memory, "predictive_distribution"))
        self.assertTrue(hasattr(mixer, "mix"))


if __name__ == "__main__":
    unittest.main()
