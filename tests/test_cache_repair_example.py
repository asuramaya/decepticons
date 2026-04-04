from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_example(relative_path: str) -> str:
    script = REPO_ROOT / relative_path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result.stdout


class CacheRepairExampleTests(unittest.TestCase):
    def test_cache_repair_uses_shared_caches(self) -> None:
        module = load_module(
            "cache_repair_model_test",
            "examples/projects/causal/cache_repair/model.py",
        )
        model = module.CacheRepairModel.build()
        corpus = (
            "cache repair prefers exact context when support is strong.\n"
            "cache repair still needs a broad prior when support collapses.\n"
        ) * 4

        fit = model.fit(corpus)
        trace = model.trace(corpus[:192])
        score = model.score(corpus[:192])
        prompt = model.predict_proba(corpus[:64])

        self.assertGreater(fit.backoff.ngram.tokens, 0)
        self.assertGreater(fit.exact.tokens, 0)
        self.assertEqual(len(fit.feature_names), fit.gate_weights.shape[0])
        self.assertGreaterEqual(fit.exact_win_rate, 0.0)
        self.assertLessEqual(fit.exact_win_rate, 1.0)
        self.assertEqual(trace.prior_probs.shape, trace.exact_probs.shape)
        self.assertEqual(trace.mixed_probs.shape, trace.prior_probs.shape)
        self.assertEqual(trace.feature_matrix.shape[1], len(trace.feature_names))
        self.assertEqual(prompt.shape, (256,))
        self.assertTrue(np.allclose(trace.mixed_probs.sum(axis=1), 1.0))
        self.assertTrue(np.all((trace.repair_strength >= 0.0) & (trace.repair_strength <= 1.0)))
        self.assertGreaterEqual(score.mean_repair_strength, 0.0)
        self.assertLessEqual(score.mean_repair_strength, 1.0)
        self.assertGreaterEqual(score.mean_exact_support, 0.0)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/causal/cache_repair/probe.py")
        smoke_output = run_example("examples/projects/causal/cache_repair/smoke.py")

        self.assertIn("project: cache_repair", probe_output)
        self.assertIn("mixed bits/byte:", probe_output)
        self.assertIn("repair strength:", smoke_output)
        self.assertIn("exact support:", smoke_output)


if __name__ == "__main__":
    unittest.main()
