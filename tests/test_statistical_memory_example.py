from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "causal" / "statistical_memory"


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


class StatisticalMemoryExampleTests(unittest.TestCase):
    def test_trace_and_score_use_both_memories(self) -> None:
        module = load_module(
            "statistical_memory_model_test",
            "examples/projects/causal/statistical_memory/model.py",
        )
        model = module.StatisticalMemoryModel.build()
        corpus = (
            "statistical memory combines exact repair with n-gram smoothing.\n"
            "statistical memory combines exact repair with n-gram smoothing.\n"
            "a local variation appears after the repeated fragment.\n"
        ) * 3

        fit = model.fit(corpus)
        trace = model.trace(corpus[:192])
        score = model.score(corpus[:192])
        prompt = model.predict_proba(corpus[:48])

        self.assertGreater(fit.ngram.tokens, 0)
        self.assertGreater(fit.exact.tokens, 0)
        self.assertGreater(sum(fit.exact.contexts_by_order), 0)
        self.assertEqual(trace.base_probs.shape, trace.exact_probs.shape)
        self.assertEqual(trace.mixed_probs.shape, trace.base_probs.shape)
        self.assertEqual(prompt.shape, (256,))
        self.assertTrue(np.all(np.isfinite(trace.mixed_probs)))
        self.assertTrue(any(any(name.startswith("exact") for name in names) for names in trace.component_names))
        self.assertGreaterEqual(score.ngram_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.exact_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.mixed_bits_per_byte, 0.0)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/causal/statistical_memory/probe.py")
        smoke_output = run_example("examples/projects/causal/statistical_memory/smoke.py")

        self.assertIn("project: statistical_memory", probe_output)
        self.assertIn("mixed bits/byte:", probe_output)
        self.assertIn("ngram train tokens:", smoke_output)
        self.assertIn("exact bits/byte:", smoke_output)


if __name__ == "__main__":
    unittest.main()
