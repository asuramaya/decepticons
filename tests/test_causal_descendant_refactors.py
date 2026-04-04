from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

from decepticons import (
    ExactContextMemory,
    LinearMemorySubstrate,
    NgramMemory,
    OscillatoryMemorySubstrate,
    SupportWeightedMixer,
)

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


class CausalDescendantRefactorTests(unittest.TestCase):
    def test_statistical_memory_is_just_kernel_composition(self) -> None:
        module = load_module(
            "statistical_memory_model_refactor_test",
            "examples/projects/causal/statistical_memory/model.py",
        )
        model = module.StatisticalMemoryModel.build()
        corpus = (
            "statistical memory combines exact repair with n-gram smoothing.\n"
            "statistical memory combines exact repair with n-gram smoothing.\n"
            "a local variation appears after the repeated fragment.\n"
        ) * 2

        fit = model.fit(corpus)
        trace = model.trace(corpus[:160])
        score = model.score(corpus[:160])

        self.assertIsInstance(model.exact_memory, ExactContextMemory)
        self.assertIsInstance(model.ngram_memory, NgramMemory)
        self.assertIsInstance(model.mixer, SupportWeightedMixer)
        self.assertGreater(fit.ngram.tokens, 0)
        self.assertGreater(fit.exact.tokens, 0)
        self.assertEqual(trace.base_probs.shape, trace.exact_probs.shape)
        self.assertEqual(trace.mixed_probs.shape, trace.base_probs.shape)
        self.assertTrue(any(name.startswith("exact") for names in trace.component_names for name in names))
        self.assertFalse(hasattr(trace, "component_weights"))
        self.assertGreaterEqual(score.mixed_bits_per_byte, 0.0)

    def test_memory_stability_uses_kernel_substrates(self) -> None:
        module = load_module(
            "memory_stability_model_refactor_test",
            "examples/projects/causal/memory_stability/model.py",
        )
        model = module.MemoryStabilityModel.build()
        experts = model.model.experts

        self.assertIsInstance(experts[0].substrate, LinearMemorySubstrate)
        self.assertIsInstance(experts[1].substrate, OscillatoryMemorySubstrate)
        self.assertGreaterEqual(len(experts), 2)

    def test_scripts_still_run(self) -> None:
        probe_output = run_example("examples/projects/causal/statistical_memory/probe.py")
        smoke_output = run_example("examples/projects/causal/memory_stability/smoke.py")

        self.assertIn("project: statistical_memory", probe_output)
        self.assertIn("mixed bits/byte:", probe_output)
        self.assertIn("project: memory_stability", smoke_output)
        self.assertIn("mixed score bits/byte:", smoke_output)


if __name__ == "__main__":
    unittest.main()
