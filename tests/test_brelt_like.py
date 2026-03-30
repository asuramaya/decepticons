from __future__ import annotations

from importlib import util
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "brelt_like"


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


class BreltLikeTests(unittest.TestCase):
    def test_trace_and_score_cover_patch_commit_shape(self) -> None:
        module = load_module("brelt_like_model_test", "examples/projects/brelt_like/model.py")
        model = module.BreltLikeByteModel(module.BreltLikeConfig())
        corpus = (
            "brelt compresses bytes into patch latents, then mixes them globally.\n"
            "short internal streams should carry the main burden.\n"
        ) * 6

        trace = model.trace(corpus[:160])
        fit_report = model.fit(corpus)
        score_report = model.score(corpus)
        probs = model.predict_proba("brelt ")

        self.assertEqual(trace.features.shape[1], model.feature_dim)
        self.assertEqual(trace.targets.shape[0], trace.features.shape[0])
        self.assertGreater(trace.patches, 0)
        self.assertLess(trace.patches, trace.tokens)
        self.assertGreater(fit_report.train_bits_per_byte, 0.0)
        self.assertGreater(score_report.bits_per_byte, 0.0)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/brelt_like/probe.py")
        smoke_output = run_example("examples/projects/brelt_like/smoke.py")

        self.assertIn("feature_dim:", probe_output)
        self.assertIn("patch_size:", probe_output)
        self.assertIn("train bits/byte:", smoke_output)
        self.assertIn("compression ratio:", smoke_output)


if __name__ == "__main__":
    unittest.main()
