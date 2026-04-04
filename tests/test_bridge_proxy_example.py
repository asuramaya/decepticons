from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "bridge" / "proxy_features"


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


class BridgeProxyExampleTests(unittest.TestCase):
    def test_trace_and_report_shapes(self) -> None:
        module = load_module("bridge_proxy_model_test", "examples/projects/bridge/proxy_features/model.py")
        model = module.BridgeProxyModel(module.BridgeProxyConfig())
        corpus = (
            "bridge features compare a causal stream and a proxy stream.\n"
            "the causal side stays simple and the proxy side stays local.\n"
        ) * 4

        trace = model.trace(corpus[:192])
        report = model.report(corpus)
        summary = model.feature_summary(corpus)

        self.assertEqual(trace.base_probs.shape, trace.proxy_probs.shape)
        self.assertEqual(trace.features.entropy.shape[0], trace.base_probs.shape[0])
        self.assertGreater(trace.tokens, trace.steps)
        self.assertGreaterEqual(report.mean_entropy, 0.0)
        self.assertLessEqual(report.mean_peak, 1.0)
        self.assertLessEqual(report.mean_candidate4, 1.0)
        self.assertGreaterEqual(report.mean_agreement, 0.0)
        self.assertLessEqual(report.mean_agreement_mass, 1.0)
        self.assertAlmostEqual(summary["mean_entropy"], report.mean_entropy, places=12)
        self.assertAlmostEqual(summary["mean_peak"], report.mean_peak, places=12)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/bridge/proxy_features/probe.py")
        smoke_output = run_example("examples/projects/bridge/proxy_features/smoke.py")

        self.assertIn("candidate_count:", probe_output)
        self.assertIn("mean_entropy:", probe_output)
        self.assertIn("mean agreement mass:", smoke_output)


if __name__ == "__main__":
    unittest.main()
