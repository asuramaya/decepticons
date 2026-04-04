from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

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


class ReplayFieldsExampleTests(unittest.TestCase):
    def test_replay_fields_wraps_noncausal_adapter(self) -> None:
        module = load_module(
            "replay_fields_model_test",
            "examples/projects/noncausal/replay_fields/model.py",
        )
        model = module.ReplayFieldsModel.build()
        corpus = (
            "title:alpha|body:repeat repeat repeat|tag:x\n"
            "title:alpha|body:repeat repeat repeat|tag:x\n"
            "title:beta|body:variation here|tag:y\n"
        )

        fit = model.fit(corpus)
        trace = model.trace(corpus)
        report = model.score(corpus)

        self.assertGreater(fit.reconstruction.forward.tokens, 0)
        self.assertGreaterEqual(len(trace.field_spans), 1)
        self.assertEqual(trace.replay_field_overlap.shape[0], trace.reconstruction.tokens)
        self.assertGreaterEqual(report.field_span_count, 1)
        self.assertGreaterEqual(report.replay_field_overlap_count, 0)
        self.assertGreaterEqual(report.replay_field_ratio, 0.0)
        self.assertLessEqual(report.replay_field_ratio, 1.0)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/noncausal/replay_fields/probe.py")
        smoke_output = run_example("examples/projects/noncausal/replay_fields/smoke.py")

        self.assertIn("project: replay_fields", probe_output)
        self.assertIn("replay field ratio:", probe_output)
        self.assertIn("replay spans:", smoke_output)
        self.assertIn("field spans:", smoke_output)


if __name__ == "__main__":
    unittest.main()
