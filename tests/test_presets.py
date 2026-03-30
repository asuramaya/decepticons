from __future__ import annotations

import unittest

from open_predictive_coder import ByteLatentPredictiveCoder
from open_predictive_coder.presets import delay_small, echo_state_small, hierarchical_small, mixed_memory_small


class PresetTests(unittest.TestCase):
    def test_presets_construct_usable_models(self) -> None:
        for config in (echo_state_small(), delay_small(), mixed_memory_small(), hierarchical_small()):
            model = ByteLatentPredictiveCoder(config=config)
            trace = model.trace("abababab")
            self.assertEqual(trace.features.shape[1], config.feature_dim)
            self.assertGreaterEqual(trace.patches, 1)


if __name__ == "__main__":
    unittest.main()
