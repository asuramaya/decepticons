from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.config import (
    DelayLineConfig,
    HierarchicalSubstrateConfig,
    LatentConfig,
    MixedMemoryConfig,
    OpenPredictiveCoderConfig,
    ReservoirConfig,
)
from open_predictive_coder.delay import DelayLineSubstrate
from open_predictive_coder.factories import (
    create_delay_line_substrate,
    create_echo_state_substrate,
    create_hierarchical_substrate,
    create_mixed_memory_substrate,
    create_substrate,
    create_substrate_for_model,
)
from open_predictive_coder.hierarchical import HierarchicalSubstrate
from open_predictive_coder.mixed_memory import MixedMemorySubstrate
from open_predictive_coder.reservoir import EchoStateReservoir


class FactoryTests(unittest.TestCase):
    def test_echo_state_factory_uses_reservoir_config(self) -> None:
        config = ReservoirConfig(size=32, connectivity=0.2, spectral_radius=0.85, leak=0.3, seed=5)
        substrate = create_echo_state_substrate(config)

        self.assertIsInstance(substrate, EchoStateReservoir)
        state = substrate.initial_state()
        self.assertEqual(state.shape, (32,))
        self.assertTrue(np.all(state == 0.0))

    def test_delay_factory_builds_shift_register(self) -> None:
        config = DelayLineConfig(history_length=3, embedding_dim=4, vocabulary_size=16, seed=11)
        substrate = create_delay_line_substrate(config)

        self.assertIsInstance(substrate, DelayLineSubstrate)
        state = substrate.initial_state()
        self.assertEqual(state.shape, (12,))
        self.assertTrue(np.all(state == 0.0))

        next_state = substrate.step(state, 7)
        history = substrate.history_view(next_state)
        np.testing.assert_allclose(history[0], substrate._token_embeddings[7])
        self.assertTrue(np.allclose(history[1:], 0.0))

    def test_mixed_memory_factory_combines_branches(self) -> None:
        config = MixedMemoryConfig(
            reservoir=ReservoirConfig(size=24, connectivity=0.2, spectral_radius=0.9, leak=0.25, seed=17),
            delay=DelayLineConfig(history_length=2, embedding_dim=5, vocabulary_size=16, seed=9),
        )
        substrate = create_mixed_memory_substrate(config)

        self.assertIsInstance(substrate, MixedMemorySubstrate)
        self.assertEqual(substrate.state_dim, 24 + 10)
        state = substrate.initial_state()
        self.assertEqual(state.shape, (34,))
        self.assertTrue(np.all(state == 0.0))

        next_state = substrate.step(state, 3)
        self.assertEqual(next_state.shape, (34,))
        self.assertTrue(np.all(np.isfinite(next_state)))
        self.assertFalse(np.allclose(substrate.reservoir_view(next_state), 0.0))
        self.assertFalse(np.allclose(substrate.delay_view(next_state), 0.0))

    def test_hierarchical_factory_builds_multiscale_substrate(self) -> None:
        config = HierarchicalSubstrateConfig(
            fast_size=8,
            mid_size=12,
            slow_size=16,
            vocabulary_size=16,
            slow_update_stride=2,
            seed=13,
        )
        substrate = create_hierarchical_substrate(config)

        self.assertIsInstance(substrate, HierarchicalSubstrate)
        state = substrate.initial_state()
        self.assertEqual(state.shape, (36,))
        self.assertTrue(np.all(state == 0.0))

    def test_type_based_factory_dispatches_by_config(self) -> None:
        reservoir = ReservoirConfig(size=16, connectivity=0.25, spectral_radius=0.9, leak=0.2, seed=3)
        delay = DelayLineConfig(history_length=2, embedding_dim=3, vocabulary_size=8, seed=4)
        mixed = MixedMemoryConfig(
            reservoir=ReservoirConfig(size=16, connectivity=0.25, spectral_radius=0.9, leak=0.2, seed=3),
            delay=DelayLineConfig(history_length=2, embedding_dim=3, vocabulary_size=8, seed=4),
        )
        hierarchical = HierarchicalSubstrateConfig(
            fast_size=8,
            mid_size=10,
            slow_size=12,
            vocabulary_size=8,
            seed=6,
        )

        self.assertIsInstance(create_substrate(reservoir), EchoStateReservoir)
        self.assertIsInstance(create_substrate(delay), DelayLineSubstrate)
        self.assertIsInstance(create_substrate(mixed), MixedMemorySubstrate)
        self.assertIsInstance(create_substrate(hierarchical), HierarchicalSubstrate)

    def test_model_level_factory_dispatches_by_substrate_kind(self) -> None:
        latent = LatentConfig(latent_dim=8, global_dim=8, reservoir_features=8, readout_l2=1e-4)
        base = OpenPredictiveCoderConfig(vocabulary_size=32, latent=latent)
        delay = OpenPredictiveCoderConfig(
            vocabulary_size=32,
            substrate_kind="delay",
            delay=DelayLineConfig(history_length=2, embedding_dim=4, vocabulary_size=32, seed=9),
            latent=latent,
        )
        mixed = OpenPredictiveCoderConfig(
            vocabulary_size=32,
            substrate_kind="mixed_memory",
            mixed_memory=MixedMemoryConfig(
                reservoir=ReservoirConfig(size=16, connectivity=0.25, spectral_radius=0.9, leak=0.2, seed=3),
                delay=DelayLineConfig(history_length=2, embedding_dim=3, vocabulary_size=32, seed=4),
            ),
            latent=latent,
        )
        hierarchical = OpenPredictiveCoderConfig(
            vocabulary_size=32,
            substrate_kind="hierarchical",
            hierarchical=HierarchicalSubstrateConfig(
                fast_size=8,
                mid_size=10,
                slow_size=12,
                vocabulary_size=32,
                seed=6,
            ),
            latent=latent,
        )

        self.assertIsInstance(create_substrate_for_model(base), EchoStateReservoir)
        self.assertIsInstance(create_substrate_for_model(delay), DelayLineSubstrate)
        self.assertIsInstance(create_substrate_for_model(mixed), MixedMemorySubstrate)
        self.assertIsInstance(create_substrate_for_model(hierarchical), HierarchicalSubstrate)

    def test_factory_rejects_unknown_config(self) -> None:
        with self.assertRaises(TypeError):
            create_substrate(object())


if __name__ == "__main__":
    unittest.main()
