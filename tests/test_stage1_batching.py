import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from run_mome import HanamuraMOMEAdapter
from src.simulation.cpu.composer import Composer, SuperblockTopology


class TestStage1Batching(unittest.TestCase):
    def setUp(self):
        self.cutoff = 4
        self.composer = Composer(cutoff=self.cutoff)
        self.topology = SuperblockTopology.build_layered(2)
        self.operator = np.diag(np.arange(self.cutoff, dtype=float))
        self.adapter = HanamuraMOMEAdapter(
            self.composer,
            self.topology,
            self.operator,
            cutoff=self.cutoff,
            mode="random",
        )
        self.D = 40  # genotype dim

    def test_batch_scoring_shapes(self):
        """Test that scoring_fn_batch returns correct shapes for a batch."""
        batch_size = 10
        genotypes = np.random.randn(batch_size, self.D)

        fitnesses, descriptors, extras = self.adapter.scoring_fn_batch(genotypes, None)

        self.assertEqual(fitnesses.shape, (batch_size, 4))
        self.assertEqual(descriptors.shape, (batch_size, 3))
        self.assertEqual(len(extras), batch_size)

    def test_batch_scoring_robustness(self):
        """Test that scoring_fn_batch handles invalid genotypes without crashing."""
        batch_size = 5
        genotypes = np.random.randn(batch_size, self.D)

        # Make one genotype invalid (e.g. by mocking evaluate_one to raise)
        # We can't easily mock evaluate_one on the instance without patching,
        # but we can rely on the fact that random genotypes might produce invalid states
        # (though unlikely to raise Exception unless we force it).
        # Let's monkeypatch evaluate_one for this test.

        original_eval = self.adapter.evaluate_one

        def mock_eval(genotype):
            if genotype[0] > 0:  # Arbitrary condition
                raise ValueError("Simulated failure")
            # Return dummy metrics instead of calling original_eval to avoid logic errors with random genotypes
            return {
                "expectation": 0.5,
                "log_prob": 2.0,
                "complexity": 10.0,
                "total_measured_photons": 1.0,
                "per_detector_max": 0.5,
            }

        self.adapter.evaluate_one = mock_eval

        try:
            # Force at least one failure
            genotypes[0, 0] = 1.0
            genotypes[1, 0] = -1.0

            fitnesses, descriptors, extras = self.adapter.scoring_fn_batch(
                genotypes, None
            )

            # Check failed index
            self.assertTrue(np.all(fitnesses[0] == -np.inf))
            self.assertEqual(descriptors[0, 0], -9999.0)
            self.assertIn("error", extras[0])

            # Check success index
            self.assertFalse(np.all(fitnesses[1] == -np.inf))

        finally:
            self.adapter.evaluate_one = original_eval


if __name__ == "__main__":
    unittest.main()
