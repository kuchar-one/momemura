import unittest
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from run_mome import decode_genotype, custom_metrics, HanamuraMOMEAdapter
from src.circuits.composer import Composer


class TestStage0Fixes(unittest.TestCase):
    def test_decode_genotype_unused_slots(self):
        """Test that decode_genotype handles unused slots correctly."""
        # Create a dummy genotype
        genotype = np.random.rand(100)
        params = decode_genotype(genotype)
        # Check that n_signal is 1 (forced)
        self.assertEqual(params["n_signal"], 1)
        # Check that other params are decoded reasonably
        self.assertIn("n_control", params)

    def test_custom_metrics_shape_handling(self):
        """Test custom_metrics with correct repertoire shape."""
        # Mock repertoire with shape (N, Pareto, Objs)
        # Let's say N=10, Pareto=5, Objs=4
        N = 10
        Pareto = 5
        Objs = 4

        # Create dummy fitnesses
        # fitnesses are maximized in QDax, so we use negative values for minimization objectives
        # -inf indicates invalid/empty
        fitnesses = jnp.full((N, Pareto, Objs), -jnp.inf)

        # Fill some values
        # Cell 0: valid solution at pareto index 0
        fitnesses = fitnesses.at[0, 0, :].set(jnp.array([-1.0, -2.0, -3.0, -4.0]))
        # Cell 1: valid solution at pareto index 0 and 1
        fitnesses = fitnesses.at[1, 0, :].set(jnp.array([-0.5, -2.5, -3.5, -4.5]))
        fitnesses = fitnesses.at[1, 1, :].set(jnp.array([-0.6, -2.4, -3.6, -4.6]))

        # Mock Repertoire object (just a simple class or namedtuple)
        from collections import namedtuple

        Repertoire = namedtuple("Repertoire", ["fitnesses"])
        repertoire = Repertoire(fitnesses=fitnesses)

        metrics = custom_metrics(repertoire)

        # Coverage: 2 cells out of 10 are filled
        self.assertAlmostEqual(metrics["coverage"], 0.2)

        # Min expectation (max fitness[0])
        # Cell 0: -1.0
        # Cell 1: -0.5 (better)
        # So max fitness[0] is -0.5. Min expectation is 0.5
        self.assertAlmostEqual(metrics["min_expectation"], 0.5)

    def test_homodyne_naming_consistency(self):
        """Check Composer method signatures for homodyne parameters."""
        c = Composer(cutoff=5)
        # Just checking if arguments exist and don't crash
        f1 = np.zeros(5, dtype=complex)
        f1[0] = 1.0
        f2 = np.zeros(5, dtype=complex)
        f2[0] = 1.0

        # Point homodyne
        res_point = c.compose_pair(f1, f2, homodyne_x=0.0, homodyne_resolution=0.1)
        self.assertEqual(len(res_point), 3)

        # Window homodyne
        res_win = c.compose_pair(f1, f2, homodyne_x=0.0, homodyne_window=0.1)
        self.assertEqual(len(res_win), 3)

    def test_jax_prng_key(self):
        """Verify jax.random.PRNGKey works (and jax.random.key might fail if deprecated)."""
        try:
            key = jax.random.PRNGKey(42)
            self.assertIsNotNone(key)
        except AttributeError:
            self.fail("jax.random.PRNGKey not found")


if __name__ == "__main__":
    unittest.main()
