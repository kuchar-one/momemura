import unittest
import numpy as np
import sys
import os
from scipy.special import eval_hermite
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.simulation.cpu.ops import (
    annihilation_operator,
    creation_operator,
    build_beamsplitter_unitary,
    quadrature_vector,
    get_phi_matrix_cached,
)
from src.simulation.cpu.composer import Composer


class TestOptimization(unittest.TestCase):
    def test_vectorized_quadrature_correctness(self):
        """Verify that vectorized quadrature matches the scalar version."""
        cutoff = 10
        xs = np.linspace(-2, 2, 5)
        hbar = 2.0

        # Scalar version (reference)
        expected = np.zeros((cutoff, len(xs)))
        for i, x in enumerate(xs):
            expected[:, i] = quadrature_vector(cutoff, x, hbar)

        # Vectorized version
        actual = get_phi_matrix_cached(cutoff, xs, hbar)

        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_quadrature_caching(self):
        """Verify that caching works and returns the same object."""
        cutoff = 5
        xs = np.array([0.1, 0.2])
        hbar = 2.0

        res1 = get_phi_matrix_cached(cutoff, xs, hbar)
        res2 = get_phi_matrix_cached(cutoff, xs, hbar)

        self.assertIs(res1, res2)

        # Different params should return different object
        res3 = get_phi_matrix_cached(cutoff + 1, xs, hbar)
        self.assertIsNot(res1, res3)

    def test_composer_integration(self):
        """Verify Composer uses the new path without crashing."""
        c = Composer(cutoff=4)
        stateA = np.array([1, 0, 0, 0], dtype=complex)
        stateB = np.array([0, 1, 0, 0], dtype=complex)

        # Window homodyne (uses leggauss + vectorized)
        rho, p, joint = c.compose_pair(
            stateA, stateB, homodyne_x=0.0, homodyne_window=0.1
        )
        self.assertEqual(rho.shape, (4, 4))
        self.assertIsInstance(p, float)

        # Point homodyne (uses vectorized point)
        rho2, p2, joint2 = c.compose_pair(
            stateA, stateB, homodyne_x=0.0, homodyne_window=None
        )
        self.assertEqual(rho2.shape, (4,))  # Pure path returns vector
        self.assertIsInstance(p2, float)


if __name__ == "__main__":
    unittest.main()
