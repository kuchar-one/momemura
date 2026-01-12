import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from src.utils.result_scanner import compute_pareto_front


class TestParetoSeeding(unittest.TestCase):
    def setUp(self):
        # Create dummy candidates
        # Fit0: -Exp (Maximize -> Min Exp)
        # Fit1: LogProb (Maximize -> Max Prob)

        self.candidates = [
            # Pareto Points (Higher Fit0, Higher Fit1)
            {
                "name": "A",
                "fit0": -1.0,
                "fit1": -1.0,
                "score": -1.0,
                "genotype": np.array([1]),
            },  # Dominated by B? No.
            {
                "name": "B",
                "fit0": -0.5,
                "fit1": -2.0,
                "score": -0.5,
                "genotype": np.array([2]),
            },  # Better Exp, Worse Prob
            {
                "name": "C",
                "fit0": -2.0,
                "fit1": -0.5,
                "score": -2.0,
                "genotype": np.array([3]),
            },  # Worse Exp, Better Prob
            # Dominated Points
            {
                "name": "D",
                "fit0": -3.0,
                "fit1": -3.0,
                "score": -3.0,
                "genotype": np.array([4]),
            },  # Dominated by A
            {
                "name": "E",
                "fit0": -1.5,
                "fit1": -2.5,
                "score": -1.5,
                "genotype": np.array([5]),
            },  # Dominated by A?
            # A(-1, -1) vs E(-1.5, -2.5): A > E in both. A dominates E.
        ]

    def test_compute_pareto_front(self):
        # Pareto: A, B, C should be on front.
        # A(-1, -1)
        # B(-0.5, -2) -> Higher Fit0, Lower Fit1.
        # C(-2, -0.5) -> Lower Fit0, Higher Fit1.

        # Check sort order logic in compute_pareto_front:
        # Sort desc by Fit0:
        # 1. B (-0.5, -2.0) -> Max Fit1 seen: -inf -> Keep (Max=-2.0)
        # 2. A (-1.0, -1.0) -> Fit1(-1.0) > Max(-2.0) -> Keep (Max=-1.0)
        # 3. E (-1.5, -2.5) -> Fit1(-2.5) !> Max(-1.0) -> Drop
        # 4. C (-2.0, -0.5) -> Fit1(-0.5) > Max(-1.0) -> Keep (Max=-0.5)
        # 5. D (-3.0, -3.0) -> Fit1(-3.0) !> Max(-0.5) -> Drop

        # Expected: B, A, C

        front = compute_pareto_front(self.candidates)
        names = [c["name"] for c in front]

        self.assertIn("A", names)
        self.assertIn("B", names)
        self.assertIn("C", names)
        self.assertNotIn("D", names)
        self.assertNotIn("E", names)

        # Order check (Sorted by Fit0 desc)
        self.assertEqual(names, ["B", "A", "C"])

    def test_sampling_logic_mock(self):
        # We can't easily mock scan_results_for_seeds directory walk without temp dirs,
        # but we can test the logic flow if we extract it.
        # For now, relying on compute_pareto_front test is good for the core logic.
        pass


if __name__ == "__main__":
    unittest.main()
