import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

# Mock dependencies not present in env
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["qutip"] = MagicMock()
sys.modules["streamlit"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend import utils
from frontend import visualizations as viz
from src.utils.result_manager import OptimizationResult


class TestFrontend(unittest.TestCase):
    def test_list_runs(self):
        runs = utils.list_runs("output")
        self.assertTrue(len(runs) > 0)
        print(f"Found {len(runs)} runs.")

    def test_load_and_process_run(self):
        runs = utils.list_runs("output")
        latest_run = runs[0]
        path = os.path.join("output", latest_run)
        print(f"Testing with run: {path}")

        result = utils.load_run(path)
        self.assertIsInstance(result, OptimizationResult)

        df = result.get_pareto_front()
        print(f"Pareto front has {len(df)} points.")

        if not df.empty:
            # Test circuit reconstruction for the first point
            idx = df.iloc[0]["genotype_idx"]
            params = result.get_circuit_params(int(idx))
            self.assertIsInstance(params, dict)

            # Test Circuit Figure
            fig = utils.get_circuit_figure(params)
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)

            # Test State Calculation
            # This relies on thewalrus/numpy which are present
            try:
                psi, prob = utils.compute_heralded_state(params, cutoff=10)
                self.assertIsInstance(psi, np.ndarray)
                self.assertTrue(0 <= prob <= 1.0)
            except ImportError:
                print("Skipping heralded state calculation (deps missing)")

            # Test Wigner (Mocked Qutip)
            # Setup mock return
            utils.qt.wigner.return_value = np.zeros((30, 30))

            xvec = np.linspace(-3, 3, 30)
            pvec = np.linspace(-3, 3, 30)
            W = utils.compute_wigner(psi, xvec, pvec)
            self.assertEqual(W.shape, (30, 30))
            # Verify call
            utils.qt.Qobj.assert_called_once()
            utils.qt.wigner.assert_called_once()

            # Test Visualization Functions (Mocked Plotly)
            viz.px.scatter.return_value = MagicMock()
            fig_pareto = viz.plot_global_pareto(df)
            self.assertIsNotNone(fig_pareto)
            viz.px.scatter.assert_called()

            viz.px.imshow.return_value = MagicMock()
            fig_heat = viz.plot_best_expectation_heatmap(df)
            self.assertIsNotNone(fig_heat)
            viz.px.imshow.assert_called()

    def test_empty_dataframe_plotting(self):
        """Ensure plotting functions handle empty DataFrames gracefully."""
        # Create empty DataFrame with expected columns
        import pandas as pd

        empty_df = pd.DataFrame(
            columns=["Expectation", "LogProb", "Complexity", "TotalPhotons"]
        )

        # Should return empty figure (or handle it without crash)
        viz.go.Figure.return_value = MagicMock()
        fig = viz.plot_global_pareto(empty_df)
        self.assertIsNotNone(fig)

        fig2 = viz.plot_best_expectation_heatmap(empty_df)
        self.assertIsNotNone(fig2)


if __name__ == "__main__":
    unittest.main()
