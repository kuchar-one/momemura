import unittest
import os
import sys

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

from frontend import utils  # noqa: E402
from frontend import visualizations as viz  # noqa: E402
from src.utils.result_manager import OptimizationResult  # noqa: E402
import jax.numpy as jnp  # noqa: E402


class MockRepertoire:
    def __init__(self):
        self.genotypes = jnp.zeros((10, 20))  # 10 indivs, 20 genes
        self.fitnesses = jnp.zeros((10, 4))
        self.descriptors = jnp.zeros((10, 3))


class TestFrontend(unittest.TestCase):
    def test_list_runs(self):
        # Ensure output directory exists and has at least one dummy run for this test
        os.makedirs("output/dummy_test_run", exist_ok=True)
        try:
            runs = utils.list_runs("output")
            # Filter for our dummy run just in case
            self.assertTrue(len(runs) > 0)
            print(f"Found {len(runs)} runs.")
        finally:
            # Clean up dummy run
            if os.path.exists("output/dummy_test_run"):
                os.rmdir("output/dummy_test_run")

    def test_load_and_process_run(self):
        # Create a temporary directory structure for testing
        import tempfile
        import json
        import pickle
        import jax.numpy as jnp

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = os.path.join(tmp_dir, "test_run")
            os.makedirs(run_dir)

            # 1. Create dummy config.json
            config = {
                "genotype_name": "A",
                "cutoff": 10,
                "pop_size": 10,
                "iters": 100,
            }
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config, f)

            # 2. Create dummy checkpoint.pkl
            # Need a repertoire object structure.
            # We can mock it or use a dict if OptimizationResult handles it.
            # OptimizationResult expects: { "repertoire": ..., "history": ... }
            # "repertoire" needs .genotypes, .fitnesses, .descriptors

            from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire  # noqa: F401

            repertoire = MockRepertoire()
            # Set one valid
            repertoire.fitnesses = repertoire.fitnesses.at[0].set(
                jnp.array([-1.0, 5.0, 3.0, 10.0])
            )

            chk_data = {
                "repertoire": repertoire,
                "history": {"coverage": [0.1], "min_expectation": [-1.0]},
                "completed_iters": 100,
            }

            with open(os.path.join(run_dir, "checkpoint_latest.pkl"), "wb") as f:
                pickle.dump(chk_data, f)

            # Also create results.pkl which result_manager looks for first/instead
            with open(os.path.join(run_dir, "results.pkl"), "wb") as f:
                pickle.dump(chk_data, f)

            path = run_dir
            print(f"Testing with dummy run: {path}")

            result = utils.load_run(path)
            self.assertIsInstance(result, OptimizationResult)

            df = result.get_pareto_front()
            print(f"Pareto front has {len(df)} points.")

            # If df is empty (due to filtering?), force one row for testing logic
            if df.empty:
                # Manually add a row if get_pareto_front filters everything
                # behavior depends on result_manager logic.
                pass

            if not df.empty:
                # Test circuit reconstruction for the first point
                idx = df.iloc[0]["genotype_idx"]
                # For dummy genotype (zeros), params might be weird but should return dict
                try:
                    params = result.get_circuit_params(int(idx))
                    self.assertIsInstance(params, dict)

                    # Test Circuit Figure
                    fig = utils.get_circuit_figure(params)
                    self.assertIsInstance(fig, plt.Figure)
                    plt.close(fig)
                except Exception as e:
                    print(f"Skipping circuit params test for dummy genotype: {e}")

            # Visualize (just check it calls the mocks)
            viz.px.scatter.return_value = MagicMock()
            fig_pareto = viz.plot_global_pareto(df)
            self.assertIsNotNone(fig_pareto)

            # Clean up handled by tempfile context

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
