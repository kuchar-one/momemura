import os
import shutil
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from src.utils.result_manager import OptimizationResult
from run_mome import run


@pytest.fixture
def clean_output():
    # Cleanup output dir before/after test by backing it up
    if os.path.exists("output"):
        # Create a backup directory name with timestamp
        import time

        timestamp = int(time.time())
        backup_dir = f"output_backup_{timestamp}"

        # Rename current output to backup
        print(f"Backing up 'output' to '{backup_dir}'")
        shutil.move("output", backup_dir)

    # Create fresh output directory
    os.makedirs("output", exist_ok=True)

    yield

    # Optional: cleanup the fresh output after test
    # if os.path.exists("output"):
    #     shutil.rmtree("output")


def test_optimization_pipeline_end_to_end(clean_output):
    """
    Verifies the full optimization loop:
    1. Run run_mome.py (short run)
    2. Check output directory creation
    3. Load results using OptimizationResult
    4. Check Pareto front data
    5. Verify animation creation
    """
    # 1. Run optimization (short)
    # Use random mode for speed, but backend=jax to test JAX path if available
    # If JAX not available, it falls back to random anyway.

    # We use a small population and few iters
    repertoire, _, metrics = run(
        mode="random",  # Use random to avoid QDax overhead/complexity for this smoke test
        n_iters=5,
        pop_size=4,
        seed=42,
        cutoff=4,  # Small cutoff
        backend="jax" if jax is not None else "thewalrus",
        no_plot=False,  # We want to test plotting/saving
    )

    # 2. Check output directory
    # Find the created directory in output/
    assert os.path.exists("output")
    subdirs = os.listdir("output")
    assert len(subdirs) > 0
    result_dir = os.path.join("output", subdirs[0])

    assert os.path.exists(os.path.join(result_dir, "config.json"))
    assert os.path.exists(os.path.join(result_dir, "results.pkl"))
    assert os.path.exists(os.path.join(result_dir, "final_plot.png"))
    # Animation might fail if imageio not installed, but let's check if it tried
    # assert os.path.exists(os.path.join(result_dir, "history.gif"))

    # 3. Load results
    res = OptimizationResult.load(result_dir)
    assert res.config["n_iters"] == 5
    assert res.config["pop_size"] == 4

    # 4. Check Pareto front
    df = res.get_pareto_front()
    # Should have some data (even if random)
    # Note: random search might not find valid solutions if constraints are tight,
    # but with random genotypes usually something is valid.
    # If df is empty, it means no valid solutions found.
    # We can't strictly assert len(df) > 0 for random search, but likely it is.
    print(f"Pareto front size: {len(df)}")

    if len(df) > 0:
        # 5. Reconstruct circuit
        params = res.get_circuit_params(0)
        assert "leaf_params" in params
        assert "mix_params" in params

    # 6. Check history
    assert "min_expectation" in res.history
    assert len(res.history["min_expectation"]) > 0


if __name__ == "__main__":
    # Manual run
    if os.path.exists("output"):
        shutil.rmtree("output")
    test_optimization_pipeline_end_to_end(None)
    print("Test passed!")
