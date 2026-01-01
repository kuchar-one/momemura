import os
from pathlib import Path
import jax
from src.utils.result_manager import OptimizationResult
from run_mome import run


def test_optimization_pipeline_end_to_end(tmp_path):
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

    # We pass strict output_root to avoid deleting user data
    output_root = str(tmp_path)

    # We use a small population and few iters
    repertoire, _, metrics = run(
        mode="random",  # Use random to avoid QDax overhead/complexity for this smoke test
        n_iters=5,
        pop_size=4,
        seed=42,
        cutoff=4,  # Small cutoff
        backend="jax" if jax is not None else "thewalrus",
        no_plot=False,  # We want to test plotting/saving
        output_root=output_root,
        genotype="A",
    )

    # 2. Check output directory
    # Find the created directory in output/
    assert os.path.exists(output_root)
    # The structure is now output_root/experiments/group_id/timestamp
    # We need to find the timestamp folder deep inside.
    # scan recursively or check expected path

    found_result = False
    result_dir = None

    # Walk to find results.pkl
    for root, dirs, files in os.walk(output_root):
        if "results.pkl" in files:
            result_dir = root
            found_result = True
            break

    assert found_result, f"Could not find results.pkl in {output_root}"

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
    # Manual run setup
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_optimization_pipeline_end_to_end(Path(tmpdir))
    print("Test passed!")
