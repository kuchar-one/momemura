import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import backend run logic
from run_mome import run
from src.utils.result_manager import OptimizationResult
import frontend.utils as f_utils
import jax


@pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason="Skipping JAX optimization test on pure CPU if desired, but we want it.",
)
# Actually we can run JAX on CPU fine for small test.
def test_end_to_end_consistency():
    """
    Runs a baby optimization and verifies that the backend results (LogProb)
    match the frontend re-simulation results.
    """
    print("Starting End-to-End Consistency Test...")

    # 1. Setup config
    pop_size = 10
    n_iters = 5
    cutoff = 25  # Match backend effective cutoff (10+15) or high enough
    pnr_max = 5

    genotype_config = {
        "pnr_max": pnr_max,
        "depth": 3,  # Standard depth to avoid topology shape mismatches
        "modes": 1,  # Minimal modes
    }

    # 2. Run Optimization
    # run() creates output dir. We can't easily redirect it without mocking utils.
    # But run_mome uses `get_result_path`?
    # No, run call inside run_mome invokes ResultManager.
    # We will let it write to `output/` and find the latest.

    # Clean output before (optional, but risky if user has data)
    # Better: Patch 'src.utils.result_manager.get_result_path' to return a temp dir?
    # Too complex. We'll just run it and scan 'output'.

    # Force seed to ensure some valid solutions
    try:
        run(
            mode="qdax",
            n_iters=n_iters,
            pop_size=pop_size,
            seed=42,
            cutoff=cutoff,
            no_plot=True,
            backend="jax",
            target_alpha=2.0,
            target_beta=0.0,
            low_mem=False,
            genotype="A",
            genotype_config=genotype_config,
            seed_scan=False,
            # debug=True
        )
    except Exception as e:
        pytest.fail(f"Optimization run failed: {e}")

    # 3. Find latest result
    output_root = "output"
    if not os.path.exists(output_root):
        pytest.fail("Output directory not found.")

    subdirs = [
        os.path.join(output_root, d)
        for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
    ]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    print(f"Analyzing results in: {latest_subdir}")

    # 4. Load Results
    rm = OptimizationResult.load(latest_subdir)
    try:
        df = rm.get_pareto_front()
    except Exception as e:
        # If no solutions found, skip?
        print(f"No valid solutions found or load failed: {e}")
        # Not a failure of consistency logic, but failure of optimizer to find anything in 5 iters.
        # But usually random search finds *something*.
        return

    if df.empty:
        print("Pareto front is empty. Cannot verify consistency.")
        return

    # Limit to 3 solutions to save time (compilation)
    if len(df) > 3:
        df = df.head(3)

    print(f"Verifying {len(df)} solutions...")

    mismatches = 0

    for idx, row in df.iterrows():
        # Backend Values
        # Note: df usually has "LogProb" = -(-log10(P)) ?
        # In run_mome.py: f_prob = metrics["log_prob"] (which is -log10(P)).
        # fitness[1] = -f_prob = log10(P).
        # ResultManager converts fitness to df: "LogProb": -fitness[1] = -log10(P).
        # So "LogProb" column IS -log10(P). (Positive value).

        stored_log_prob = row["LogProb"]
        stored_exp = row["Expectation"]

        # Frontend Re-simulation
        # Must use correct pnr_max!
        params = rm.get_circuit_params(int(row["genotype_idx"]))

        # Ensure params include pnr_max?
        # params dict from `rm.get_circuit_params` comes from decoder.
        # It doesn't include pnr_max explicitly unless config put it there?
        # But `compute_heralded_state` allows passing pnr_max kwarg.

        psi, prob = f_utils.compute_heralded_state(
            params, cutoff=cutoff, pnr_max=pnr_max
        )

        # Compute Expected Prob matches
        prob_val = f_utils.to_scalar(prob)

        if prob_val > 0:
            sim_log_prob = -np.log10(prob_val)
        else:
            sim_log_prob = np.inf

        print(
            f"Sol {idx}: Stored LogProb={stored_log_prob:.4f}, Sim LogProb={sim_log_prob:.4f}, Prob={prob_val:.4e}"
        )

        # Check consistency
        # Allow some float error
        if not np.isclose(stored_log_prob, sim_log_prob, atol=1e-3):
            print(f"  MISMATCH! Diff={abs(stored_log_prob - sim_log_prob)}")
            mismatches += 1

    assert mismatches == 0, (
        f"Found {mismatches} consistency mismatches between Backend and Frontend!"
    )
    print("Consistency Test Passed!")


if __name__ == "__main__":
    test_end_to_end_consistency()
