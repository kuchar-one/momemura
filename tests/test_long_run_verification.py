import os
import sys
import numpy as np
import shutil

# Force CPU
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# jax.config.update("jax_disable_jit", True)

sys.path.append(os.getcwd())

from run_mome import run
from src.utils.result_manager import OptimizationResult
from frontend.utils import compute_heralded_state

OUTPUT_DIR = "tests/test_output_long_run"


def test_long_run_verification():
    """
    Runs a short but complete optimization (MOME).
    Verifies that for every solution in the Pareto front:
    1. Simulated Probability (Frontend) matches Stored LogProb (Backend).
    2. Simulated Probability <= 1.0.
    """

    # 0. Cleanup
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # 1. Run Optimization
    # Pass depth in genotype_config
    genotype_config = {
        "depth": 3,
        "pnr_max": 3,  # explicit
    }

    print("Starting Optimization...")
    run(
        mode="random",
        genotype="B30B",
        n_iters=2,  # Iterations (generations)
        pop_size=4,
        output_root=OUTPUT_DIR,
        seed=42,
        genotype_config=genotype_config,
        correction_cutoff=30,  # Use dynamic limits to match user scenario
        cutoff=12,
        backend="jax",  # Force JAX backend for debugging
        no_plot=True,  # Disable plotting to save time
    )
    print("Optimization Finished.")

    # 2. Load Results
    exp_dir = os.path.join(OUTPUT_DIR, "experiments")
    if not os.path.exists(exp_dir):
        assert False, "Experiments dir not created"

    # Find group dir
    group_dirs = os.listdir(exp_dir)
    if not group_dirs:
        assert False, "No group dir found"
    group_dir_path = os.path.join(exp_dir, group_dirs[0])

    # Find run dir (timestamp)
    run_dirs = os.listdir(group_dir_path)
    if not run_dirs:
        assert False, "No run dir found"
    run_dir = os.path.join(group_dir_path, run_dirs[0])

    print(f"Loading results from {run_dir}")

    result = OptimizationResult.load(run_dir)
    df = result.get_pareto_front()

    print(f"Found {len(df)} solutions on Pareto front.")

    if len(df) == 0:
        print("Warning: No solutions found.")
        return

    # 3. Verify inconsistencies
    mismatches = []
    explosions = []

    config = result.config

    # Dynamic Limits Logic (from app.py fix)
    base_cutoff = int(config.get("cutoff", 12))
    corr_cutoff = config.get("correction_cutoff")
    sim_cutoff = base_cutoff
    if corr_cutoff is not None and int(corr_cutoff) > base_cutoff:
        sim_cutoff = int(corr_cutoff)

    pnr_max = int(config.get("pnr_max", 3))

    print(
        f"Simulating with cutoff={sim_cutoff} (Base={base_cutoff}, Corr={corr_cutoff})"
    )

    for idx, row in df.iterrows():
        # Retrieve circuit params (decoded)
        g_idx = int(row["genotype_idx"])
        params = result.get_circuit_params(g_idx)

        # Simulate
        psi, prob = compute_heralded_state(params, cutoff=sim_cutoff, pnr_max=pnr_max)

        # Stored LogProb is usually NLL (Positive).
        sim_nll = -np.log10(prob) if prob > 0 else np.inf

        stored_nll = float(row["LogProb"])

        # Compare NLLs
        if abs(stored_nll - sim_nll) > 0.1:  # Tolerance
            mismatches.append((g_idx, stored_nll, sim_nll))

        if prob > 1.00001:
            explosions.append((g_idx, prob))

    # 4. Report
    if len(mismatches) > 0:
        print(f"FAILED: {len(mismatches)} mismatches found.")
        for mid, st, sim in mismatches[:5]:
            print(f"ID {mid}: Stored {st:.4f}, Sim {sim:.4f}")

    if len(explosions) > 0:
        print(f"FAILED: {len(explosions)} probability explosions (P > 1) found.")
        for eid, p in explosions[:5]:
            print(f"ID {eid}: Prob {p}")

    assert len(mismatches) == 0, "Frontend-Backend Mismatch Detected"
    assert len(explosions) == 0, "Probability Explosion Detected"

    print("Verification Passed: All probabilities consistent and valid.")


if __name__ == "__main__":
    test_long_run_verification()
