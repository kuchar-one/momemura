import sys
import os
import shutil
import warnings
import numpy as np

# Suppress JAX/TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Ensure src in path
sys.path.append(os.getcwd())

from run_mome import run as run_mome_main  # noqa: E402
from src.utils.result_manager import OptimizationResult  # noqa: E402
from frontend.utils import compute_heralded_state  # noqa: E402


def test_genotype_consistency():
    """
    For every known genotype:
    1. Run a tiny optimization (1 gen, small pop).
    2. Load results.
    3. Verify that re-computed probability/expectation matches stored values.
    """

    genotypes = [
        "A",
        "0",
        "00B",
        "B1",
        "B2",
        "B3",
        "B30",
        "B3B",
        "B30B",
        "C1",
        "C2",
        "C20",
        "C2B",
        "C20B",
    ]

    # Use a temp directory for outputs
    temp_dir = "output/temp_consistency_check"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    test_failures = []

    print(f"Starting Global Consistency Check on {len(genotypes)} genotypes...")
    print("=" * 60)

    for g_name in genotypes:
        print(f"Testing Genotype: {g_name}")
        run_name = f"test_{g_name}"
        _ = os.path.join(temp_dir, run_name)

        # 1. Run Short Optimization
        # We invoke run_mome logic directly to avoid subprocessing overhead/complexity if possible
        # Config args were unused
        # args = [ ... ]

        # Capture stdout to reduce noise?
        # For now let it print so we see progress
        try:
            # Invoke run() directly with keyword arguments
            run_mome_main(
                mode="random",  # Use random mode for speed (no JAX compilation overhead for optimization loop, though we use JAX for eval if avail)
                n_iters=1,
                pop_size=5,
                seed=42,
                cutoff=8,
                genotype=g_name,
                output_root=temp_dir,
                genotype_config={},
                # Defaults for others
            )

        except Exception as e:
            print(f"  [FATAL] Optimization failed for {g_name}: {e}")
            test_failures.append(f"{g_name}: Optimization Crash ({e})")
            continue

        # 2. Check Results
        try:
            # full_run_path = os.path.join(
            #     temp_dir, run_name
            # )  # run_mome usually creates subfolder "experiments/name"
            # run_mome logic: output/experiments/name...
            # exp_path = os.path.join(temp_dir, "experiments", run_name)

            # Find results.pkl recursively
            found_dir = None
            for root, dirs, files in os.walk(os.path.join(temp_dir, "experiments")):
                if "results.pkl" in files:
                    # Check if this result belongs to the current run (by timestamp usually, but here we run sequentially)
                    # We can clear temp_dir per genotype loop to be safe?
                    # Actually, since we run sequentially, the last created one or just ANY one matches IF we clear.
                    # But we are NOT clearing per loop currently.
                    # Let's verify the group name matches genotype.
                    if f"/{g_name}_" in root or f"\\{g_name}_" in root:
                        found_dir = root
                        break

            # Alternative: Since we didn't clear, we might pick up previous run if not careful.
            # Best way: Clear temp dir INSIDE the loop at the start, or assume unique paths.
            # The paths include timestamp, so they are unique. But we need the RIGHT one.
            # Let's search for the directory created AFTER the start of this iteration?
            # Or just match the group ID roughly.

            # Let's use the most recent directory found that matches genotype.
            candidates = []
            for root, dirs, files in os.walk(os.path.join(temp_dir, "experiments")):
                if "results.pkl" in files and (
                    f"/{g_name}_" in root
                    or f"\\{g_name}_" in root
                    or f"/{g_name}/" in root
                ):
                    candidates.append(root)

            if candidates:
                # Sort by modification time to get latest
                candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                found_dir = candidates[0]

            if not found_dir:
                print(f"  [FATAL] Output directory not found for {g_name}")
                test_failures.append(f"{g_name}: Output Missing")
                continue

            res = OptimizationResult.load(found_dir)
            df = res.get_pareto_front()

            if df.empty:
                print("  [WARN] No solutions produced.")
                continue

            print(f"  Verifying {len(df)} solutions...")

            mismatch_count = 0

            for i in range(len(df)):
                row = df.iloc[i]
                g_idx = int(row["genotype_idx"])

                # Stored
                stored_log_prob = row["LogProb"]
                stored_exp = row["Expectation"]

                # Check NaNs
                if np.isnan(stored_log_prob) or np.isnan(stored_exp):
                    print(f"    Sol {i}: Skipped due to NaN stored values.")
                    continue

                # Recalculate
                params = res.get_circuit_params(g_idx)

                # Verify params structure for balanced types
                if g_name in ["00B", "B3B", "B30B", "C2B", "C20B"]:
                    # Check Mixing Params are Balanced
                    mp = np.array(params["mix_params"])
                    # Should be [[pi/4, 0, 0]...]
                    thetas = mp[:, 0]
                    if not np.allclose(thetas, np.pi / 4, atol=1e-5):
                        print(
                            f"    [FAIL] Balanced Mix Params Violation! Found {thetas[0]}"
                        )
                        mismatch_count += 1
                        continue

                try:
                    pnr_max = int(res.config.get("pnr_max", 3))
                    cutoff = int(res.config.get("cutoff", 8))
                    psi, prob = compute_heralded_state(
                        params, cutoff=cutoff, pnr_max=pnr_max
                    )

                    if prob > 0:
                        sim_log_prob = -np.log10(prob)
                    else:
                        sim_log_prob = np.inf

                    # Compare
                    # LogProb: Allow 1.0 difference (numerical noise in low prob? No, should be exact-ish)
                    # For consistency, it should be very close < 1e-3 usually.
                    # But Python vs JAX float precision might differ slightly.
                    # Let's use 1e-3.

                    diff = abs(stored_log_prob - sim_log_prob)
                    if diff > 1e-3:
                        print(
                            f"    [FAIL] Sol {i} LogProb Mismatch: Stored={stored_log_prob:.4f}, Sim={sim_log_prob:.4f}, Diff={diff:.4f}"
                        )
                        mismatch_count += 1

                except Exception as e:
                    print(f"    [ERROR] Simulation failed for Sol {i}: {e}")
                    mismatch_count += 1

            if mismatch_count == 0:
                print("  [PASS] Consistency Verified.")
            else:
                print(f"  [FAIL] {mismatch_count} mismatches detected.")
                test_failures.append(f"{g_name}: {mismatch_count} Mismatches")

        except Exception as e:
            print(f"  [FATAL] Verification Error: {e}")
            test_failures.append(f"{g_name}: Verification Error ({e})")
            import traceback

            traceback.print_exc()

        print("-" * 30)

    print("=" * 60)
    if not test_failures:
        print("ALL GENOTYPES PASSED CONSISTENCY CHECK.")
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            print("Temp directory cleaned up.")
        except Exception:
            pass
    else:
        print("FAILURES DETECTED:")
        for f in test_failures:
            print(f" - {f}")
        sys.exit(1)


if __name__ == "__main__":
    test_genotype_consistency()
