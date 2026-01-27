import os
import tempfile
import pytest
import numpy as np
import sys
from pathlib import Path

# Adjust path to find src
sys.path.append(os.getcwd())

from src.utils.result_manager import OptimizationResult
from unittest.mock import patch
from run_mome import run


# Mock Repertoire (Global scope for pickling)
class MockRepertoire:
    def __init__(self, g, f):
        self.genotypes = g
        self.fitnesses = f
        self.descriptors = np.zeros((len(g), 1, 3))  # Dummy desc


def test_single_objective_end_to_end():
    """
    Verifies:
    1. Seeding pickup (creating a fake 'good' seed).
    2. Single Objective Optimization execution.
    3. Result Saving (structure matches QDax).
    4. Result Loading (OptimizationResult works).
    """

    # 1. Setup Temporary Directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_root = Path(tmp_dir)

        # --- Create a FAKE SEED ---
        geno_name = "A"
        depth = 3  # Fixed depth for JAX Superblock

        # Create a "Good" genotype (Vacuum)
        from src.genotypes.genotypes import get_genotype_decoder

        decoder = get_genotype_decoder(geno_name, depth=depth, config={"modes": 2})
        good_geno = np.zeros(decoder.get_length(depth), dtype=np.float32)

        # Create a Result containing this seed
        group_id = f"{geno_name}_c10_a1p00_b0p00"  # c10
        seed_dir = output_root / "experiments" / group_id / "seed_run"
        seed_dir.mkdir(parents=True)

        # Fitness: [Exp, LogProb, ...] -> We set Exp = -0.05 (Good!)
        f_good = np.array([[-0.05, -1.0, 0.0, 0.0]])
        g_good = good_geno[np.newaxis, :]  # (1, D)

        # Add Pareto Dim
        f_good = f_good[:, np.newaxis, :]
        g_good = g_good[:, np.newaxis, :]

        rep = MockRepertoire(g_good, f_good)

        seed_res = OptimizationResult(
            repertoire=rep,
            config={"genotype": geno_name, "modes": 2},
            centroids=np.zeros((1, 3)),
        )
        seed_res.save(str(seed_dir))

        print(f"Created fake seed in {seed_dir}")

        # --- Run Single Objective with Scanning ---
        print("Running Optimization...")
        repertoire, _, metrics = run(
            mode="single",
            backend="jax",
            genotype=geno_name,
            # depth is not a valid kwarg for run(), use config
            genotype_config={"depth": depth, "modes": 2, "pnr_max": 3},
            pop_size=4,
            n_iters=5,
            seed_scan=True,
            output_root=str(output_root),
            seed=42,
            target_alpha=1.0,
            cutoff=10,
            no_plot=True,
            debug=True,
        )

        # --- VERIFICATION 1: Seeding ---
        # The 'metrics' dict contains 'min_expectation' history.
        min_exp_start = metrics["min_expectation"][0]
        print(f"Start Expectation: {min_exp_start}")

        # It should be <= 1.8 (Vacuum is ~1.67 against GKP operator)
        assert min_exp_start < 1.8, (
            f"Seeding Failed! Start Exp {min_exp_start} is too high (Expected ~1.67 for Vacuum)"
        )

        # --- VERIFICATION 2: Saving ---
        exp_group_dir = output_root / "experiments" / group_id

        # Find the NEW run (not the seed run)
        subdirs = [
            d for d in exp_group_dir.iterdir() if d.is_dir() and d.name != "seed_run"
        ]
        assert len(subdirs) > 0, "No output directory created!"

        latest_run = subdirs[0]
        print(f"Checking output in {latest_run}")

        assert (latest_run / "results.pkl").exists(), "results.pkl missing!"
        assert (latest_run / "config.json").exists(), "config.json missing!"

        # --- VERIFICATION 3: Loading ---
        loaded_res = OptimizationResult.load(str(latest_run))
        assert loaded_res is not None
        assert loaded_res.repertoire is not None
        # Check centroids presence if my fix worked
        assert loaded_res.centroids is not None, "Centroids missing from saved result!"

        print("Verification Successful!")


# from unittest.mock import patch


def test_interrupt_saving():
    """
    Verifies that run() catches KeyboardInterrupt (triggered by SIGTERM handler)
    and saves the results correctly.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_root = Path(tmp_dir)

        # Setup
        geno_name = "B30B"
        depth = 3
        pop_size = 4

        # Determine D dynamically
        from src.genotypes.genotypes import get_genotype_decoder

        # Use config matching run call
        config = {"modes": 2, "pnr_max": 3}
        decoder = get_genotype_decoder(geno_name, depth=depth, config=config)
        D = decoder.get_length(depth)

        # Mock Return Values
        # Fitness: (Pop, 4)
        dummy_fitness = np.zeros((pop_size, 4))
        # Descriptors: (Pop, 3)
        dummy_desc = np.zeros((pop_size, 3))
        # Extras: needs 'gradients' matching (Pop, D)
        dummy_grads = np.zeros((pop_size, D))
        dummy_extras = {"gradients": dummy_grads}

        # Mock Side Effect
        # 1. Init call -> Success
        # 2. Loop Iter 0 -> Success
        # 3. Loop Iter 1 -> Interrupt

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise KeyboardInterrupt("Simulated SIGTERM/Interrupt")
            return dummy_fitness, dummy_desc, dummy_extras

        # Patch jax_scoring_fn_batch in runner module (source of import)
        with patch(
            "src.simulation.jax.runner.jax_scoring_fn_batch", side_effect=side_effect
        ):
            print("Running Interrupt Test...")
            try:
                run(
                    mode="single",
                    backend="jax",
                    genotype=geno_name,
                    genotype_config={"depth": depth, "modes": 2, "pnr_max": 3},
                    pop_size=pop_size,
                    n_iters=10,  # Enough to hit interrupt
                    seed_scan=False,
                    output_root=str(output_root),
                    seed=42,
                    target_alpha=1.0,
                    cutoff=10,
                    no_plot=True,
                    debug=True,
                )
            except KeyboardInterrupt:
                pytest.fail("KeyboardInterrupt leaked! run() should catch it.")
            except Exception as e:
                print(f"Caught unexpected exception: {e}")
                # raise e # Uncomment to debug
                pass

        # Verify Saving
        # Should be in experiments/...
        exp_dir = output_root / "experiments"
        assert exp_dir.exists()

        found_results = False
        for root, dirs, files in os.walk(exp_dir):
            if "results.pkl" in files:
                found_results = True
                print(f"Found results in {root}")
                # Optional: Load checks
                res = OptimizationResult.load(root)
                assert res is not None
                assert res.centroids is not None
                break

        assert found_results, "Did not find results.pkl after interrupt!"
        print("Interrupt Verification Successful!")


if __name__ == "__main__":
    test_single_objective_end_to_end()
    test_interrupt_saving()
