import os
import sys
import argparse
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import OptimizationResult
try:
    from src.utils.result_manager import OptimizationResult
except ImportError:
    print("Error: Could not import OptimizationResult. Run from project root.")
    sys.exit(1)


def find_result_files(root_dir: str) -> List[Path]:
    """Recursively find all results.pkl files."""
    results = []
    root = Path(root_dir)
    print(f"Scanning {root} for results.pkl...", flush=True)

    for path in root.rglob("results.pkl"):
        results.append(path)

    return results


def get_quick_stats(res: OptimizationResult) -> Dict[str, int]:
    """
    Extract stats without triggering heavy JAX compilation/slicing
    typically done in get_experiment_stats() -> get_pareto_front().
    """
    # 1. Generations & Evaluations (From History)
    n_gens = 0
    if "min_expectation" in res.history:
        # History is list of floats per chunk
        n_chunks = len(res.history["min_expectation"])
        chunk_size = int(res.config.get("chunk_size", 100))
        n_gens = n_chunks * chunk_size

    pop_size = int(res.config.get("pop_size", 1))
    n_evals = n_gens * pop_size

    # 2. Solutions (From Repertoire)
    n_solutions = 0
    if res.repertoire is not None:
        # Repertoire has .fitnesses (N_cells, Pareto_len, N_objs)
        # We just need to count how many are valid (!= -inf)
        fits = res.repertoire.fitnesses

        # Checking validity.
        # If fits is JAX array, use jnp
        # We assume fits[:, 0] checks validity

        # Flatten
        flat_fits = fits.reshape(-1, fits.shape[-1])
        valid_mask = flat_fits[:, 0] != -np.inf

        # Sum
        n_solutions = int(jnp.sum(valid_mask))

    return {
        "total_generations": n_gens,
        "total_evaluations": n_evals,
        "total_solutions": n_solutions,
    }


def count_stats(output_dir: str = "output"):
    result_files = find_result_files(output_dir)

    if not result_files:
        print("No results.pkl files found.")
        return

    print(f"Found {len(result_files)} result files. Processing...", flush=True)

    total_gens = 0
    total_evals = 0
    total_solutions = 0
    run_stats = []

    for i, res_file in enumerate(result_files):
        run_dir = res_file.parent

        # Progress
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(result_files)}] Processing {run_dir}...", flush=True)

        try:
            # Load result
            res = OptimizationResult.load(str(run_dir))

            # Use Optimized Stats Extraction
            stats = get_quick_stats(res)

            gens = stats.get("total_generations", 0)
            evals = stats.get("total_evaluations", 0)
            sols = stats.get("total_solutions", 0)

            total_gens += gens
            total_evals += evals
            total_solutions += sols

            run_stats.append(
                {
                    "path": str(run_dir.relative_to(output_dir)),
                    "gens": gens,
                    "evals": evals,
                    "sols": sols,
                }
            )

        except Exception:
            pass  # Skip corrupt/incompatible

    # Display Report
    print("\n" + "=" * 80)
    print(f"GLOBAL SIMULATION STATS (Output: {output_dir})")
    print("=" * 80)
    print(f"{'Run Path':<50} | {'Gens':<8} | {'Circuits':<10} | {'Sols':<8}")
    print("-" * 80)

    run_stats.sort(key=lambda x: x["path"])

    for stat in run_stats:
        print(
            f"{stat['path']:<50} | {stat['gens']:<8} | {stat['evals']:<10} | {stat['sols']:<8}"
        )

    print("-" * 80)
    print(f"TOTAL RUNS SCANNED:   {len(run_stats)} / {len(result_files)}")
    print(f"TOTAL GENERATIONS:    {total_gens:,}")
    print(f"TOTAL EVALUATIONS:    {total_evals:,} (Simulated Circuits)")
    print(f"TOTAL ARCHIVED SOLS:  {total_solutions:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count total simulations across output directory."
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Root output directory to scan."
    )
    args = parser.parse_args()

    count_stats(args.output)
