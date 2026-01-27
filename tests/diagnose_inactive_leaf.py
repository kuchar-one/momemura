"""
Diagnostic script to trace the exact issue with inactive leaf contributions.
"""

import jax
import jax.numpy as jnp
import numpy as np

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.genotypes.genotypes import get_genotype_decoder
from src.simulation.jax.runner import jax_get_heralded_state
from src.simulation.jax.composer import jax_superblock, jax_hermite_phi_matrix


def diagnose_solution(run_path: str, idx: int, cutoff: int = 15):
    """Load a specific solution and trace leaf contributions."""
    from src.utils.result_manager import AggregatedOptimizationResult

    result = AggregatedOptimizationResult.load_group(run_path)

    # Call get_pareto_front to populate _cached_valid_genotypes
    df = result.get_pareto_front()
    genotypes = np.array(result._cached_valid_genotypes)
    fitnesses = np.array(
        df[["Expectation", "LogProb", "Complexity", "TotalPhotons"]].values
    )

    g = genotypes[idx]
    print(f"=== Diagnosing Solution {idx} ===")
    print(f"Fitness: {fitnesses[idx]}")
    print()

    # Decode genotype
    config = {"modes": 3, "pnr_max": 3, "depth": 3}
    decoder = get_genotype_decoder("00B", depth=3, config=config)
    params = decoder.decode(jnp.array(g), cutoff)

    leaf_active = params["leaf_active"]
    print("=== Leaf Active Flags ===")
    for i in range(8):
        print(f"  Leaf {i}: {'ACTIVE' if leaf_active[i] else 'INACTIVE'}")
    print()

    # Get leaf params
    leaf_params = params["leaf_params"]
    print("=== Leaf PNRs (as decoded) ===")
    for i in range(8):
        n_ctrl = int(leaf_params["n_ctrl"][i])
        pnr = leaf_params["pnr"][i][:n_ctrl] if n_ctrl > 0 else []
        print(f"  Leaf {i}: n_ctrl={n_ctrl}, PNR={pnr}, active={bool(leaf_active[i])}")
    print()

    # Compute heralded states for ALL leaves
    print("=== Computing Heralded States ===")
    for i in range(8):
        vec, prob, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
            {k: v[i] for k, v in leaf_params.items()}, cutoff, pnr_max=15
        )

        # DEBUG: Print the actual leaf params being used
        n_ctrl_i = int(leaf_params["n_ctrl"][i])
        pnr_i = leaf_params["pnr"][i]
        print(
            f"    DEBUG: n_ctrl={n_ctrl_i}, raw_pnr={pnr_i[:n_ctrl_i] if n_ctrl_i > 0 else []}"
        )

        norm = jnp.linalg.norm(vec)
        # Measure non-Gaussianity: photon number variance vs coherent state
        n_op = jnp.arange(cutoff)
        probs = jnp.abs(vec) ** 2
        mean_n = jnp.sum(n_op * probs)
        var_n = jnp.sum((n_op - mean_n) ** 2 * probs)
        print(
            f"  Leaf {i}: prob={prob:.4e}, norm={norm:.4f}, <n>={mean_n:.2f}, var(n)={var_n:.2f}, active={bool(leaf_active[i])}"
        )
    print()

    # Now run the full superblock with active flags
    print("=== Running Superblock ===")
    get_heralded = jax.vmap(lambda p: jax_get_heralded_state(p, cutoff, 15))
    leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes = get_heralded(
        leaf_params
    )

    hom_x = params.get("homodyne_x", jnp.array(0.0))
    mix_params = params["mix_params"]

    hom_xs = jnp.atleast_1d(hom_x)
    phi_mat = jax_hermite_phi_matrix(hom_xs, cutoff)
    phi_vec = phi_mat[:, 0] if jnp.ndim(hom_x) == 0 else phi_mat.T

    V_matrix = jnp.zeros((cutoff, 1))
    dx_weights = jnp.zeros(1)

    final_state, _, joint_prob, is_active, max_pnr, total_sum_pnr, active_modes = (
        jax_superblock(
            leaf_vecs,
            leaf_probs,
            leaf_active,
            leaf_max_pnrs,
            leaf_total_pnrs,
            leaf_modes,
            mix_params,
            hom_x,
            0.0,
            0.0,
            phi_vec,
            V_matrix,
            dx_weights,
            cutoff,
            True,  # homodyne_window_is_none
            False,  # homodyne_x_is_none
            True,  # homodyne_resolution_is_none
        )
    )

    # Analyze output state
    probs = jnp.abs(final_state) ** 2
    n_op = jnp.arange(cutoff)
    mean_n = jnp.sum(n_op * probs)
    var_n = jnp.sum((n_op - mean_n) ** 2 * probs)

    print(f"Joint prob: {joint_prob:.4e}")
    print(f"Output <n>: {mean_n:.2f}, var(n): {var_n:.2f}")
    print(f"Max PNR: {max_pnr}, Total PNR: {total_sum_pnr}")
    print()

    # TEST: What if we set inactive leaves to vacuum BEFORE superblock?
    print("=== TEST: Replacing inactive leaf states with vacuum ===")
    vacuum = jnp.zeros(cutoff).at[0].set(1.0)
    leaf_vecs_fixed = jnp.where(leaf_active[:, None], leaf_vecs, vacuum[None, :])
    leaf_probs_fixed = jnp.where(leaf_active, leaf_probs, 1.0)

    (
        final_state_fixed,
        _,
        joint_prob_fixed,
        is_active_f,
        max_pnr_f,
        total_sum_pnr_f,
        active_modes_f,
    ) = jax_superblock(
        leaf_vecs_fixed,
        leaf_probs_fixed,
        leaf_active,
        leaf_max_pnrs,
        leaf_total_pnrs,
        leaf_modes,
        mix_params,
        hom_x,
        0.0,
        0.0,
        phi_vec,
        V_matrix,
        dx_weights,
        cutoff,
        True,
        False,
        True,
    )

    probs_f = jnp.abs(final_state_fixed) ** 2
    mean_n_f = jnp.sum(n_op * probs_f)
    var_n_f = jnp.sum((n_op - mean_n_f) ** 2 * probs_f)

    print(f"Joint prob (fixed): {joint_prob_fixed:.4e}")
    print(f"Output <n> (fixed): {mean_n_f:.2f}, var(n) (fixed): {var_n_f:.2f}")
    print()

    # Compare states
    print("=== State Comparison ===")
    diff = jnp.linalg.norm(final_state - final_state_fixed)
    print(f"||original - fixed||: {diff:.6f}")

    if diff > 0.01:
        print("** SIGNIFICANT DIFFERENCE - inactive leaves ARE contributing! **")
    else:
        print("States match - inactive leaves were already being ignored correctly.")


if __name__ == "__main__":
    # Run on the problematic example
    run_path = "output/experiments/00B_c30_a1p00_b1p41"
    idx = 9184
    diagnose_solution(run_path, idx)
