#!/usr/bin/env python3
"""
Test the homodyne point post-selection effect on Gaussianity.

Hypothesis: Point homodyne projection at extreme x values might introduce
numerical artifacts that make the state appear non-Gaussian despite all
leaves being pure Gaussian states.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np

from src.simulation.jax.runner import jax_get_heralded_state
from src.simulation.jax.composer import jax_superblock, jax_hermite_phi_matrix
from src.utils.gkp_operator import construct_gkp_operator


def test_homodyne_at_various_x_positions():
    """
    Test that Gaussian states remain Gaussian after homodyne post-selection
    at different x-positions.
    """
    print("=" * 80)
    print("TEST: Homodyne Post-Selection at Various X Positions")
    print("=" * 80)

    cutoff = 30
    pnr_max = 15
    n_leaves = 8

    # All pure Gaussian leaves (n_ctrl=0)
    leaf_params = {
        "n_ctrl": jnp.zeros(n_leaves, dtype=jnp.int32),
        "pnr": jnp.zeros((n_leaves, 2), dtype=jnp.int32),
        "r": jnp.array([[0.5, 0.0, 0.0]] * n_leaves),
        "phases": jnp.zeros((n_leaves, 9)),
        "disp": jnp.zeros((n_leaves, 3), dtype=jnp.complex64),
    }

    leaf_active = jnp.ones(n_leaves, dtype=bool)

    # Compute leaf states
    leaf_params_vmappable = {
        "n_ctrl": leaf_params["n_ctrl"],
        "pnr": leaf_params["pnr"],
        "r": leaf_params["r"],
        "phases": leaf_params["phases"],
        "disp": leaf_params["disp"],
    }

    get_heralded = jax.vmap(lambda p: jax_get_heralded_state(p, cutoff, pnr_max))
    (leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes) = (
        get_heralded(leaf_params_vmappable)
    )

    # GKP operator
    alpha = 1.0
    beta = 1.0 + 1.0j
    operator = construct_gkp_operator(cutoff, alpha, beta, backend="jax")

    # Test various homodyne x positions
    x_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0]

    print("\nHomodyne X | Expectation | Assessment")
    print("-" * 45)

    results = []
    for x in x_values:
        # Create phi vectors for all 7 nodes with same x
        hom_x = jnp.full(7, x)
        phi_mat = jax_hermite_phi_matrix(hom_x, cutoff)
        phi_vec = phi_mat.T

        # Balanced BS mixing
        mix_params = jnp.zeros((7, 3))
        mix_params = mix_params.at[:, 0].set(jnp.pi / 4)

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
                False,  # homodyne_x_is_none (NOT None, we have x values)
                True,  # homodyne_resolution_is_none
            )
        )

        # Compute expectation
        op_psi = operator @ final_state
        exp_val = float(jnp.real(jnp.vdot(final_state, op_psi)))

        if exp_val < 1.0:
            assessment = "❌ NON-GAUSSIAN"
        else:
            assessment = "✅ Gaussian"

        print(f"x = {x:5.1f}   | {exp_val:11.4f} | {assessment}")
        results.append((x, exp_val))

    # Check if any position produces non-Gaussian output
    non_gaussian_count = sum(1 for _, e in results if e < 1.0)

    print("\n" + "=" * 80)
    if non_gaussian_count > 0:
        print(
            f"RESULT: {non_gaussian_count}/{len(results)} positions produced non-Gaussian output!"
        )
        print("This confirms homodyne post-selection introduces non-Gaussianity.")
        return False
    else:
        print("RESULT: All positions produced Gaussian output.")
        return True


def test_specific_solution_homodyne_values():
    """
    Test with the actual homodyne x values from solution 3711.
    """
    print("\n" + "=" * 80)
    print("TEST: Using Actual Homodyne X Values from Solution 3711")
    print("=" * 80)

    # These would need to be extracted from the actual genotype
    # For now, let's test with typical optimization values (around ±4)

    cutoff = 30
    pnr_max = 15
    n_leaves = 8

    # Same Gaussian leaves
    leaf_params = {
        "n_ctrl": jnp.zeros(n_leaves, dtype=jnp.int32),
        "pnr": jnp.zeros((n_leaves, 2), dtype=jnp.int32),
        "r": jnp.array([[0.5, 0.0, 0.0]] * n_leaves),
        "phases": jnp.zeros((n_leaves, 9)),
        "disp": jnp.zeros((n_leaves, 3), dtype=jnp.complex64),
    }

    leaf_active = jnp.ones(n_leaves, dtype=bool)

    get_heralded = jax.vmap(lambda p: jax_get_heralded_state(p, cutoff, pnr_max))
    (leaf_vecs, leaf_probs, _, leaf_max_pnrs, leaf_total_pnrs, leaf_modes) = (
        get_heralded(leaf_params)
    )

    # GKP operator
    alpha = 1.0
    beta = 1.0 + 1.0j
    operator = construct_gkp_operator(cutoff, alpha, beta, backend="jax")

    # Test with different homodyne patterns
    hom_patterns = [
        ("All zero", jnp.zeros(7)),
        ("All 1.0", jnp.ones(7)),
        ("Mixed ±1", jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])),
        ("Sequential", jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])),
        ("Large positive", jnp.ones(7) * 4.0),
        ("Large negative", jnp.ones(7) * -4.0),
        ("Extreme", jnp.ones(7) * 8.0),
    ]

    print("\nPattern           | Expectation | Assessment")
    print("-" * 55)

    for name, hom_x in hom_patterns:
        phi_mat = jax_hermite_phi_matrix(hom_x, cutoff)
        phi_vec = phi_mat.T

        mix_params = jnp.zeros((7, 3))
        mix_params = mix_params.at[:, 0].set(jnp.pi / 4)

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
                True,
                False,  # homodyne_x_is_none = False
                True,
            )
        )

        op_psi = operator @ final_state
        exp_val = float(jnp.real(jnp.vdot(final_state, op_psi)))
        prob = float(joint_prob)

        if exp_val < 1.0:
            assessment = f"❌ NON-GAUSSIAN (prob={prob:.6f})"
        else:
            assessment = f"✅ Gaussian (prob={prob:.6f})"

        print(f"{name:17} | {exp_val:11.4f} | {assessment}")

    return True


def test_hermite_function_conditioning():
    """
    Check if Hermite functions become poorly conditioned at extreme x values.
    """
    print("\n" + "=" * 80)
    print("TEST: Hermite Function Conditioning at Various X Values")
    print("=" * 80)

    cutoff = 30

    x_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    print("\nX Value | Max |phi_n| | Min |phi_n| | phi Norm | Assessment")
    print("-" * 70)

    for x in x_values:
        xs = jnp.array([x])
        phi_mat = jax_hermite_phi_matrix(xs, cutoff)
        phi_vec = phi_mat[:, 0]  # Shape (cutoff,)

        max_phi = float(jnp.max(jnp.abs(phi_vec)))
        min_phi = float(jnp.min(jnp.abs(phi_vec)))
        norm = float(jnp.linalg.norm(phi_vec))

        if max_phi > 1e10 or min_phi < 1e-20:
            assessment = "⚠️ NUMERICAL ISSUE"
        else:
            assessment = "✅ OK"

        print(
            f"x = {x:3.1f}  | {max_phi:10.3e} | {min_phi:10.3e} | {norm:8.3e} | {assessment}"
        )

    return True


if __name__ == "__main__":
    test_hermite_function_conditioning()
    test_homodyne_at_various_x_positions()
    test_specific_solution_homodyne_values()
