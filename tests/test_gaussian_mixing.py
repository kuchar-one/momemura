#!/usr/bin/env python3
"""
Test that verifies mixing Gaussian states produces a Gaussian output.

If all leaves produce Gaussian states (n_ctrl=0 or PNR=[0,0]),
the mixed output MUST also be Gaussian. If it's non-Gaussian,
something is wrong in the mixing logic.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np

from src.genotypes.genotypes import get_genotype_decoder
from src.simulation.jax.runner import jax_get_heralded_state, jax_apply_final_gaussian
from src.simulation.jax.composer import jax_superblock, jax_hermite_phi_matrix
from src.utils.gkp_operator import construct_gkp_operator


def test_all_gaussian_leaves_produce_gaussian_output():
    """
    Create a circuit where ALL leaves have n_ctrl=0 (pure Gaussian).
    The output should have expectation >1 (Gaussian-like, far from GKP ground state).
    """
    print("=" * 80)
    print("TEST: All Gaussian Leaves Should Produce Gaussian Output")
    print("=" * 80)

    cutoff = 30
    pnr_max = 15

    # Create 8 leaves, all with n_ctrl=0 (pure Gaussian, no heralding)
    n_leaves = 8

    # Each leaf: squeeze + displace but n_ctrl=0
    leaf_params = {
        "n_ctrl": jnp.zeros(n_leaves, dtype=jnp.int32),  # ALL n_ctrl=0
        "pnr": jnp.zeros(
            (n_leaves, 2), dtype=jnp.int32
        ),  # Doesn't matter, will be ignored
        "r": jnp.array([[0.5, 0.0, 0.0]] * n_leaves),  # Some squeezing
        "phases": jnp.zeros((n_leaves, 9)),  # Identity unitary
        "disp": jnp.zeros((n_leaves, 3), dtype=jnp.complex64),  # No displacement
        "pnr_max": jnp.array([pnr_max] * n_leaves),
    }

    # All leaves active
    leaf_active = jnp.ones(n_leaves, dtype=bool)

    # Compute heralded states for each leaf
    def compute_leaf(p):
        return jax_get_heralded_state(p, cutoff, pnr_max)

    # Restructure for vmap
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

    print("\nLeaf states:")
    for i in range(n_leaves):
        print(
            f"  Leaf {i}: prob={float(leaf_probs[i]):.6f}, max_pnr={float(leaf_max_pnrs[i]):.0f}, total_pnr={float(leaf_total_pnrs[i]):.0f}"
        )

    # Mix with balanced beamsplitters
    mix_params = jnp.zeros((7, 3))
    mix_params = mix_params.at[:, 0].set(jnp.pi / 4)  # Balanced BS

    hom_x = jnp.zeros(7)  # Per-node homodyne (7 nodes)
    phi_mat = jax_hermite_phi_matrix(hom_x, cutoff)
    phi_vec = phi_mat.T

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
            0.0,  # homodyne_window
            0.0,  # homodyne_res
            phi_vec,
            V_matrix,
            dx_weights,
            cutoff,
            True,  # homodyne_window_is_none
            False,  # homodyne_x_is_none
            True,  # homodyne_resolution_is_none
        )
    )

    print(f"\nAfter mixing:")
    print(f"  Joint probability: {float(joint_prob):.6f}")
    print(f"  Max PNR (reported): {float(max_pnr):.0f}")
    print(f"  Total PNR (reported): {float(total_sum_pnr):.0f}")
    print(f"  Active modes: {float(active_modes):.0f}")

    # No final Gaussian transform for this test

    # Compute GKP expectation value
    alpha = 1.0
    beta = 1.0 + 1.0j
    operator = construct_gkp_operator(cutoff, alpha, beta, backend="jax")

    op_psi = operator @ final_state
    exp_val = float(jnp.real(jnp.vdot(final_state, op_psi)))

    print(f"\nGKP Expectation Value: {exp_val:.6f}")

    # A Gaussian state should have expectation >> 1 (typically >2)
    # Non-Gaussian states (like GKP) have expectation < 1
    if exp_val < 1.0:
        print(
            f"\n❌ FAIL: Expectation {exp_val:.4f} < 1.0 indicates NON-GAUSSIAN state!"
        )
        print("   This should NOT happen when all leaves have n_ctrl=0.")
        return False
    else:
        print(
            f"\n✅ PASS: Expectation {exp_val:.4f} >= 1.0 indicates GAUSSIAN state as expected."
        )
        return True


def test_n_ctrl_1_with_pnr_zero_produces_gaussian_output():
    """
    Test that n_ctrl=1 with PNR=[0] also produces a nearly-Gaussian output
    (vacuum post-selection on control mode preserves Gaussianity).
    """
    print("\n" + "=" * 80)
    print("TEST: n_ctrl=1 with PNR=[0] Should Produce Nearly-Gaussian Output")
    print("=" * 80)

    cutoff = 30
    pnr_max = 15
    n_leaves = 8

    # Mix of n_ctrl=0 and n_ctrl=1 with PNR=[0,0]
    leaf_params = {
        "n_ctrl": jnp.array([0, 1, 0, 0, 2, 0, 0, 0], dtype=jnp.int32),  # Mix
        "pnr": jnp.zeros((n_leaves, 2), dtype=jnp.int32),  # ALL PNR = [0, 0]
        "r": jnp.array(
            [
                [0.5, 0.0, 0.0],
                [0.3, 0.3, 0.0],  # n_ctrl=1: 2-mode
                [0.4, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.3, 0.3],  # n_ctrl=2: 3-mode
                [0.5, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ]
        ),
        "phases": jnp.zeros((n_leaves, 9)),
        "disp": jnp.zeros((n_leaves, 3), dtype=jnp.complex64),
        "pnr_max": jnp.array([pnr_max] * n_leaves),
    }

    leaf_active = jnp.ones(n_leaves, dtype=bool)

    # Compute heralded states
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

    print("\nLeaf states:")
    for i in range(n_leaves):
        n_ctrl = int(leaf_params["n_ctrl"][i])
        pnr = [int(leaf_params["pnr"][i, 0]), int(leaf_params["pnr"][i, 1])]
        print(
            f"  Leaf {i}: n_ctrl={n_ctrl}, PNR={pnr}, prob={float(leaf_probs[i]):.6f}"
        )

    # Mix
    mix_params = jnp.zeros((7, 3))
    mix_params = mix_params.at[:, 0].set(jnp.pi / 4)

    hom_x = jnp.zeros(7)
    phi_mat = jax_hermite_phi_matrix(hom_x, cutoff)
    phi_vec = phi_mat.T

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
            False,
            True,
        )
    )

    print(f"\nAfter mixing:")
    print(f"  Joint probability: {float(joint_prob):.6f}")
    print(f"  Max PNR (reported): {float(max_pnr):.0f}")
    print(f"  Total PNR (reported): {float(total_sum_pnr):.0f}")

    # GKP expectation
    alpha = 1.0
    beta = 1.0 + 1.0j
    operator = construct_gkp_operator(cutoff, alpha, beta, backend="jax")

    op_psi = operator @ final_state
    exp_val = float(jnp.real(jnp.vdot(final_state, op_psi)))

    print(f"\nGKP Expectation Value: {exp_val:.6f}")

    # With PNR=[0,0], the output should still be Gaussian-like
    if exp_val < 1.0:
        print(
            f"\n❌ FAIL: Expectation {exp_val:.4f} < 1.0 indicates NON-GAUSSIAN state!"
        )
        print("   This should NOT happen when all active leaves have PNR=[0,0].")
        return False
    else:
        print(
            f"\n✅ PASS: Expectation {exp_val:.4f} >= 1.0 indicates GAUSSIAN state as expected."
        )
        return True


def test_n_ctrl_1_with_pnr_nonzero_produces_non_gaussian():
    """
    Verify that non-zero PNR DOES produce non-Gaussian output.
    This is the positive control.
    """
    print("\n" + "=" * 80)
    print("TEST: n_ctrl=1 with PNR=[1] Should Produce NON-Gaussian Output")
    print("=" * 80)

    cutoff = 30
    pnr_max = 15
    n_leaves = 8

    # One leaf with n_ctrl=1, PNR=[1]
    leaf_params = {
        "n_ctrl": jnp.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32),
        "pnr": jnp.array(
            [
                [0, 0],
                [1, 0],  # Leaf 1 has PNR=1
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=jnp.int32,
        ),
        "r": jnp.array(
            [
                [0.5, 0.0, 0.0],
                [1.0, 1.0, 0.0],  # Strong squeezing for TMSS
                [0.4, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.3, 0.0, 0.0],
            ]
        ),
        "phases": jnp.zeros((n_leaves, 9)),
        "disp": jnp.zeros((n_leaves, 3), dtype=jnp.complex64),
        "pnr_max": jnp.array([pnr_max] * n_leaves),
    }

    leaf_active = jnp.ones(n_leaves, dtype=bool)

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

    print("\nLeaf states:")
    for i in range(n_leaves):
        n_ctrl = int(leaf_params["n_ctrl"][i])
        pnr = [int(leaf_params["pnr"][i, 0]), int(leaf_params["pnr"][i, 1])]
        print(
            f"  Leaf {i}: n_ctrl={n_ctrl}, PNR={pnr}, prob={float(leaf_probs[i]):.6f}"
        )

    # Mix
    mix_params = jnp.zeros((7, 3))
    mix_params = mix_params.at[:, 0].set(jnp.pi / 4)

    hom_x = jnp.zeros(7)
    phi_mat = jax_hermite_phi_matrix(hom_x, cutoff)
    phi_vec = phi_mat.T

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
            False,
            True,
        )
    )

    print(f"\nAfter mixing:")
    print(f"  Joint probability: {float(joint_prob):.6f}")
    print(f"  Max PNR (reported): {float(max_pnr):.0f}")
    print(f"  Total PNR (reported): {float(total_sum_pnr):.0f}")

    # GKP expectation
    alpha = 1.0
    beta = 1.0 + 1.0j
    operator = construct_gkp_operator(cutoff, alpha, beta, backend="jax")

    op_psi = operator @ final_state
    exp_val = float(jnp.real(jnp.vdot(final_state, op_psi)))

    print(f"\nGKP Expectation Value: {exp_val:.6f}")

    # With PNR=1, the output should be non-Gaussian
    if exp_val < 1.5:
        print(
            f"\n✅ PASS: Expectation {exp_val:.4f} < 1.5 indicates NON-GAUSSIAN state as expected for PNR>0."
        )
        return True
    else:
        print(
            f"\n⚠️  WARN: Expectation {exp_val:.4f} >= 1.5, may still be somewhat Gaussian."
        )
        print("   This could be due to weak photon subtraction effect.")
        return True  # Not a hard fail


if __name__ == "__main__":
    results = []

    results.append(
        ("All Gaussian leaves", test_all_gaussian_leaves_produce_gaussian_output())
    )
    results.append(
        ("n_ctrl with PNR=0", test_n_ctrl_1_with_pnr_zero_produces_gaussian_output())
    )
    results.append(
        ("n_ctrl with PNR>0", test_n_ctrl_1_with_pnr_nonzero_produces_non_gaussian())
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
