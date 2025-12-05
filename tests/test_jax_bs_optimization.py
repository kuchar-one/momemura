import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.circuits.jax_composer import jax_u_bs, jax_apply_bs_vec, jax_compose_pair

# Ensure we test in float32 as requested
jax.config.update("jax_enable_x64", False)


def test_jax_apply_bs_vec_correctness():
    """Verify jax_apply_bs_vec matches U @ vec."""
    cutoff = 25
    key = jax.random.PRNGKey(42)

    for _ in range(5):
        key, k1, k2, k3 = jax.random.split(key, 4)
        theta = jax.random.uniform(k1, minval=0, maxval=2 * jnp.pi)
        phi = jax.random.uniform(k2, minval=0, maxval=2 * jnp.pi)

        # Random complex vector
        vec_real = jax.random.normal(k3, (cutoff**2,))
        vec_imag = jax.random.normal(k3, (cutoff**2,))
        vec = vec_real + 1j * vec_imag
        vec = vec / jnp.linalg.norm(vec)

        # 1. Dense Matrix Approach
        U = jax_u_bs(theta, phi, cutoff)
        expected = U @ vec

        # 2. Optimized Vector Approach
        actual = jax_apply_bs_vec(vec, theta, phi, cutoff)

        # Check match
        # In float32, tolerance needs to be looser
        diff = jnp.linalg.norm(expected - actual)
        print(f"Theta={theta:.4f}, Phi={phi:.4f}, Diff={diff:.4e}")

        assert diff < 1e-4, f"Mismatch: {diff}"

    # Check small cutoff consistency
    cutoff_small = 10
    for _ in range(2):
        key, k1, k2, k3 = jax.random.split(key, 4)
        theta = jax.random.uniform(k1, minval=0, maxval=2 * jnp.pi)
        phi = jax.random.uniform(k2, minval=0, maxval=2 * jnp.pi)

        vec = jax.random.normal(k3, (cutoff_small**2,)) + 1j * jax.random.normal(
            k3, (cutoff_small**2,)
        )
        vec = vec / jnp.linalg.norm(vec)

        U = jax_u_bs(theta, phi, cutoff_small)  # Uses expm
        expected = U @ vec
        actual = jax_apply_bs_vec(
            vec, theta, phi, cutoff_small
        )  # Uses decomposed logic

        diff = jnp.linalg.norm(expected - actual)
        print(
            f"Small Cutoff ({cutoff_small}) Theta={theta:.4f}, Phi={phi:.4f}, Diff={diff:.4e}"
        )
        assert diff < 1e-3, f"Small Cutoff Mismatch: {diff}"


def test_jax_compose_pair_optimization():
    """Verify jax_compose_pair uses optimization correctly."""
    cutoff = 10
    key = jax.random.PRNGKey(123)

    theta = 0.5
    phi = 0.3

    # Random pure states
    vecA = jax.random.normal(key, (cutoff,)) + 1j * jax.random.normal(key, (cutoff,))
    vecA /= jnp.linalg.norm(vecA)

    vecB = jax.random.normal(key, (cutoff,)) + 1j * jax.random.normal(key, (cutoff,))
    vecB /= jnp.linalg.norm(vecB)

    # 1. Unoptimized (Pass U, dummy theta/phi)
    U = jax_u_bs(theta, phi, cutoff)
    # We can't easily force unoptimized path because I modified the code to ALWAYS use optimized if conditions met.
    # But we can simulate the "expected" result by manually doing the math.

    psi_in = jnp.kron(vecA, vecB)
    psi_out_expected = U @ psi_in

    # 2. Optimized (Pass theta/phi)
    # jax_compose_pair calls jax_apply_bs_vec internally
    # We need to check the output of jax_compose_pair

    # Case: Homodyne Point (returns vector)
    phi_vec = jnp.ones(cutoff) + 0j  # Dummy

    # Call jax_compose_pair
    res_vec, p_meas, joint = jax_compose_pair(
        vecA,
        vecB,
        U,
        1.0,
        1.0,
        homodyne_x=0.0,
        homodyne_window=0.0,
        homodyne_resolution=0.0,
        phi_vec=phi_vec,
        V_matrix=jnp.zeros((1, 1)),
        dx_weights=jnp.zeros(1),
        cutoff=cutoff,
        homodyne_window_is_none=True,
        homodyne_x_is_none=False,  # Point
        homodyne_resolution_is_none=True,
        theta=theta,
        phi=phi,
    )

    # Manually compute expected
    psi2d = psi_out_expected.reshape((cutoff, cutoff))
    v_expected = psi2d @ phi_vec
    p_density = jnp.real(jnp.vdot(v_expected, v_expected))
    vec_cond_expected = v_expected / jnp.sqrt(p_density)

    diff = jnp.linalg.norm(res_vec - vec_cond_expected)
    print(f"Compose Pair Diff: {diff:.4e}")
    assert diff < 1e-4


if __name__ == "__main__":
    test_jax_apply_bs_vec_correctness()
    test_jax_compose_pair_optimization()
    print("All tests passed!")
