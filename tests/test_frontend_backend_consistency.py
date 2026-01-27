"""
Test for frontend-backend consistency after n_ctrl fix.

Verifies that:
1. Frontend compute_state_with_jax uses the same logic as backend
2. n_ctrl=0 returns prob=1.0 in frontend simulation
"""

import numpy as np
import jax.numpy as jnp
import sys
import os

# Ensure module imports work
sys.path.insert(0, os.getcwd())

from src.simulation.jax.runner import jax_get_heralded_state
from frontend.utils import compute_state_with_jax, JAX_AVAILABLE


def test_frontend_backend_consistency_n_ctrl_zero():
    """Test that frontend and backend produce same result for n_ctrl=0."""
    if not JAX_AVAILABLE:
        print("SKIP: JAX not available")
        return

    # Build params structure as frontend expects
    N = 3
    L = 8  # Number of leaves

    leaf_params = {
        "r": np.zeros((L, N), dtype=np.float32),
        "phases": np.zeros((L, N * N), dtype=np.float32),
        "disp": np.zeros((L, N), dtype=np.complex64),
        "n_ctrl": np.zeros(L, dtype=np.int32),  # All n_ctrl=0
        "pnr": np.zeros((L, N - 1), dtype=np.int32),
    }

    # Set some squeezing on first leaf
    leaf_params["r"][0, 0] = 0.5

    params = {
        "leaf_params": leaf_params,
        "leaf_active": np.array([True] + [False] * 7),
        "mix_params": np.zeros((7, 3), dtype=np.float32),
        "homodyne_x": 0.0,
        "homodyne_window": 0.0,
        "final_gauss": {"r": 0.0, "phi": 0.0, "varphi": 0.0, "disp": 0.0 + 0j},
    }

    cutoff = 15

    # Direct backend call for leaf 0
    backend_params = {
        "r": jnp.array(leaf_params["r"][0]),
        "phases": jnp.array(leaf_params["phases"][0]),
        "disp": jnp.array(leaf_params["disp"][0]),
        "n_ctrl": jnp.array(0),
        "pnr": jnp.array(leaf_params["pnr"][0]),
    }

    vec_backend, prob_backend, _, _, _, _ = jax_get_heralded_state(
        backend_params, cutoff, pnr_max=3
    )

    # n_ctrl=0 should give prob=1.0
    assert np.isclose(float(prob_backend), 1.0, atol=1e-6), (
        f"Backend prob should be 1.0 for n_ctrl=0, got {prob_backend}"
    )

    print(f"Backend: prob={prob_backend} ✓")

    # Frontend simulation (full pipeline)
    psi_frontend, prob_frontend = compute_state_with_jax(
        params, cutoff=cutoff, pnr_max=3
    )

    # With only one active leaf at n_ctrl=0, prob should be 1.0
    assert np.isclose(float(prob_frontend), 1.0, atol=1e-6), (
        f"Frontend prob should be 1.0 for n_ctrl=0, got {prob_frontend}"
    )

    print(f"Frontend: prob={prob_frontend} ✓")

    # States should be normalized
    assert np.isclose(np.sum(np.abs(psi_frontend) ** 2), 1.0, atol=1e-6), (
        "Frontend state should be normalized"
    )

    print("PASS: Frontend-backend consistency verified for n_ctrl=0")


def test_frontend_backend_consistency_n_ctrl_nonzero():
    """Test that frontend and backend produce consistent results for n_ctrl>0."""
    if not JAX_AVAILABLE:
        print("SKIP: JAX not available")
        return

    N = 3
    L = 8

    leaf_params = {
        "r": np.zeros((L, N), dtype=np.float32),
        "phases": np.zeros((L, N * N), dtype=np.float32),
        "disp": np.zeros((L, N), dtype=np.complex64),
        "n_ctrl": np.zeros(L, dtype=np.int32),
        "pnr": np.zeros((L, N - 1), dtype=np.int32),
    }

    # First leaf: n_ctrl=2, PNR=[0,0], with displacement
    leaf_params["r"][0] = [0.3, 0.3, 0.0]
    leaf_params["n_ctrl"][0] = 2
    leaf_params["pnr"][0] = [0, 0]
    leaf_params["disp"][0] = [1.0 + 0j, 0.5 + 0j, 0.0 + 0j]

    params = {
        "leaf_params": leaf_params,
        "leaf_active": np.array([True] + [False] * 7),
        "mix_params": np.zeros((7, 3), dtype=np.float32),
        "homodyne_x": 0.0,
        "homodyne_window": 0.0,
        "final_gauss": {"r": 0.0, "phi": 0.0, "varphi": 0.0, "disp": 0.0 + 0j},
    }

    cutoff = 10

    # Backend
    backend_params = {
        "r": jnp.array(leaf_params["r"][0]),
        "phases": jnp.array(leaf_params["phases"][0]),
        "disp": jnp.array(leaf_params["disp"][0]),
        "n_ctrl": jnp.array(2),
        "pnr": jnp.array(leaf_params["pnr"][0]),
    }

    _, prob_backend, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
        backend_params, cutoff, pnr_max=3
    )

    # n_ctrl=2 should give heralded prob < 1.0
    assert float(prob_backend) < 1.0, (
        f"Backend prob should be <1.0 for heralded, got {prob_backend}"
    )
    assert float(prob_backend) > 0.0, f"Backend prob should be >0.0, got {prob_backend}"

    print(f"Backend: prob={prob_backend}, max_pnr={max_pnr}, total_pnr={total_pnr} ✓")

    # Frontend
    psi_frontend, prob_frontend = compute_state_with_jax(
        params, cutoff=cutoff, pnr_max=3
    )

    # Probabilities should match closely (small differences from pipeline stages)
    assert np.isclose(float(prob_frontend), float(prob_backend), rtol=1e-2), (
        f"Frontend ({prob_frontend}) and backend ({prob_backend}) probs should match within 1%"
    )

    print(f"Frontend: prob={prob_frontend} ✓")
    print("PASS: Frontend-backend consistency verified for n_ctrl>0")


if __name__ == "__main__":
    test_frontend_backend_consistency_n_ctrl_zero()
    test_frontend_backend_consistency_n_ctrl_nonzero()
    print("\nAll frontend-backend consistency tests passed!")
