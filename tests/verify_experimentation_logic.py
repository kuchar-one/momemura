import sys
import os

import numpy as np
import jax.numpy as jnp

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.simulation.jax.runner import jax_get_heralded_state
from src.simulation.jax.composer import (
    jax_compose_pair,
    jax_u_bs,
    jax_hermite_phi_matrix,
)


def test_jax_logic():
    print("Testing JAX Logic...")
    cutoff = 10

    # Mock Config A (Gaussian)
    n_c = 1
    p_a = {
        "n_control": n_c,
        "r": 1.0,
        "phase": 0.0,
        "uc_theta": [0.0],
        "uc_phi": [0.0],
        "uc_varphi": [0.0],
        "pnr": [1],  # list
    }

    # Construct leaf_params_a (as in app)
    # Ensure n_ctrl and r are 0-D arrays (Scalars)
    leaf_params_a = {
        "n_ctrl": jnp.array(n_c),  # Scalar (0-D) - FIXED
        "tmss_r": jnp.array(p_a["r"]),  # Scalar (0-D) - FIXED
        "us_phase": jnp.array([p_a["phase"]]),
        "uc_theta": jnp.array(p_a["uc_theta"]),
        "uc_phi": jnp.array(p_a["uc_phi"]),
        "uc_varphi": jnp.array(p_a["uc_varphi"]),
        "disp_s": jnp.array([0.0]),
        "disp_c": jnp.zeros(n_c),
        "pnr": jnp.array(p_a["pnr"]),  # Shape (N_C,) e.g. (1,)
    }

    print("Calls jax_get_heralded_state A...")
    vec_a_jax, prob_a_jax, _, _, _, _ = jax_get_heralded_state(
        leaf_params_a, cutoff, pnr_max=3
    )
    # Output should be (cutoff,)
    print(f"State A Shape: {vec_a_jax.shape}, Prob: {prob_a_jax}")

    # Mock Config B (Fock)
    print("Calls Fock B...")
    vec_b = np.zeros(cutoff)
    vec_b[1] = 1.0
    vec_b_jax = jnp.array(vec_b)
    prob_b = 1.0

    # Mix
    print("Mixing...")
    theta = np.pi / 4
    phi = 0.0
    hom_val = 0.0

    U = jax_u_bs(theta, phi, cutoff)
    hom_xs = jnp.atleast_1d(jnp.array(hom_val))
    phi_mat = jax_hermite_phi_matrix(hom_xs, cutoff)
    phi_vec = phi_mat[:, 0]

    # In Point Mode: homodyne_window_is_none=True.
    # homodyne_resolution_is_none controls scaling.
    # If True -> No scaling (density).
    # If False -> Scale by res.
    # We passed res=1.0 and res_none=True in previous attempt?
    # Actually if res_none is True, res arg is ignored.
    # We want density? Yes.

    out_jax, prob_meas, joint_jax = jax_compose_pair(
        vec_a_jax,
        vec_b_jax,
        U,
        prob_a_jax,
        prob_b,
        jnp.array(hom_val),
        0.0,
        1.0,
        phi_vec,
        None,
        None,
        cutoff,
        True,
        False,
        True,  # window_none=True, x_none=False, res_none=True
        theta=theta,
        phi=phi,
    )
    print(f"Output Shape: {out_jax.shape}, Prob Meas: {prob_meas}, Joint: {joint_jax}")

    print("Test Passed!")


if __name__ == "__main__":
    test_jax_logic()
