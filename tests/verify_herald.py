import jax.numpy as jnp
from src.simulation.jax.runner import jax_get_heralded_state


def verify_pnr_mismatch():
    print("Verifying PNR Max Mismatch...")

    # Parameters approximating User's Leaf 0
    # Leaf 0 [✅ ACTIVE]: TMSS(r=1.75), n_ctrl=2, PNR=[16, 15], DispS=4.31-4.53j

    # Param construction for SINGLE instance (not vmapped)
    # n_ctrl must be scalar int
    params = {
        "tmss_r": jnp.array(1.75),  # Scalar
        "us_phase": jnp.array([0.0]),  # (1,)
        "uc_theta": jnp.array(
            [0.0]
        ),  # (1,) is enough for N_C=2 pairs? N_C=2 -> 1 pair (0,1).
        "uc_phi": jnp.array([0.0]),
        "uc_varphi": jnp.array([0.0, 0.0]),  # (2,)
        "disp_s": jnp.array([4.31 - 4.53j]),  # (1,)
        "disp_c": jnp.array([0.0, 0.0]),  # (2,)
        "pnr": jnp.array([16, 15]),  # (2,)
        "n_ctrl": jnp.array(2),  # Scalar (array(2))
    }

    # Test 1: pnr_max = 20 (Correct)
    print("\n--- Test 1: pnr_max = 20 (Correct) ---")
    vec, prob, _, _, _, _ = jax_get_heralded_state(params, cutoff=60, pnr_max=20)
    print(f"Prob (pnr_max=20): {prob}")

    # Test 2: pnr_max = 3 (Buggy Backend Default)
    print("\n--- Test 2: pnr_max = 3 (Buggy) ---")
    vec_bug, prob_bug, _, _, _, _ = jax_get_heralded_state(params, cutoff=60, pnr_max=3)
    print(f"Prob (pnr_max=3): {prob_bug}")

    print(
        "\nIf pnr_max=20 yields ~1e-4 and pnr_max=3 yields ~1e-30 or garbage, diagnosis is confirmed."
    )


if __name__ == "__main__":
    verify_pnr_mismatch()
