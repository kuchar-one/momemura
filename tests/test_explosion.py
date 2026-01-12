import os
import sys

# Force CPU JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.simulation.jax.runner import jax_get_heralded_state


def test_probability_explosion():
    """
    Reproduction script for Probability > 1 (LogProb << 0).
    Parameters taken from user report (Leaf 0 of the exploded solution).
    """

    r_val = [1.840133547782898, -2.3920814990997314, 1.4598146677017212]

    params = {
        "n_ctrl": jnp.array(2),
        "r": jnp.array(r_val),
        "phases": jnp.zeros(9),  # To be overridden
        # "disp": jnp.zeros(3, dtype=jnp.complex64), # To be overridden
        "pnr": jnp.array([1, 14]),
    }

    cutoff = 30

    print(
        "Testing with random phases AND LARGE DISPLACEMENT to trigger odd photon sectors..."
    )
    key = jax.random.PRNGKey(0)

    # Try multiple iterations
    for i in range(20):
        key, subkey = jax.random.split(key)
        phases = jax.random.uniform(subkey, shape=(9,), minval=0, maxval=2 * np.pi)

        # KEY: Large Displacement to mimic solution described in screenshot (disp ~ 5.5)
        disp = jnp.array([5.5 + 0j, 5.5 + 0j, 5.5 + 0j], dtype=jnp.complex64)

        # Randomize phase of displacement too?
        key, subkey = jax.random.split(key)
        disp_phase = jax.random.uniform(subkey, shape=(3,), minval=0, maxval=2 * np.pi)
        disp = disp * jnp.exp(1j * disp_phase)

        params_rand = params.copy()
        params_rand["phases"] = phases
        params_rand["disp"] = disp

        # Use valid pnr_max for PNR=[1, 14] to avoid OOB first
        try:
            vec, prob, _, _, _, _ = jax_get_heralded_state(
                params_rand, cutoff, pnr_max=15
            )
            # Iter check
            if prob > 1.1:
                print(f"!!! EXPLOSION DETECTED !!! Iter {i}: Prob={prob}")
                print(f"Params keys: {params_rand.keys()}")
                break
        except Exception as e:
            print(f"Error iter {i}: {e}")
            pass

    print("Done scanning.")


if __name__ == "__main__":
    test_probability_explosion()
