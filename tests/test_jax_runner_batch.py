import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
from src.simulation.jax.runner import jax_scoring_fn_batch


def test_jax_scoring_fn_batch_shapes():
    """
    Verifies that jax_scoring_fn_batch returns correct shapes
    and handles batch sizes not divisible by device count (via internal padding).
    """
    jax.config.update("jax_enable_x64", True)
    cutoff = 6
    batch_size = 13  # Prime number, likely not divisible by n_devices
    genotype_dim = 256

    # 1. Random Genotypes
    key = jax.random.PRNGKey(0)
    genotypes = jax.random.normal(key, (batch_size, genotype_dim))

    # 2. Operator (Dummy)
    # Operator size is (N, N) where N=cutoff? No, operator is used for expectation.
    # In run_mome.py:
    # operator = construct_gkp_operator(cutoff, ...)
    # It is usually (cutoff, cutoff) for single mode?
    # Or (cutoff^N, cutoff^N)?
    # In adapter, operator is passed to evaluate_one.
    # evaluate_one computes:
    # if final_state.ndim == 1: vdot(final_state, operator @ final_state)
    # final_state is (cutoff,) vector (since we use pure pipeline).
    # So operator must be (cutoff, cutoff).

    operator = jnp.eye(cutoff, dtype=jnp.complex128)

    # 3. Run Batch Scoring
    fitnesses, descriptors = jax_scoring_fn_batch(genotypes, cutoff, operator)

    # 4. Verify Shapes
    # fitnesses: (batch_size, 4)
    # descriptors: (batch_size, 3)

    print(f"Fitnesses shape: {fitnesses.shape}")
    print(f"Descriptors shape: {descriptors.shape}")

    assert fitnesses.shape == (batch_size, 4)
    assert descriptors.shape == (batch_size, 3)

    # Check for NaNs
    assert not jnp.any(jnp.isnan(fitnesses))
    assert not jnp.any(jnp.isnan(descriptors))

    # Check that fitness values are reasonable
    # Expectation of Identity should be <psi|I|psi> = 1.0 (since normalized)
    # fitness[0] is -expectation. So should be approx -1.0.
    # (Allowing for some numerical noise or if state is zero?)
    # Herald returns normalized state (or zero if prob=0).
    # If prob=0, fitness is NaN or handled?
    # In jax_runner, checking prob_clipped:
    # prob_clipped = max(joint_prob, 1e-30)
    # log_prob = -log10(prob_clipped)
    # If joint_prob is 0, exp_val is 0.

    # We can check specific values if desired, but shape is main concern for pmap/padding.


if __name__ == "__main__":
    test_jax_scoring_fn_batch_shapes()
