import jax
import jax.numpy as jnp
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.jax.runner import jax_scoring_fn_batch
from src.genotypes.genotypes import get_genotype_decoder

# Configure JAX for 64-bit precision if needed, though usually float32 is fine
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("name", ["A", "B1", "B2", "C1", "C2", "B30B"])
def test_runner_with_genotype_types(name):
    """
    Verifies that jax_scoring_fn_batch runs with different genotype types
    and correct lengths.
    """
    cutoff = 6
    depth = 3
    # Use default modes=3
    config = {"modes": 3}
    decoder = get_genotype_decoder(name, depth=depth, config=config)
    length = decoder.get_length(depth)

    batch_size = 4
    key = jax.random.PRNGKey(42)
    # Generate random params
    genotypes = jax.random.normal(key, (batch_size, length))

    # Dummy operator (Identity)
    operator = jnp.eye(cutoff, dtype=jnp.complex128)

    # Run scoring
    fitnesses, descriptors, extras = jax_scoring_fn_batch(
        genotypes, cutoff, operator, genotype_name=name, genotype_config=config
    )

    assert fitnesses.shape == (batch_size, 4)
    assert descriptors.shape == (batch_size, 3)

    # Basic value checks
    assert not jnp.any(jnp.isnan(fitnesses))
    # Expectation logic: <psi|I|psi> = 1.0 -> fitness[0] = -1.0
    # Allow some tolerance for numerics or failed herald (0 prob)
    # If 0 prob, exp value is 0.

    print(f"Genotype {name} fitnesses:\n{fitnesses}")


if __name__ == "__main__":
    # fast run
    test_runner_with_genotype_types("A")
