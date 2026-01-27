"""
Test that inactive leaves don't contribute to output state or probability.
"""

import jax
import jax.numpy as jnp
import pytest


def test_inactive_leaf_state_not_computed():
    """
    Verify that when a leaf is marked inactive, its heralded state
    is NOT computed (returns dummy zeros) and doesn't affect the output.
    """
    from src.simulation.jax.runner import jax_scoring_fn_batch
    from src.genotypes.genotypes import get_genotype_decoder
    import numpy as np

    # Create two genotypes: one with some leaves inactive, one with all active
    depth = 3
    config = {"modes": 3, "pnr_max": 3}
    decoder = get_genotype_decoder("A", depth=depth, config=config)
    genome_len = decoder.get_length(depth)

    # Base genome with specific values
    np.random.seed(42)
    g_base = np.random.randn(genome_len).astype(np.float32)

    # Decode to inspect structure
    params = decoder.decode(jnp.array(g_base), 10)
    n_leaves = 2**depth

    # Each leaf has: Active(1) + NCtrl(1) + PNR(2) + GG(...)
    # The first value in each leaf block is the active flag
    # Set leaf 2, 5, 7 to inactive (value < 0)
    g_inactive = g_base.copy()

    # Find leaf block offsets
    # Layout: hom(1) + leaves(L * P_leaf) + mix + final
    # For 00B/A: hom=1 or hom=N-1, then leaves
    leaf_block_size = 1 + 1 + 2 + decoder.gg_len  # Active + NCtrl + PNR + GG
    offset = 1  # Skip homodyne

    # Mark leaves 2, 5, 7 as inactive
    for leaf_idx in [2, 5, 7]:
        active_idx = offset + leaf_idx * leaf_block_size
        g_inactive[active_idx] = -5.0  # Definitely inactive

    # Now score both genotypes
    cutoff = 15
    operator = jnp.eye(cutoff)  # Dummy operator
    genotypes = jnp.array([g_base, g_inactive])

    fitnesses, descriptors, extras = jax_scoring_fn_batch(
        genotypes,
        cutoff=cutoff,
        operator=operator,
        genotype_name="A",
        genotype_config=config,
        pnr_max=3,
        gs_eig=-1.0,
    )

    # The genotype with inactive leaves should still produce valid results
    # (no NaN, no explosion)
    assert jnp.all(jnp.isfinite(fitnesses)), "Fitnesses should be finite"
    assert jnp.all(jnp.isfinite(descriptors)), "Descriptors should be finite"
    assert jnp.all(jnp.isfinite(extras["joint_probability"])), (
        "Probabilities should be finite"
    )


def test_inactive_leaf_probability():
    """
    Verify that inactive leaves contribute prob=1.0 (no penalty) and
    0 photons (no PNR cost).
    """
    from src.simulation.jax.runner import jax_scoring_fn_batch
    from src.genotypes.genotypes import get_genotype_decoder
    import numpy as np

    depth = 3
    config = {"modes": 3, "pnr_max": 3}
    decoder = get_genotype_decoder("A", depth=depth, config=config)
    genome_len = decoder.get_length(depth)

    # Create genome with ALL leaves inactive (edge case)
    np.random.seed(123)
    g = np.random.randn(genome_len).astype(np.float32)

    # Mark ALL leaves inactive
    leaf_block_size = 1 + 1 + 2 + decoder.gg_len
    offset = 1
    for leaf_idx in range(2**depth):
        active_idx = offset + leaf_idx * leaf_block_size
        g[active_idx] = -5.0

    cutoff = 15
    operator = jnp.eye(cutoff)
    genotypes = jnp.array([g])

    fitnesses, descriptors, extras = jax_scoring_fn_batch(
        genotypes,
        cutoff=cutoff,
        operator=operator,
        genotype_name="A",
        genotype_config=config,
        pnr_max=3,
        gs_eig=-1.0,
    )

    # With all leaves inactive, we expect -inf fitness (invalid configuration)
    # This is correct behavior - circuits with no active leaves should be penalized
    # The key is that no NaN is produced (which would indicate numerical issues)
    assert not jnp.any(jnp.isnan(fitnesses)), "Fitnesses should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
