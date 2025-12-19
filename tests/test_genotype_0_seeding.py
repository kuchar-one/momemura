import numpy as np
import jax.numpy as jnp
from src.genotypes.converter import upgrade_genotype, create_vacuum_genotype
from src.genotypes.genotypes import (
    Design0Genotype,
    DesignAGenotype,
    get_genotype_decoder,
)


def test_upgrade_A_to_0():
    """Verify seeding Genotype 0 from Genotype A."""
    depth = 3

    # Create valid Genotype A
    decA = DesignAGenotype(depth=depth)
    lenA = decA.get_length(depth)
    gA = np.random.randn(lenA).astype(np.float32)

    # Set known hom value
    hom_val_raw = 0.5
    gA[0] = hom_val_raw

    # Convert to 0
    g0 = upgrade_genotype(gA, "A", "0", depth=depth)

    dec0 = Design0Genotype(depth=depth)
    len0 = dec0.get_length(depth)

    assert g0.shape == (len0,)

    # Check Homodyne Broadcast
    # g0 starts with 7 hom values
    hom_vec = g0[:7]
    assert np.allclose(hom_vec, hom_val_raw)

    # Verify rest is similar
    # Blocks start at 1 for A, 7 for 0.
    # Check some block values
    # A block starts at 1. 0 block starts at 7.
    assert np.allclose(gA[1:10], g0[7:16])


def test_seed_conversion_integration():
    """Simulate loading a seed and decoding it."""
    depth = 3
    gA = create_vacuum_genotype("A", depth=depth)

    # Upgrade
    g0 = upgrade_genotype(gA, "A", "0", depth=depth)

    # Decode
    dec0 = get_genotype_decoder("0", depth=depth)
    params = dec0.decode(jnp.array(g0), cutoff=10)

    # Check it works
    assert params["homodyne_x"].shape == (7,)
