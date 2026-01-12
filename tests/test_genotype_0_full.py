import jax
import jax.numpy as jnp
from src.genotypes.genotypes import (
    Design0Genotype,
    DesignAGenotype,
    get_genotype_decoder,
)


def test_design0_decoding():
    """Verify Design0Genotype structure vs DesignA."""
    depth = 3

    # Init Decoders
    dec0 = Design0Genotype(depth=depth)
    decA = DesignAGenotype(depth=depth)

    len0 = dec0.get_length(depth)
    lenA = decA.get_length(depth)

    # Difference should be (L-1) - 1 parameters for Homodyne
    # A has 1 scalar. 0 has (L-1)=7 scalars.
    # Diff = 6.
    assert len0 == lenA + 6

    # Decode random
    key = jax.random.PRNGKey(42)
    g0 = jax.random.normal(key, (len0,))

    params0 = dec0.decode(g0, cutoff=10)

    # Check Homodyne
    hom_x = params0["homodyne_x"]
    assert hom_x.ndim == 1
    assert hom_x.shape[0] == 7  # L-1
    assert -5.1 <= jnp.min(hom_x)  # Tanh * 5
    assert jnp.max(hom_x) <= 5.1

    # Check other params structure
    assert params0["mix_params"].shape == (7, 3)
    assert params0["leaf_params"]["r"].shape == (8, 3)


def test_design0_simulation():
    """Run simulation with Design0 to verify composer handling of vector homodyne."""
    cutoff = 10
    depth = 3
    # Use config default modes=3
    dec = Design0Genotype(depth=depth)
    length = dec.get_length()

    # Create genotype with varying homodyne X
    # Set all hom_x to 0.0 except one
    g = jnp.zeros(length)

    # Decode to find which indices map to what
    # But for a simple run test, randomness is fine.
    key = jax.random.PRNGKey(123)
    g = jax.random.normal(key, (length,))

    # Run Scoring
    from src.simulation.jax.runner import _score_batch_shard

    # Genotype batch: (1, length)
    g_batch = g[None, :]

    # Dummy Operator (Vacuum projector)
    operator = jnp.zeros((cutoff, cutoff))
    operator = operator.at[0, 0].set(1.0)

    # Run
    # This invokes decoder.decode -> jax_get_heralded -> jax_superblock
    # Note: _score_batch_shard takes genotype_name to get decoder.
    fitness, descriptors, _ = _score_batch_shard(g_batch, cutoff, operator, "0")

    # Check output shapes
    # Fitness: [-exp, -logprob, -complexity, -pnrs]
    assert fitness.shape == (1, 4)
    assert descriptors.shape == (1, 3)

    # Check values sanity
    print(f"\nFitness: {fitness}")

    # Ideally, randomness yields garbage fitness but valid floats.
    assert not jnp.any(jnp.isnan(fitness))


def test_genotype_0_factory():
    d = get_genotype_decoder("0")
    assert isinstance(d, Design0Genotype)
    # design0 alias was removed or never existed, registry strictly uses "0"
    # Removing incompatible check
    # d = get_genotype_decoder("design0")
    # assert isinstance(d, Design0Genotype)
