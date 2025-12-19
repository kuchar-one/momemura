import pytest
import jax
import jax.numpy as jnp
from unittest.mock import MagicMock


def test_biased_emitter_instantiation():
    try:
        from src.optimization.emitters import BiasedMixingEmitter
    except ImportError:
        pytest.skip("Emitter usage requires src/optimization/emitters.py")

    try:
        from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire  # noqa: F401
    except ImportError:
        # Mock Repertoire if qdax missing
        pass

    # Mock repertoire
    # Genotypes: (10, 5)
    # Fitnesses: (10, 4)
    # 0th obj is -ExpVal

    repertoire = MagicMock()
    repertoire.genotypes = jnp.zeros((10, 5))
    repertoire.fitnesses = jnp.zeros((10, 4))

    # Set fitness[0] -> ExpVal
    # Index 0 has -10 (Low Fitness, High ExpVal)
    # Index 1 has -1 (High Fitness, Low ExpVal)
    # We want index 1 to be picked more often.
    fitnesses = jnp.zeros((10, 4))
    fitnesses = fitnesses.at[0, 0].set(-100.0)
    fitnesses = fitnesses.at[1, 0].set(-0.1)
    # Others -inf
    fitnesses = fitnesses.at[2:, 0].set(-jnp.inf)

    repertoire.fitnesses = fitnesses
    repertoire.genotypes = jnp.arange(10 * 5).reshape(10, 5).astype(float)

    # Init Emitter
    def mut_fn(x, k):
        return x

    def var_fn(x1, x2, k):
        return x1

    emitter = BiasedMixingEmitter(
        mutation_fn=mut_fn, variation_fn=var_fn, batch_size=10, temperature=1.0
    )

    # Test emit
    key = jax.random.PRNGKey(0)
    # This calls emit -> computes logits -> samples
    # With T=1.0: indices 1 should be heavily favored over 0 (-0.1 vs -100).
    # -0.1 is exp(-0.1) ~ 0.9. -100 is exp(-100) ~ 0.
    # So we expect mostly index 1 parents.

    new_pop, _ = emitter.emit(repertoire, None, key)

    # new_pop should mostly be copies of genotype[1] (which is range 5..9)
    # genotype[0] is range 0..4

    # Check values
    # We expect heavy presence of values >= 5.
    count_high = jnp.sum(new_pop >= 5)
    count_total = new_pop.size

    assert count_high / count_total > 0.9, (
        "Biased emitter failed to prioritize high fitness."
    )


def test_pipeline_genotype_0_random():
    """Verify run_mome accepts genotype='0'."""
    import run_mome
    from unittest.mock import patch

    with patch("run_mome.plot_mome_results"):
        run_mome.run(
            mode="random",
            seed=42,
            n_iters=1,
            pop_size=2,
            cutoff=4,
            genotype="0",
            no_plot=True,
            low_mem=True,
        )
