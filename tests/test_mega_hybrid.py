import jax
import jax.numpy as jnp
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import emitter classes
from src.optimization.emitters import (
    HybridEmitter,
    BiasedMixingEmitter,
    MixingEmitter,
    MOMEOMGMEGAEmitter,
)


def test_mega_hybrid_composition():
    """
    Verifies that we can manually construct the Mega-Hybrid hierarchy
    and that batch sizes are distributed correctly.
    """

    def mutation_fn(x, r):
        return x

    def variation_fn(x1, x2, r):
        return x1

    total_pop = 100
    hybrid_ratio = 0.2  # 20% Gradient, 80% Inner Hybrid

    # 1. Gradient (20%)
    grad_emitter = MOMEOMGMEGAEmitter(
        mutation_fn, variation_fn, variation_percentage=0.0, batch_size=total_pop
    )

    # 2. Inner Hybrid (80%)
    n_inner = int(total_pop * (1 - hybrid_ratio))  # 80

    std_emitter = MixingEmitter(
        mutation_fn, variation_fn, variation_percentage=0.5, batch_size=n_inner
    )
    biased_emitter = BiasedMixingEmitter(
        mutation_fn,
        variation_fn,
        variation_percentage=0.5,
        batch_size=n_inner,
        temperature=5.0,
    )

    inner_hybrid = HybridEmitter(
        exploration_emitter=std_emitter,
        intensification_emitter=biased_emitter,
        intensification_ratio=0.2,  # 80/20 split inside (20% biased)
        batch_size=n_inner,
    )

    # 3. Outer Hybrid
    mega_emitter = HybridEmitter(
        exploration_emitter=inner_hybrid,
        intensification_emitter=grad_emitter,
        intensification_ratio=hybrid_ratio,
        batch_size=total_pop,
    )

    # Assertions on Batch Sizes
    # Outer split: 20 grad, 80 inner
    assert mega_emitter.n_intensify == 20
    assert mega_emitter.n_explore == 80

    # Check that sub-emitters got updated by HybridEmitter init
    assert grad_emitter.batch_size == 20
    assert inner_hybrid.batch_size == 80

    # Inner split: 16 biased, 64 standard (20% of 80 is 16)
    assert inner_hybrid.n_intensify == 16
    assert inner_hybrid.n_explore == 64

    assert biased_emitter.batch_size == 16
    assert std_emitter.batch_size == 64

    print("Mega-Hybrid Composition Verified.")


def test_mega_hybrid_emit():
    """
    Test emitting from the structure.
    """

    def mutation_fn(x, r):
        return x

    def variation_fn(x1, x2, r):
        return x1

    total_pop = 10
    # ratio 0.2 -> 2 gradient, 8 inner
    # inner 0.5 -> 4 biased, 4 standard

    grad_emitter = MOMEOMGMEGAEmitter(
        mutation_fn, variation_fn, variation_percentage=0.0, batch_size=total_pop
    )
    std_emitter = MixingEmitter(
        mutation_fn, variation_fn, variation_percentage=0.5, batch_size=total_pop
    )
    biased_emitter = BiasedMixingEmitter(
        mutation_fn, variation_fn, variation_percentage=0.5, batch_size=total_pop
    )

    inner_hybrid = HybridEmitter(std_emitter, biased_emitter, 0.2, batch_size=8)
    mega_emitter = HybridEmitter(inner_hybrid, grad_emitter, 0.2, batch_size=10)

    # Mock Repertoire
    class MockRepertoire:
        def __init__(self):
            self.genotypes = jnp.zeros((20, 5))
            self.fitnesses = jnp.zeros((20, 2))
            self.extra_scores = {"gradients": jnp.zeros((20, 5))}

        def select(self, key, batch_size, selector=None):
            # Mock select returning object with genotypes
            indices = jax.random.randint(key, (batch_size,), 0, 20)

            class Sample:
                def __init__(self, g):
                    self.genotypes = g
                    self.fitnesses = jnp.zeros((g.shape[0], 2))
                    self.descriptors = jnp.zeros((g.shape[0], 3))

            return Sample(self.genotypes[indices])

    rep = MockRepertoire()
    key = jax.random.PRNGKey(0)

    # Emit
    new_pop, _ = mega_emitter.emit(rep, None, key)

    assert new_pop.shape == (10, 5)
    print(f"Emission Shape: {new_pop.shape}")


if __name__ == "__main__":
    test_mega_hybrid_composition()
    test_mega_hybrid_emit()
