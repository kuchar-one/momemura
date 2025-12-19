import jax
import jax.numpy as jnp

try:
    from qdax.core.emitters.standard_emitters import MixingEmitter
except ImportError:
    # Fallback or dummy if qdax not installed (for linting/testing without env)
    class MixingEmitter:
        def __init__(self, *args, **kwargs):
            pass


class BiasedMixingEmitter(MixingEmitter):
    """
    MixingEmitter that biases parent selection towards higher fitness (lower ExpVal).
    Standard MixingEmitter samples uniformly from the repertoire.
    This emitter uses softmax selection based on the first objective (Prediction/ExpVal).
    """

    def __init__(
        self,
        mutation_fn,
        variation_fn,
        variation_percentage=1.0,
        batch_size=32,
        temperature=1.0,
        **kwargs,
    ):
        super().__init__(
            mutation_fn=mutation_fn,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=batch_size,
        )
        self._temperature = temperature

        # Dynamic Aggressiveness parameters
        # If total_bins is provided, we scale temperature based on coverage.
        # High Temp = Broad Exploration
        # Low Temp = Aggressive Exploitation (Low ExpVal)
        self._total_bins = kwargs.get("total_bins", None)
        self._base_temp = kwargs.get("base_temp", 5.0)
        self._aggressive_temp = kwargs.get("aggressive_temp", 0.05)  # Super aggressive
        self._start_pressure = kwargs.get("start_pressure_at", 0.01)  # 1% coverage
        self._full_pressure = kwargs.get("full_pressure_at", 0.25)  # 25% coverage

    def emit(self, repertoire, emitter_state, random_key):
        """
        Emit new population.
        """
        # 1. Flatten Repertoire
        flat_genotypes = repertoire.genotypes.reshape(
            -1, repertoire.genotypes.shape[-1]
        )
        flat_fitnesses = repertoire.fitnesses.reshape(
            -1, repertoire.fitnesses.shape[-1]
        )

        # 2. Identify Valid
        # MOME: invalid fitness is -inf
        # Check first objective
        obj0 = flat_fitnesses[:, 0]
        valid_mask = obj0 > -jnp.inf

        # 3. Determine Temperature
        current_temp = self._temperature  # Default static

        if self._total_bins is not None and self._total_bins > 0:
            # Count filled CENTROIDS, not solutions
            # Repertoire fitnesses shape: (Centroids, ParetoSize, Obj)
            # A centroid is filled if at least one solution exists (index 0)
            centroid_filled = repertoire.fitnesses[:, 0, 0] > -jnp.inf
            filled_count = jnp.sum(centroid_filled)
            coverage = filled_count / self._total_bins

            # Linear ramp from Base -> Aggressive
            # progress 0.0 -> Base
            # progress 1.0 -> Aggressive
            progress = (coverage - self._start_pressure) / (
                self._full_pressure - self._start_pressure
            )
            progress = jnp.clip(progress, 0.0, 1.0)

            # Logarithmic interpolation might be better for temperature?
            # Or simple linear on value.
            current_temp = (
                self._base_temp * (1.0 - progress) + self._aggressive_temp * progress
            )

            # Optional: Print/Log debug? (Cannot easily print from JIT, but emit is JIT-ed?)
            # Actually emit is usually jitted. We shouldn't print.

        # 4. Compute Logits
        # Logits = obj0 / T
        logits = jnp.where(valid_mask, obj0 / current_temp, -jnp.inf)

        # 5. Sample Parents
        n_parents = self._batch_size * 2
        key_select, key_mut, key_cross = jax.random.split(random_key, 3)

        parent_indices = jax.random.categorical(key_select, logits, shape=(n_parents,))

        parents = flat_genotypes[parent_indices]
        x1 = parents[: self._batch_size]
        x2 = parents[self._batch_size :]

        # 6. Variation & Mutation
        x_new = self._variation_fn(x1, x2, key_cross)
        x_new = self._mutation_fn(x_new, key_mut)

        return x_new, emitter_state
