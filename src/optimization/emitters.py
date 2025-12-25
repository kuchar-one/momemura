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

        # 5. Split Logic (Variation vs Mutation) to match standard MixingEmitter behavior
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        key_select, key_mut, key_cross = jax.random.split(random_key, 3)

        # We need parents for Variation (2 per offspring) and Mutation (1 per offspring)
        # We sample them separately to keep logic clean and robust

        # --- A. VARIATION BATCH ---
        if n_variation > 0:
            key_select, subkey = jax.random.split(key_select)
            # Sample biased parents (2 * n_variation)
            idx_var = jax.random.categorical(subkey, logits, shape=(n_variation * 2,))
            parents_var = flat_genotypes[idx_var]

            x1 = parents_var[:n_variation]
            x2 = parents_var[n_variation:]

            # Apply Variation (Crossover)
            x_variation = self._variation_fn(x1, x2, key_cross)
        else:
            x_variation = jnp.zeros((0, flat_genotypes.shape[1]))

        # --- B. MUTATION BATCH ---
        if n_mutation > 0:
            key_select, subkey = jax.random.split(key_select)
            # Sample biased parents (n_mutation)
            idx_mut = jax.random.categorical(subkey, logits, shape=(n_mutation,))
            parents_mut = flat_genotypes[idx_mut]

            # Apply Mutation
            x_mutation = self._mutation_fn(parents_mut, key_mut)
        else:
            x_mutation = jnp.zeros((0, flat_genotypes.shape[1]))

        # --- C. CONCATENATE ---
        if n_variation == 0:
            x_new = x_mutation
        elif n_mutation == 0:
            x_new = x_variation
        else:
            x_new = jnp.concatenate([x_variation, x_mutation], axis=0)

        return x_new, emitter_state


class HybridEmitter(MixingEmitter):
    """
    Hybrid Emitter that delegates to two sub-emitters:
    1. Exploration Emitter (Standard Uniform)
    2. Intensification Emitter (Biased/Elite)

    Splits the batch size according to `intensification_ratio`.
    """

    def __init__(
        self,
        exploration_emitter,
        intensification_emitter,
        intensification_ratio=0.2,
        batch_size=32,
    ):
        # Call super to initialize MixingEmitter fields (including batch_size property)
        # We use exploration emitter's functions as placeholders
        mut_fn = getattr(exploration_emitter, "_mutation_fn", None)
        if mut_fn is None:
            mut_fn = getattr(exploration_emitter, "mutation_fn", lambda x, r: x)

        var_fn = getattr(exploration_emitter, "_variation_fn", None)
        if var_fn is None:
            var_fn = getattr(exploration_emitter, "variation_fn", lambda x1, x2, r: x1)

        super().__init__(
            mutation_fn=mut_fn,
            variation_fn=var_fn,
            variation_percentage=0.5,
            batch_size=batch_size,
        )

        self.exploration_emitter = exploration_emitter
        self.intensification_emitter = intensification_emitter
        self.intensification_ratio = intensification_ratio

        # Calculate split once (static batch size)
        self.n_intensify = int(batch_size * intensification_ratio)
        self.n_explore = batch_size - self.n_intensify

        # Ensure sub-emitters have correct batch sizes for their portion
        # We assume the user/runner constructs them, but we must override their batch_size
        # to match the split.
        # Check if they have _batch_size or batch_size attr
        if hasattr(self.exploration_emitter, "_batch_size"):
            self.exploration_emitter._batch_size = self.n_explore
        elif hasattr(self.exploration_emitter, "batch_size"):
            self.exploration_emitter.batch_size = self.n_explore

        if hasattr(self.intensification_emitter, "_batch_size"):
            self.intensification_emitter._batch_size = self.n_intensify
        elif hasattr(self.intensification_emitter, "batch_size"):
            self.intensification_emitter.batch_size = self.n_intensify

    def emit(self, repertoire, emitter_state, random_key):
        """
        Emit offspring using both strategies.
        """
        key_explore, key_intensify = jax.random.split(random_key)

        # 1. Intensification Stream
        if self.n_intensify > 0:
            x_intensify, state_intensify = self.intensification_emitter.emit(
                repertoire, emitter_state, key_intensify
            )
        else:
            # Handle 0 case to ensure correct shape/type
            # We need D dimension. Assuming we can get it from repertoire or just empty 2d array?
            # Repertoire genotypes: (Pop, D). We want (0, D).
            D = repertoire.genotypes.shape[-1]
            x_intensify = jnp.zeros((0, D))
            state_intensify = emitter_state

        # 2. Exploration Stream
        if self.n_explore > 0:
            x_explore, state_explore = self.exploration_emitter.emit(
                repertoire, emitter_state, key_explore
            )
        else:
            D = repertoire.genotypes.shape[-1]
            x_explore = jnp.zeros((0, D))
            state_explore = emitter_state

        # 3. Concatenate
        x_new = jnp.concatenate([x_intensify, x_explore], axis=0)

        # State management: MixingEmitter is usually stateless (None).
        # We return one of them or None.
        return x_new, state_explore
