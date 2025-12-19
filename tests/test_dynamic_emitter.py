import pytest
import jax
import jax.numpy as jnp
from unittest.mock import MagicMock


def test_dynamic_emitter_logic():
    try:
        from src.optimization.emitters import BiasedMixingEmitter
    except ImportError:
        pytest.skip("Emitter usage requires src/optimization/emitters.py")

    # Mock repertoire
    # Genotypes: (5, 1, 5) -> D=5
    # Fitnesses: (5, 1, 4) -> 0th obj ExpVal
    # We set max_bins = 10

    # 5 Centroids.
    # Case 1: Low Coverage (1 centroid filled). Coverage = 1/10 = 10%.
    # Should be mostly High Temp (Base=5.0).
    # Scores: Index 0: -0.1 (High Fit), Index 1: -100.0 (Low Fit).
    # Logic:
    # 0.1 < 5.0 (Base Temp).
    # exp(-0.1/5.0) ~ exp(-0.02) ~ 0.98.
    # exp(-100/5.0) ~ exp(-20) ~ 1e-9.
    # Wait, even with Temp=5.0, -0.1 vs -100 is huge diff.
    # I need closer fitness values to distinguish Soft vs Hard.

    # Let's say fitnesses are -1.0 and -2.0.
    # Temp=5.0: exp(-0.2) vs exp(-0.4) -> 0.81 vs 0.67. (Ratios ~ 1.2x).
    # Temp=0.1: exp(-10) vs exp(-20) -> 4e-5 vs 2e-9. (Ratios ~ 20000x).

    repertoire = MagicMock()
    # 5 Centroids
    g_arr = jnp.zeros((5, 1, 5))
    f_arr = jnp.zeros((5, 1, 4)) - jnp.inf  # Default empty

    # Fill idx 0 and 1
    f_arr = f_arr.at[0, 0, 0].set(-1.0)  # Better
    f_arr = f_arr.at[1, 0, 0].set(-2.0)  # Worse

    repertoire.genotypes = g_arr
    repertoire.fitnesses = f_arr

    def mut_fn(x, k):
        return x

    def var_fn(x1, x2, k):
        return x1

    # Start Pressure at 20% (0.2). Full at 40% (0.4).
    # Base Temp 5.0, Aggressive 0.1.
    emitter = BiasedMixingEmitter(
        mut_fn,
        var_fn,
        batch_size=1000,
        total_bins=10,
        base_temp=5.0,
        aggressive_temp=0.1,
        start_pressure_at=0.2,
        full_pressure_at=0.4,
    )

    key = jax.random.PRNGKey(0)

    # Case A: Coverage = 2 filled / 10 = 20%. Matches start pressure.
    # Progress = 0. Temp = Base = 5.0.
    # Ratio of picking Idx 0 vs Idx 1 should be roughly exp(-1/5) / exp(-2/5) = exp(0.2) ~ 1.22.

    x_new, _ = emitter.emit(repertoire, None, key)
    # x_new comes from x1, x2. x1 are parents.
    # We can't see parent indices directly, but x_new depends on them.
    # Since var/mut are identity, x_new IS parents.
    # We trace back logic.
    # But wait, genotypes are 0s. I need distinct genotypes to Identify.

    g_arr = g_arr.at[0, 0, 0].set(10.0)  # Idx 0 -> 10
    g_arr = g_arr.at[1, 0, 0].set(20.0)  # Idx 1 -> 20
    repertoire.genotypes = g_arr

    x_new, _ = emitter.emit(repertoire, None, key)

    # Count 10s and 20s
    count_10 = jnp.sum(x_new[..., 0] == 10.0)
    count_20 = jnp.sum(x_new[..., 0] == 20.0)

    ratio = count_10 / (count_20 + 1e-9)
    print(f"Low Coverage Ratio (Expected ~1.22): {ratio}")

    # assert ratio < 2.0 (High Temp behavior)
    assert ratio < 2.5, f"Expected Ratio ~1.22, got {ratio}. Temp too low?"
    assert ratio > 0.8, "Something wrong with sampling."

    # Case B: High Coverage. Fill more.
    # Fill idx 2,3,4. Total 5/10 = 50%. > Full Pressure (40%).
    # Temp = Aggressive = 0.1.
    # Ratio: exp(-1/0.1) / exp(-2/0.1) = exp(-10) / exp(-20) = exp(10) ~ 22026.
    # So we should see almost exclusively Idx 0 (10.0).

    f_arr = f_arr.at[2:5, 0, 0].set(-5.0)  # Even worse
    g_arr = g_arr.at[2:5, 0, 0].set(30.0)  # Distinct
    repertoire.fitnesses = f_arr
    repertoire.genotypes = g_arr

    x_new, _ = emitter.emit(repertoire, None, key)

    count_10 = jnp.sum(x_new[..., 0] == 10.0)
    count_20 = jnp.sum(x_new[..., 0] == 20.0)

    # Idx 0 (-1.0) vs Idx 1 (-2.0).
    # Ratio should be HUGE.

    print(f"High Coverage counts: 10={count_10}, 20={count_20}")

    if count_20 == 0:
        ratio = 99999.0
    else:
        ratio = count_10 / count_20

    assert ratio > 10.0, f"Expected Huge Ratio, got {ratio}. Temp too high?"
