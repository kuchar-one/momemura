"""
Test for n_ctrl=0 behavior: should return prob=1.0 for single-mode Gaussian state.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.simulation.jax.runner import jax_get_heralded_state


def test_n_ctrl_zero_returns_prob_one():
    """When n_ctrl=0, the leaf is a single-mode Gaussian with prob=1.0."""

    # Create params for a leaf with n_ctrl=0
    # 3 modes total (1 signal + 2 controls), but n_ctrl=0 means no heralding
    N = 3
    params = {
        "r": jnp.array([0.5, 0.0, 0.0]),  # Some squeezing on signal mode
        "phases": jnp.zeros(N * N),  # Identity unitary
        "disp": jnp.zeros(N, dtype=jnp.complex64),  # No displacement
        "n_ctrl": jnp.array(0),  # No control modes
        "pnr": jnp.array([0, 0]),  # PNR values (should be ignored)
    }

    cutoff = 10
    pnr_max = 3

    vec, prob, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
        params, cutoff, pnr_max
    )

    # n_ctrl=0 should give probability 1.0
    assert np.isclose(float(prob), 1.0, atol=1e-6), f"Expected prob=1.0, got {prob}"

    # max_pnr and total_pnr should be 0
    assert float(max_pnr) == 0.0, f"Expected max_pnr=0, got {max_pnr}"
    assert float(total_pnr) == 0.0, f"Expected total_pnr=0, got {total_pnr}"

    # State should be normalized
    norm = np.sum(np.abs(np.array(vec)) ** 2)
    assert np.isclose(norm, 1.0, atol=1e-6), (
        f"Expected normalized state, got norm={norm}"
    )

    print(f"PASS: n_ctrl=0 gives prob={prob}, max_pnr={max_pnr}, total_pnr={total_pnr}")


def test_n_ctrl_nonzero_returns_heralded_prob():
    """When n_ctrl>0, the leaf uses heralding and prob depends on PNR outcomes."""

    N = 3
    # Use TMSS-style squeezing between modes 0-1 and 0-2
    # For TMSS, we need phases that create correlation between signal and control
    params = {
        "r": jnp.array([0.3, 0.3, 0.0]),  # Squeezing on modes 0 and 1
        "phases": jnp.zeros(N * N),  # Identity unitary
        "disp": jnp.zeros(N, dtype=jnp.complex64),
        "n_ctrl": jnp.array(2),  # Use 2 control modes
        "pnr": jnp.array([0, 0]),  # PNR=0 outcomes (vacuum on control modes)
    }

    cutoff = 10
    pnr_max = 3

    vec, prob, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
        params, cutoff, pnr_max
    )

    # n_ctrl=2 with any PNR should have 0 < prob <= 1.0 for heralded case
    assert float(prob) <= 1.0, f"Expected prob<=1.0 for heralded, got {prob}"
    assert float(prob) > 0.0, f"Expected prob>0 for heralded, got {prob}"

    # PNR=[0,0] means max=0 and total=0
    assert float(max_pnr) == 0.0, f"Expected max_pnr=0, got {max_pnr}"
    assert float(total_pnr) == 0.0, f"Expected total_pnr=0, got {total_pnr}"

    # State should be normalized
    norm = np.sum(np.abs(np.array(vec)) ** 2)
    assert np.isclose(norm, 1.0, atol=1e-6), (
        f"Expected normalized state, got norm={norm}"
    )

    print(f"PASS: n_ctrl=2 gives prob={prob}, max_pnr={max_pnr}, total_pnr={total_pnr}")


def test_n_ctrl_one_partial_heralding():
    """When n_ctrl=1, only uses first control mode for heralding."""

    N = 3
    params = {
        "r": jnp.array([0.3, 0.3, 0.0]),
        "phases": jnp.zeros(N * N),
        "disp": jnp.zeros(N, dtype=jnp.complex64),
        "n_ctrl": jnp.array(1),  # Use only 1 control mode
        "pnr": jnp.array([0, 99]),  # Second value should be masked to 0
    }

    cutoff = 10
    pnr_max = 3

    vec, prob, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
        params, cutoff, pnr_max
    )

    # n_ctrl=1 should have heralded probability
    assert float(prob) <= 1.0, f"Expected prob<=1.0 for heralded, got {prob}"
    assert float(prob) > 0.0, f"Expected prob>0 for heralded, got {prob}"

    # Only first PNR (0) should count, second is masked
    assert float(max_pnr) == 0.0, f"Expected max_pnr=0, got {max_pnr}"
    assert float(total_pnr) == 0.0, f"Expected total_pnr=0, got {total_pnr}"

    print(f"PASS: n_ctrl=1 gives prob={prob}, max_pnr={max_pnr}, total_pnr={total_pnr}")


def test_n_ctrl_with_nonzero_pnr():
    """Test that non-zero PNR values are correctly tracked and have non-zero probability.

    Uses displacement to generate coherent states which have photon number distributions.
    """

    N = 3

    # Use displacement to create coherent states with photon population
    # Coherent state |alpha> has P(n) = exp(-|alpha|^2) * |alpha|^(2n) / n!
    # For alpha=1, P(1) ≈ 0.37, so we should see non-zero prob for PNR=1
    params = {
        "r": jnp.array([0.3, 0.3, 0.0]),  # Some squeezing
        "phases": jnp.zeros(N * N),
        "disp": jnp.array(
            [0.5 + 0j, 1.0 + 0j, 0.5 + 0j], dtype=jnp.complex64
        ),  # Displacement!
        "n_ctrl": jnp.array(2),  # Use 2 control modes
        "pnr": jnp.array([1, 0]),  # PNR=[1, 0]
    }

    cutoff = 12
    pnr_max = 3

    vec, prob, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
        params, cutoff, pnr_max
    )

    print(f"Testing displaced state with PNR=[1,0]: prob={prob}")

    # PNR=[1,0] should give max=1 and total=1
    assert float(max_pnr) == 1.0, f"Expected max_pnr=1, got {max_pnr}"
    assert float(total_pnr) == 1.0, f"Expected total_pnr=1, got {total_pnr}"

    # Should have non-zero probability
    assert float(prob) > 0, f"Expected prob>0, got {prob}"
    print(f"  -> PASS: prob={prob:.6f}")

    # Test with PNR=[1,1]
    params2 = {
        "r": jnp.array([0.5, 0.5, 0.5]),
        "phases": jnp.zeros(N * N),
        "disp": jnp.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j], dtype=jnp.complex64),
        "n_ctrl": jnp.array(2),
        "pnr": jnp.array([1, 1]),  # PNR=[1, 1]
    }

    vec2, prob2, _, max_pnr2, total_pnr2, _ = jax_get_heralded_state(
        params2, cutoff, pnr_max
    )

    print(f"Testing displaced state with PNR=[1,1]: prob={prob2}")

    # PNR=[1,1] should give max=1 and total=2
    assert float(max_pnr2) == 1.0, f"Expected max_pnr=1, got {max_pnr2}"
    assert float(total_pnr2) == 2.0, f"Expected total_pnr=2, got {total_pnr2}"

    # Should have non-zero probability
    assert float(prob2) > 0, f"Expected prob>0, got {prob2}"
    print(f"  -> PASS: prob={prob2:.6f}")

    # Test with PNR=[2,1]
    params3 = {
        "r": jnp.array([0.8, 0.8, 0.5]),
        "phases": jnp.zeros(N * N),
        "disp": jnp.array([1.5 + 0j, 2.0 + 0j, 1.0 + 0j], dtype=jnp.complex64),
        "n_ctrl": jnp.array(2),
        "pnr": jnp.array([2, 1]),  # PNR=[2, 1]
    }

    vec3, prob3, _, max_pnr3, total_pnr3, _ = jax_get_heralded_state(
        params3, cutoff, pnr_max
    )

    print(f"Testing displaced state with PNR=[2,1]: prob={prob3}")

    # PNR=[2,1] should give max=2 and total=3
    assert float(max_pnr3) == 2.0, f"Expected max_pnr=2, got {max_pnr3}"
    assert float(total_pnr3) == 3.0, f"Expected total_pnr=3, got {total_pnr3}"

    # Should have non-zero probability
    assert float(prob3) > 0, f"Expected prob>0, got {prob3}"
    print(f"  -> PASS: prob={prob3:.6f}")

    print(
        f"PASS: All non-zero PNR tests passed with probs: {prob:.4f}, {prob2:.4f}, {prob3:.4f}"
    )


if __name__ == "__main__":
    test_n_ctrl_zero_returns_prob_one()
    test_n_ctrl_nonzero_returns_heralded_prob()
    test_n_ctrl_one_partial_heralding()
    test_n_ctrl_with_nonzero_pnr()
    print("\nAll n_ctrl tests passed!")
