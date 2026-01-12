import jax
import jax.numpy as jnp
import numpy as np
from src.simulation.jax.herald import jax_pure_state_amplitude
from thewalrus.symplectic import squeezing, beam_splitter, expand
from thewalrus.quantum import pure_state_amplitude


def test_recurrence_debug():
    N = 4  # Test 4D recurrence
    cutoff = 8  # Lower cutoff for speed
    hbar = 2.0
    dim = 2 * N

    # 1. Construct Physical Covariance (Squeeze + Mix)
    S = np.eye(dim)
    for i in range(N):
        # Squeeze each mode
        S1 = squeezing(0.5, 0.0)
        S = S @ expand(S1, [i], N)

    # Mix modes 0 and 1
    BS = expand(beam_splitter(np.pi / 4, 0), [0, 1], N)
    S = BS @ S

    # Vacuum covariance
    cov_vac = np.eye(dim) * hbar / 2
    cov = S @ cov_vac @ S.T

    # Add displacement
    mu = np.random.randn(dim) * 0.5

    # 2. Run JAX (Recurrence 4D)
    jax.config.update("jax_disable_jit", True)

    pnr_outcome = (1, 1, 1)  # Control modes 1, 2, 3
    # n_sig = 1 (Mode 0)

    mu_j = jnp.array(mu)
    cov_j = jnp.array(cov)

    print(f"Testing N={N}, PNR={pnr_outcome}")
    print("Running JAX...")

    norm_jax, prob_jax = jax_pure_state_amplitude(
        mu_j, cov_j, pnr_outcome, cutoff, hbar
    )
    raw_jax = norm_jax * jnp.sqrt(prob_jax)

    print(f"JAX Prob: {prob_jax}")

    # 3. Run Walrus
    print("Running Walrus...")
    walrus_amps = []
    for k in range(cutoff):
        # Pattern: [k, pnr[0], pnr[1], pnr[2]]
        pattern = [k] + list(pnr_outcome)
        amp = pure_state_amplitude(mu, cov, pattern, hbar=hbar)
        walrus_amps.append(amp)

    walrus_amps = np.array(walrus_amps)
    walrus_prob = np.sum(np.abs(walrus_amps) ** 2)

    print(f"Walrus Prob: {walrus_prob}")

    # 4. Compare
    print("\nComparison (First 5):")
    for k in range(5):
        print(f"k={k}: JAX={raw_jax[k]:.6f}, Walrus={walrus_amps[k]:.6f}")

    diff = np.abs(raw_jax - walrus_amps)
    max_diff = np.max(diff)
    print(f"\nMax Diff: {max_diff:.3e}")

    if max_diff > 1e-5:
        print("FAIL: Mismatch detected!")
    else:
        print("SUCCESS: Matches!")


if __name__ == "__main__":
    test_recurrence_debug()
