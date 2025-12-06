import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import jax
import jax.numpy as jnp
import numpy as np
from thewalrus.quantum import pure_state_amplitude as walrus_pure_state_amplitude
from src.simulation.jax.herald import (
    jax_get_heralded_state,
    jax_pure_state_amplitude,
    vacuum_covariance,
    two_mode_squeezer_symplectic,
    expand_mode_symplectic,
    passive_unitary_to_symplectic,
    complex_alpha_to_qp,
)


def test_compare():
    jax.config.update("jax_enable_x64", True)

    # 1. Setup random Gaussian state (1 sig, 1 ctrl)
    N = 2
    hbar = 2.0

    cutoff = 5
    pnr = (1,)

    # Run multiple trials
    n_trials = 5
    ratios = []

    for i in range(n_trials):
        print(f"--- Trial {i} ---")
        # Test Vacuum
        print("--- Vacuum Test ---")
        cov = vacuum_covariance(N)
        mu = np.zeros(2 * N)

        # Walrus
        mu_np = np.array(mu)
        cov_np = np.array(cov)

        psi_walrus = np.zeros(cutoff, dtype=np.complex128)
        for n in range(cutoff):
            full_pattern = [n] + [0]  # pnr=(0,)
            amp = walrus_pure_state_amplitude(
                mu_np, cov_np, full_pattern, hbar=hbar, check_purity=False
            )
            psi_walrus[n] = amp
        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)

        # JAX
        psi_jax, prob_jax = jax_pure_state_amplitude(mu, cov, (0,), cutoff, hbar)

        print(f"Prob Walrus (Vac): {prob_walrus}")
        print(f"Prob JAX (Vac):    {prob_jax}")

        # Random symplectic
        # S = S_sq @ S_rot
        r = np.random.uniform(0.1, 1.0)
        S_sq = two_mode_squeezer_symplectic(r)

        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        t = np.cos(theta)
        r_bs = np.sin(theta)
        U = np.array([[t, -np.exp(-1j * phi) * r_bs], [np.exp(1j * phi) * r_bs, t]])
        S_rot = passive_unitary_to_symplectic(jnp.array(U))
        S = S_rot @ S_sq

        cov_vac = vacuum_covariance(N)
        cov = S @ cov_vac @ S.T

        # Random displacement
        alpha = (
            jnp.array(
                [
                    np.random.randn() + 1j * np.random.randn(),
                    np.random.randn() + 1j * np.random.randn(),
                ]
            )
            * 0.5
        )
        mu = complex_alpha_to_qp(alpha)

        # Walrus
        mu_np = np.array(mu)
        cov_np = np.array(cov)

        # Compute detQ for correlation check
        from thewalrus.quantum import Qmat, Amat

        Q_walrus = Qmat(cov_np, hbar=hbar)
        sign, logdet = np.linalg.slogdet(Q_walrus)
        detQ = np.exp(logdet)

        psi_walrus = np.zeros(cutoff, dtype=np.complex128)
        for n in range(cutoff):
            full_pattern = [n] + list(pnr)
            amp = walrus_pure_state_amplitude(
                mu_np, cov_np, full_pattern, hbar=hbar, check_purity=False
            )
            psi_walrus[n] = amp
        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)

        # JAX
        psi_jax, prob_jax = jax_pure_state_amplitude(mu, cov, pnr, cutoff, hbar)

        # Normalize Walrus (copy)
        if prob_walrus > 1e-15:
            psi_walrus_norm = psi_walrus / np.sqrt(prob_walrus)
        else:
            psi_walrus_norm = psi_walrus

        diff_psi = np.linalg.norm(psi_walrus_norm - psi_jax)
        ratio = prob_walrus / prob_jax if prob_jax > 1e-15 else 0
        ratios.append(ratio)

        print(f"Prob Walrus: {prob_walrus:.6e}")
        print(f"Prob JAX:    {prob_jax:.6e}")
        print(f"Ratio:       {ratio:.6f}")
        print(f"Diff Psi:    {diff_psi:.6e}")

    print(f"Mean Ratio: {np.mean(ratios)}")
    print(f"Std Ratio:  {np.std(ratios)}")


if __name__ == "__main__":
    test_compare()
