"""
Consistency test between JAX implementation and TheWalrus.
Tests 1-mode and 2-mode Gaussian circuits with PNR post-selection.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from thewalrus.quantum import pure_state_amplitude as walrus_pure_state_amplitude
from thewalrus.quantum import Qmat
from src.simulation.jax.herald import (
    jax_pure_state_amplitude,
    jax_get_full_amplitudes,
    vacuum_covariance,
    two_mode_squeezer_symplectic,
    passive_unitary_to_symplectic,
    complex_alpha_to_qp,
)


# Enable float64 for better precision
jax.config.update("jax_enable_x64", True)

HBAR = 2.0
CUTOFF = 15


def single_mode_squeezer_symplectic(r: float) -> np.ndarray:
    """
    Symplectic matrix for single-mode squeezing.
    S = [[exp(-r), 0], [0, exp(r)]] in xp ordering.
    """
    return np.diag([np.exp(-r), np.exp(r)])


def single_mode_rotation_symplectic(theta: float) -> np.ndarray:
    """
    Symplectic matrix for single-mode phase rotation.
    [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]] in xp ordering.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


class Test1ModeGaussian:
    """Test 1-mode Gaussian states (no heralding, prob=1.0)."""

    def test_vacuum_1mode(self):
        """1-mode vacuum state should have prob=1.0 and all amplitude in |0>."""
        N = 1
        cov = vacuum_covariance(N, HBAR)
        mu = np.zeros(2 * N)

        # TheWalrus: compute amplitudes for each Fock state
        psi_walrus = np.zeros(CUTOFF, dtype=np.complex128)
        for n in range(CUTOFF):
            amp = walrus_pure_state_amplitude(
                mu, np.array(cov), [n], hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp

        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)

        print(f"\n1-mode vacuum:")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  psi[0:5]: {np.abs(psi_walrus[:5]) ** 2}")

        assert np.isclose(prob_walrus, 1.0, atol=1e-6), (
            f"Vacuum prob should be 1.0, got {prob_walrus}"
        )
        assert np.abs(psi_walrus[0]) ** 2 > 0.99, "Vacuum should have >99% in |0>"

    def test_squeezed_1mode(self):
        """1-mode squeezed vacuum: prob=1.0, even photon distribution."""
        N = 1
        r = 0.5

        S = single_mode_squeezer_symplectic(r)
        cov_vac = vacuum_covariance(N, HBAR)
        cov = S @ cov_vac @ S.T
        mu = np.zeros(2 * N)

        # TheWalrus
        psi_walrus = np.zeros(CUTOFF, dtype=np.complex128)
        for n in range(CUTOFF):
            amp = walrus_pure_state_amplitude(
                mu, np.array(cov), [n], hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp

        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)

        print(f"\n1-mode squeezed (r={r}):")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  |psi[0]|^2: {np.abs(psi_walrus[0]) ** 2:.4f}")
        print(f"  |psi[2]|^2: {np.abs(psi_walrus[2]) ** 2:.4f}")
        print(f"  |psi[1]|^2: {np.abs(psi_walrus[1]) ** 2:.4f} (should be ~0)")

        assert np.isclose(prob_walrus, 1.0, atol=1e-4), (
            f"Squeezed prob should be 1.0, got {prob_walrus}"
        )
        # Squeezed vacuum has only even photon states
        assert np.abs(psi_walrus[1]) ** 2 < 1e-10, "Odd photons should be 0"

    def test_displaced_1mode(self):
        """1-mode coherent state: prob=1.0, Poissonian distribution."""
        N = 1
        alpha = 1.5 + 0.3j

        cov = vacuum_covariance(N, HBAR)
        mu = complex_alpha_to_qp(jnp.array([alpha]), HBAR)

        # TheWalrus
        psi_walrus = np.zeros(CUTOFF, dtype=np.complex128)
        for n in range(CUTOFF):
            amp = walrus_pure_state_amplitude(
                np.array(mu), np.array(cov), [n], hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp

        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)

        print(f"\n1-mode displaced (alpha={alpha}):")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  |alpha|^2 = {np.abs(alpha) ** 2:.2f} (mean photon)")

        assert np.isclose(prob_walrus, 1.0, atol=1e-4), (
            f"Coherent prob should be 1.0, got {prob_walrus}"
        )

    def test_general_gaussian_1mode(self):
        """1-mode general Gaussian (squeezed + rotated + displaced): prob~1.0."""
        N = 1
        r = 0.5  # Lower squeezing for less probability leak
        theta = np.pi / 6  # Rotation angle
        alpha = 0.5 + 0.3j  # Lower displacement

        S_sq = single_mode_squeezer_symplectic(r)
        S_rot = single_mode_rotation_symplectic(theta)
        S = S_rot @ S_sq

        cov_vac = vacuum_covariance(N, HBAR)
        cov = S @ cov_vac @ S.T
        mu = np.array(complex_alpha_to_qp(jnp.array([alpha]), HBAR))

        # Use larger cutoff for high-photon states
        large_cutoff = 25

        # TheWalrus
        psi_walrus = np.zeros(large_cutoff, dtype=np.complex128)
        for n in range(large_cutoff):
            amp = walrus_pure_state_amplitude(
                mu, cov, [n], hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp

        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)

        print(f"\n1-mode general Gaussian (r={r}, theta={theta:.2f}, alpha={alpha}):")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")

        # With adequate cutoff, prob should be very close to 1.0
        assert np.isclose(prob_walrus, 1.0, atol=1e-3), (
            f"General Gaussian prob should be ~1.0, got {prob_walrus}"
        )


class Test2ModeWithHeralding:
    """Test 2-mode Gaussian with PNR heralding on 1 control mode."""

    def test_2mode_herald_vacuum(self):
        """2-mode vacuum, herald on PNR=0: should get vacuum back, prob=1."""
        N = 2
        cov = vacuum_covariance(N, HBAR)
        mu = np.zeros(2 * N)
        pnr = (0,)

        # TheWalrus
        psi_walrus = np.zeros(CUTOFF, dtype=np.complex128)
        for n in range(CUTOFF):
            full_pattern = [n] + list(pnr)
            amp = walrus_pure_state_amplitude(
                mu, np.array(cov), full_pattern, hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp
        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)

        # JAX
        jax_cov = jnp.array(cov)
        jax_mu = jnp.array(mu)
        psi_jax, prob_jax = jax_pure_state_amplitude(jax_mu, jax_cov, pnr, CUTOFF, HBAR)

        print(f"\n2-mode vacuum, PNR=0:")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  Prob JAX:       {float(prob_jax):.6e}")

        assert np.isclose(prob_walrus, prob_jax, rtol=1e-4), (
            f"Prob mismatch: Walrus={prob_walrus:.6e}, JAX={float(prob_jax):.6e}"
        )
        assert np.isclose(prob_walrus, 1.0, atol=1e-6), (
            "Vacuum herald should give prob=1"
        )

    def test_2mode_tmss_herald(self):
        """2-mode TMSS heralded on PNR=1: should give single-photon state."""
        N = 2
        r = 0.8

        S = two_mode_squeezer_symplectic(r)
        cov_vac = vacuum_covariance(N, HBAR)
        cov = S @ np.array(cov_vac) @ S.T
        mu = np.zeros(2 * N)
        pnr = (1,)  # Herald on 1 photon

        # TheWalrus
        psi_walrus = np.zeros(CUTOFF, dtype=np.complex128)
        for n in range(CUTOFF):
            full_pattern = [n] + list(pnr)
            amp = walrus_pure_state_amplitude(
                mu, cov, full_pattern, hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp
        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)
        psi_walrus_norm = (
            psi_walrus / np.sqrt(prob_walrus) if prob_walrus > 1e-15 else psi_walrus
        )

        # JAX
        jax_cov = jnp.array(cov)
        jax_mu = jnp.array(mu)
        psi_jax, prob_jax = jax_pure_state_amplitude(jax_mu, jax_cov, pnr, CUTOFF, HBAR)

        diff_psi = np.linalg.norm(np.array(psi_jax) - psi_walrus_norm)

        print(f"\n2-mode TMSS (r={r}), PNR=1:")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  Prob JAX:       {float(prob_jax):.6e}")
        print(f"  |psi - psi_walrus|: {diff_psi:.6e}")
        print(f"  |psi[1]|^2: {np.abs(psi_walrus_norm[1]) ** 2:.4f} (should be ~1)")

        assert np.isclose(prob_walrus, prob_jax, rtol=1e-4), (
            f"Prob mismatch: Walrus={prob_walrus:.6e}, JAX={float(prob_jax):.6e}"
        )
        assert diff_psi < 1e-6, f"State vector mismatch: {diff_psi}"
        # TMSS heralded on PNR=1 gives single photon
        assert np.abs(psi_walrus_norm[1]) ** 2 > 0.9, "Heralded |1> should dominate"

    def test_2mode_random_gaussian_consistency(self):
        """Random 2-mode Gaussian: JAX should match TheWalrus exactly."""
        np.random.seed(42)

        for trial in range(3):
            N = 2
            r = np.random.uniform(0.2, 1.0)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, 2 * np.pi)

            # Build random 2-mode Gaussian
            S_sq = np.array(two_mode_squeezer_symplectic(r))
            t = np.cos(theta)
            r_bs = np.sin(theta)
            U = np.array([[t, -np.exp(-1j * phi) * r_bs], [np.exp(1j * phi) * r_bs, t]])
            S_rot = np.array(passive_unitary_to_symplectic(jnp.array(U)))
            S = S_rot @ S_sq

            cov_vac = vacuum_covariance(N, HBAR)
            cov = S @ np.array(cov_vac) @ S.T

            alpha = np.random.randn(2) + 1j * np.random.randn(2)
            alpha *= 0.5
            mu = np.array(complex_alpha_to_qp(jnp.array(alpha), HBAR))

            pnr = (np.random.randint(0, 3),)

            # TheWalrus
            psi_walrus = np.zeros(CUTOFF, dtype=np.complex128)
            for n in range(CUTOFF):
                full_pattern = [n] + list(pnr)
                amp = walrus_pure_state_amplitude(
                    mu, cov, full_pattern, hbar=HBAR, check_purity=False
                )
                psi_walrus[n] = amp
            prob_walrus = np.sum(np.abs(psi_walrus) ** 2)
            psi_walrus_norm = (
                psi_walrus / np.sqrt(prob_walrus) if prob_walrus > 1e-15 else psi_walrus
            )

            # JAX
            jax_cov = jnp.array(cov)
            jax_mu = jnp.array(mu)
            psi_jax, prob_jax = jax_pure_state_amplitude(
                jax_mu, jax_cov, pnr, CUTOFF, HBAR
            )

            diff_psi = np.linalg.norm(np.array(psi_jax) - psi_walrus_norm)

            print(f"\n2-mode random trial {trial}, PNR={pnr}:")
            print(f"  Prob TheWalrus: {prob_walrus:.6e}")
            print(f"  Prob JAX:       {float(prob_jax):.6e}")
            print(f"  |psi - psi_walrus|: {diff_psi:.6e}")

            assert np.isclose(prob_walrus, prob_jax, rtol=1e-3), (
                f"Trial {trial}: Prob mismatch: Walrus={prob_walrus:.6e}, JAX={float(prob_jax):.6e}"
            )
            assert diff_psi < 1e-5, f"Trial {trial}: State vector mismatch: {diff_psi}"


class Test3ModeWithHeralding:
    """Test 3-mode Gaussian with PNR heralding on 2 control modes."""

    def test_3mode_random_gaussian_consistency(self):
        """Random 3-mode Gaussian: JAX should match TheWalrus."""
        np.random.seed(123)

        N = 3
        # Simple 3-mode Gaussian: independent squeezers + beam splitter + displacement
        r_vals = np.random.uniform(0.1, 0.5, size=N)

        # Build covariance from squeezers
        cov_vac = vacuum_covariance(N, HBAR)
        S_diag = np.diag(np.concatenate([np.exp(-r_vals), np.exp(r_vals)]))
        cov = S_diag @ np.array(cov_vac) @ S_diag.T

        alpha = np.random.randn(N) + 1j * np.random.randn(N)
        alpha *= 0.3
        mu = np.array(complex_alpha_to_qp(jnp.array(alpha), HBAR))

        pnr = (0, 1)  # Herald on modes 1,2

        # TheWalrus
        psi_walrus = np.zeros(CUTOFF, dtype=np.complex128)
        for n in range(CUTOFF):
            full_pattern = [n] + list(pnr)
            amp = walrus_pure_state_amplitude(
                mu, cov, full_pattern, hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp
        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)
        psi_walrus_norm = (
            psi_walrus / np.sqrt(prob_walrus) if prob_walrus > 1e-15 else psi_walrus
        )

        # JAX
        jax_cov = jnp.array(cov)
        jax_mu = jnp.array(mu)
        psi_jax, prob_jax = jax_pure_state_amplitude(jax_mu, jax_cov, pnr, CUTOFF, HBAR)

        diff_psi = np.linalg.norm(np.array(psi_jax) - psi_walrus_norm)

        print(f"\n3-mode Gaussian, PNR={pnr}:")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  Prob JAX:       {float(prob_jax):.6e}")
        print(f"  |psi - psi_walrus|: {diff_psi:.6e}")

        assert np.isclose(prob_walrus, prob_jax, rtol=1e-3), (
            f"Prob mismatch: Walrus={prob_walrus:.6e}, JAX={float(prob_jax):.6e}"
        )
        assert diff_psi < 1e-5, f"State vector mismatch: {diff_psi}"


class TestJaxGetHeraldedStateConsistency:
    """Test jax_get_heralded_state matches TheWalrus for n_ctrl=0,1,2."""

    def test_n_ctrl_0_state_matches_thewalrus(self):
        """n_ctrl=0: 1-mode Gaussian state should match TheWalrus."""
        from src.simulation.jax.runner import (
            jax_get_heralded_state,
            jax_clements_unitary,
        )

        # Create params for n_ctrl=0
        r = 0.4
        theta = np.pi / 5
        alpha = 0.6 + 0.2j

        params = {
            "r": jnp.array([r, 0.0, 0.0]),
            "phases": jnp.array([theta] + [0.0] * 8),
            "disp": jnp.array([alpha, 0.0, 0.0], dtype=jnp.complex64),
            "n_ctrl": jnp.array(0),
            "pnr": jnp.array([0, 0]),
        }

        cutoff = 20
        pnr_max = 3

        vec_jax, prob_jax, _, _, _, _ = jax_get_heralded_state(params, cutoff, pnr_max)

        # TheWalrus: build same 1-mode Gaussian using JAX's Clements construction
        N = 1
        U_jax = np.array(jax_clements_unitary(jnp.array([theta]), N))
        S_pass = np.array(passive_unitary_to_symplectic(jnp.array(U_jax)))
        S_sq = np.diag([np.exp(-r), np.exp(r)])
        S_total = S_pass @ S_sq

        cov_vac = np.array(vacuum_covariance(N, HBAR))
        cov = S_total @ cov_vac @ S_total.T
        mu = np.array(complex_alpha_to_qp(jnp.array([alpha]), HBAR))

        psi_walrus = np.zeros(cutoff, dtype=np.complex128)
        for n in range(cutoff):
            amp = walrus_pure_state_amplitude(
                mu, cov, [n], hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp

        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)
        psi_walrus_norm = (
            psi_walrus / np.sqrt(prob_walrus) if prob_walrus > 1e-15 else psi_walrus
        )

        diff_psi = np.linalg.norm(np.array(vec_jax) - psi_walrus_norm)

        print(f"\nn_ctrl=0 state comparison:")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  Prob JAX:       {float(prob_jax):.6e}")
        print(f"  |psi_jax - psi_walrus|: {diff_psi:.6e}")

        assert np.isclose(prob_walrus, prob_jax, rtol=1e-3), (
            f"Prob mismatch: Walrus={prob_walrus:.6e}, JAX={float(prob_jax):.6e}"
        )
        assert diff_psi < 1e-5, f"State vector mismatch: {diff_psi}"

    def test_n_ctrl_1_state_matches_thewalrus(self):
        """n_ctrl=1: 2-mode heralded state should match TheWalrus."""
        from src.simulation.jax.runner import (
            jax_get_heralded_state,
            jax_clements_unitary,
        )

        np.random.seed(42)
        r_vals = np.array([0.5, 0.4, 0.0])
        phases = np.random.uniform(0, np.pi, 9)
        alpha = np.array([0.3 + 0.2j, 0.4 - 0.1j, 0.0], dtype=np.complex64)
        pnr_outcome = 1

        params = {
            "r": jnp.array(r_vals),
            "phases": jnp.array(phases),
            "disp": jnp.array(alpha),
            "n_ctrl": jnp.array(1),
            "pnr": jnp.array([pnr_outcome, 0]),
        }

        cutoff = 15
        pnr_max = 3

        vec_jax, prob_jax, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
            params, cutoff, pnr_max
        )

        # Build same 2-mode Gaussian for TheWalrus
        N = 2
        r_2 = r_vals[:2]
        phases_2 = phases[:4]

        S_sq = np.diag(np.concatenate([np.exp(-r_2), np.exp(r_2)]))
        U = np.array(jax_clements_unitary(jnp.array(phases_2), N))
        S_pass = np.array(passive_unitary_to_symplectic(jnp.array(U)))
        S = S_pass @ S_sq

        cov_vac = vacuum_covariance(N, HBAR)
        cov = S @ np.array(cov_vac) @ S.T
        mu = np.array(complex_alpha_to_qp(jnp.array(alpha[:2]), HBAR))

        psi_walrus = np.zeros(cutoff, dtype=np.complex128)
        for n in range(cutoff):
            amp = walrus_pure_state_amplitude(
                mu, cov, [n, pnr_outcome], hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp

        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)
        psi_walrus_norm = (
            psi_walrus / np.sqrt(prob_walrus) if prob_walrus > 1e-15 else psi_walrus
        )

        diff_psi = np.linalg.norm(np.array(vec_jax) - psi_walrus_norm)

        print(f"\nn_ctrl=1 state comparison (PNR={pnr_outcome}):")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  Prob JAX:       {float(prob_jax):.6e}")
        print(f"  |psi_jax - psi_walrus|: {diff_psi:.6e}")

        assert np.isclose(prob_walrus, prob_jax, rtol=1e-3), (
            f"Prob mismatch: Walrus={prob_walrus:.6e}, JAX={float(prob_jax):.6e}"
        )
        assert diff_psi < 1e-5, f"State vector mismatch: {diff_psi}"

    def test_n_ctrl_2_state_matches_thewalrus(self):
        """n_ctrl=2: 3-mode heralded state should match TheWalrus."""
        from src.simulation.jax.runner import (
            jax_get_heralded_state,
            jax_clements_unitary,
        )

        np.random.seed(123)
        r_vals = np.array([0.3, 0.4, 0.5])
        phases = np.random.uniform(0, np.pi, 9)
        alpha = np.array([0.2 + 0.1j, 0.3 - 0.2j, 0.1 + 0.3j], dtype=np.complex64)
        pnr_0, pnr_1 = 0, 1

        params = {
            "r": jnp.array(r_vals),
            "phases": jnp.array(phases),
            "disp": jnp.array(alpha),
            "n_ctrl": jnp.array(2),
            "pnr": jnp.array([pnr_0, pnr_1]),
        }

        cutoff = 15
        pnr_max = 3

        vec_jax, prob_jax, _, max_pnr, total_pnr, _ = jax_get_heralded_state(
            params, cutoff, pnr_max
        )

        # Build same 3-mode Gaussian for TheWalrus
        N = 3
        S_sq = np.diag(np.concatenate([np.exp(-r_vals), np.exp(r_vals)]))
        U = np.array(jax_clements_unitary(jnp.array(phases), N))
        S_pass = np.array(passive_unitary_to_symplectic(jnp.array(U)))
        S = S_pass @ S_sq

        cov_vac = vacuum_covariance(N, HBAR)
        cov = S @ np.array(cov_vac) @ S.T
        mu = np.array(complex_alpha_to_qp(jnp.array(alpha), HBAR))

        psi_walrus = np.zeros(cutoff, dtype=np.complex128)
        for n in range(cutoff):
            amp = walrus_pure_state_amplitude(
                mu, cov, [n, pnr_0, pnr_1], hbar=HBAR, check_purity=False
            )
            psi_walrus[n] = amp

        prob_walrus = np.sum(np.abs(psi_walrus) ** 2)
        psi_walrus_norm = (
            psi_walrus / np.sqrt(prob_walrus) if prob_walrus > 1e-15 else psi_walrus
        )

        diff_psi = np.linalg.norm(np.array(vec_jax) - psi_walrus_norm)

        print(f"\nn_ctrl=2 state comparison (PNR=[{pnr_0},{pnr_1}]):")
        print(f"  Prob TheWalrus: {prob_walrus:.6e}")
        print(f"  Prob JAX:       {float(prob_jax):.6e}")
        print(f"  |psi_jax - psi_walrus|: {diff_psi:.6e}")

        assert np.isclose(prob_walrus, prob_jax, rtol=1e-3), (
            f"Prob mismatch: Walrus={prob_walrus:.6e}, JAX={float(prob_jax):.6e}"
        )
        assert diff_psi < 1e-5, f"State vector mismatch: {diff_psi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
