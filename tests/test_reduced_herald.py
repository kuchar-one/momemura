"""Regression test: reduced_herald == heralded_output (state AND probability).

reduced_herald conditions the n=0 control modes analytically in moment space
and builds the remaining small system with the stable Hermite recurrence,
avoiding the per-amplitude loop hafnians that make heralded_output intractable
and ill-conditioned at high squeezing.  On small, well-conditioned generators
the two must agree exactly.
"""
import numpy as np
import pytest

from frontend.gbs_optimizer import heralded_output, reduced_herald


def _random_generator(rng, N=3, r_max=0.9):
    import thewalrus.symplectic as symp
    rs = rng.uniform(0.2, r_max, N)
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    U = np.linalg.qr(A)[0]
    Ssq = np.block([[np.diag(np.exp(-rs)), np.zeros((N, N))],
                    [np.zeros((N, N)), np.diag(np.exp(rs))]])
    S = symp.interferometer(U) @ Ssq
    cov = S @ S.T
    mu = rng.normal(0, 0.7, 2 * N)
    return cov, mu


@pytest.mark.parametrize("seed", [7, 11, 23])
@pytest.mark.parametrize("n", [[0, 0], [1, 0], [2, 1], [0, 2]])
def test_reduced_herald_matches_heralded_output(seed, n):
    rng = np.random.default_rng(seed)
    cov, mu = _random_generator(rng)
    L = 12
    a, pa = heralded_output(cov, mu, 0, [1, 2], n, cutoff=L)
    b, pb = reduced_herald(cov, mu, 0, [1, 2], n, cutoff=L)
    assert abs(abs(np.vdot(a, b)) ** 2 - 1.0) < 1e-9
    assert pb == pytest.approx(pa, rel=1e-6)


def test_reduced_herald_all_vacuum_pure_gaussian():
    """All-zero PNR pattern -> output must be an exactly Gaussian pure state."""
    rng = np.random.default_rng(3)
    cov, mu = _random_generator(rng)
    L = 20
    psi, prob = reduced_herald(cov, mu, 0, [1, 2], [0, 0], cutoff=L)
    a = np.diag(np.sqrt(np.arange(1, L)), k=1)
    am = psi.conj() @ (a @ psi)
    a2 = psi.conj() @ (a @ a @ psi)
    nb = float((psi.conj() @ (a.conj().T @ a @ psi)).real)
    da2 = a2 - am ** 2
    dn = nb - abs(am) ** 2
    det = (1 + 2 * dn.real) ** 2 - 4 * abs(da2) ** 2
    assert det == pytest.approx(1.0, abs=1e-6)   # purity+Gaussianity criterion
    assert 0 < prob <= 1
