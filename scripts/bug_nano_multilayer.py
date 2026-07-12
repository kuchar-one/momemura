"""Multi-layer nano: 4 single-mode squeezed leaves, depth-2 tree, no controls.
Compares Fock-space full-tree mixing (path 2 style) vs symplectic moment-space
full-tree mixing (path 3 style: get_bs_symplectic + measure_homodyne, drop B).
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from thewalrus.quantum import state_vector
from frontend.independent_verifier import (
    _build_gaussian_moments, _fock_bs_unitary, _hermite_phi,
)
from frontend.gaussian_decomposition import get_bs_symplectic, measure_homodyne

HBAR = 2.0
CUT = 18


def leaf_moments(r):
    return _build_gaussian_moments(np.array([r]), np.array([0.0]), np.array([0.0 + 0j]), 1)


def leaf_fock(r):
    mu, cov = leaf_moments(r)
    sv = np.asarray(state_vector(mu, cov, cutoff=CUT, hbar=HBAR, normalize=False, check_purity=False))
    return sv / (np.linalg.norm(sv) + 1e-300)


def fock_mix(sA, sB, theta, phi, x):
    U = _fock_bs_unitary(theta, phi, CUT)
    out = (U @ np.kron(sA, sB)).reshape((CUT, CUT))
    v = out @ _hermite_phi(x, CUT)   # keep A, project B
    return v / (np.linalg.norm(v) + 1e-300)


def fock_tree(rs, thetas, phis, xs):
    s = [leaf_fock(r) for r in rs]
    a = fock_mix(s[0], s[1], thetas[0], phis[0], xs[0])
    b = fock_mix(s[2], s[3], thetas[1], phis[1], xs[1])
    return fock_mix(a, b, thetas[2], phis[2], xs[2])


def symp_tree(rs, thetas, phis, xs):
    # assemble 4-mode xp moments, modes 0..3
    N = 4
    V = np.zeros((2 * N, 2 * N)); mu = np.zeros(2 * N)
    for i, r in enumerate(rs):
        m, c = leaf_moments(r)
        V[i, i] = c[0, 0]; V[i, N + i] = c[0, 1]
        V[N + i, i] = c[1, 0]; V[N + i, N + i] = c[1, 1]
        mu[i] = m[0]; mu[N + i] = m[1]
    modes = [0, 1, 2, 3]  # track leaf id at each position

    def mix(V, mu, modes, leafA, leafB, theta, phi, x):
        Ncur = len(modes)
        iA = modes.index(leafA); iB = modes.index(leafB)
        S = get_bs_symplectic(theta, phi, Ncur, iA, iB)
        V = S @ V @ S.T; mu = S @ mu
        V, mu = measure_homodyne(V, mu, iB, Ncur, x)
        modes.pop(iB)
        return V, mu, modes

    V, mu, modes = mix(V, mu, modes, 0, 1, thetas[0], phis[0], xs[0])
    V, mu, modes = mix(V, mu, modes, 2, 3, thetas[1], phis[1], xs[1])
    V, mu, modes = mix(V, mu, modes, 0, 2, thetas[2], phis[2], xs[2])  # keep leaf0 line
    sv = np.asarray(state_vector(mu, V, cutoff=CUT, hbar=HBAR, normalize=False, check_purity=False)).ravel()
    return sv / (np.linalg.norm(sv) + 1e-300)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(4):
        rs = rng.uniform(0.2, 0.5, 4)
        thetas = [np.pi / 4] * 3
        phis = [0.0, 0.0, 0.0]
        xs = rng.uniform(-0.4, 0.4, 3)
        f = fock_tree(rs, thetas, phis, xs)
        s = symp_tree(rs, thetas, phis, xs)
        F = abs(np.vdot(f, s)) ** 2
        print(f"trial {trial}: F(fock_tree, symp_tree) = {F:.5f}")
