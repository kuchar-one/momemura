"""Nano isolation of the path-3 (symplectic moment-space) bug.

Compares Fock-space mixing (paths 1/2) against symplectic moment-space mixing
(path 3) on a 2-mode toy: two single-mode squeezed states, one BS, optional
point homodyne on mode B. Isolates whether the bug is in get_bs_symplectic or
in measure_homodyne.
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from thewalrus.quantum import state_vector
import thewalrus.symplectic as symp

from frontend.independent_verifier import (
    _build_gaussian_moments, _fock_bs_unitary, _hermite_phi,
)
from frontend.gaussian_decomposition import get_bs_symplectic, measure_homodyne

HBAR = 2.0
CUT = 16


def fock_overlap(a, b):
    a = a / (np.linalg.norm(a) + 1e-300)
    b = b / (np.linalg.norm(b) + 1e-300)
    return abs(np.vdot(a, b)) ** 2


def leaf_state_fock(r, cutoff):
    """1-mode squeezed (r, phase=0) Fock vector via thewalrus."""
    mu, cov = _build_gaussian_moments(np.array([r]), np.array([0.0]), np.array([0.0 + 0j]), 1)
    sv = np.asarray(state_vector(mu, cov, cutoff=cutoff, hbar=HBAR, normalize=False, check_purity=False))
    return sv / (np.linalg.norm(sv) + 1e-300), mu, cov


def joint_moments(rA, rB):
    muA, covA = _build_gaussian_moments(np.array([rA]), np.array([0.0]), np.array([0.0 + 0j]), 1)
    muB, covB = _build_gaussian_moments(np.array([rB]), np.array([0.0]), np.array([0.0 + 0j]), 1)
    # assemble 2-mode xp-ordered moments: order (x0,x1,p0,p1)
    V = np.zeros((4, 4)); mu = np.zeros(4)
    # mode 0 = A, mode 1 = B
    for (m, c, off) in [(muA, covA, 0), (muB, covB, 1)]:
        V[off, off] = c[0, 0]; V[off, 2 + off] = c[0, 1]
        V[2 + off, off] = c[1, 0]; V[2 + off, 2 + off] = c[1, 1]
        mu[off] = m[0]; mu[2 + off] = m[1]
    return mu, V


def test_bs_only(theta, phi, rA, rB):
    """Compare 2-mode Fock-BS state vs symplectic-BS state (no homodyne)."""
    psiA, _, _ = leaf_state_fock(rA, CUT)
    psiB, _, _ = leaf_state_fock(rB, CUT)
    psi_in = np.kron(psiA, psiB)
    U = _fock_bs_unitary(theta, phi, CUT)
    psi_fock = U @ psi_in  # shape cut^2, indices [a,b]

    mu, V = joint_moments(rA, rB)
    S = get_bs_symplectic(theta, phi, 2, 0, 1)
    Vp = S @ V @ S.T
    mup = S @ mu
    psi_symp = np.asarray(state_vector(mup, Vp, cutoff=CUT, hbar=HBAR,
                                       normalize=False, check_purity=False)).ravel()
    return fock_overlap(psi_fock, psi_symp)


def test_with_homodyne(theta, phi, rA, rB, xval):
    """Compare Fock BS+homodyne(B) vs symplectic BS+homodyne(B)."""
    psiA, _, _ = leaf_state_fock(rA, CUT)
    psiB, _, _ = leaf_state_fock(rB, CUT)
    psi_in = np.kron(psiA, psiB)
    U = _fock_bs_unitary(theta, phi, CUT)
    psi_out = (U @ psi_in).reshape((CUT, CUT))
    phi_vec = _hermite_phi(xval, CUT)
    psi_fock = psi_out @ phi_vec  # keep A, project B

    mu, V = joint_moments(rA, rB)
    S = get_bs_symplectic(theta, phi, 2, 0, 1)
    Vp = S @ V @ S.T
    mup = S @ mu
    Vn, mun = measure_homodyne(Vp, mup, 1, 2, xval)  # homodyne mode B(=1)
    psi_symp = np.asarray(state_vector(mun, Vn, cutoff=CUT, hbar=HBAR,
                                       normalize=False, check_purity=False)).ravel()
    return fock_overlap(psi_fock, psi_symp)


if __name__ == "__main__":
    print("== BS only (no homodyne) ==")
    for (th, ph) in [(np.pi/4, 0.0), (0.7, 0.0), (np.pi/4, 0.5), (1.0, 1.3)]:
        F = test_bs_only(th, ph, 0.6, 0.3)
        print(f"  theta={th:.4f} phi={ph:.3f}  F(fock,symp)={F:.5f}")
    print("== BS + point homodyne(B) ==")
    for (th, ph, x) in [(np.pi/4, 0.0, 0.0), (np.pi/4, 0.0, 0.3), (0.7, 0.0, 0.5), (np.pi/4, 0.5, 0.3)]:
        F = test_with_homodyne(th, ph, 0.6, 0.3, x)
        print(f"  theta={th:.4f} phi={ph:.3f} x={x}  F(fock,symp)={F:.5f}")
