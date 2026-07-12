"""Truncation-free check of the BS convention used in path 3.

Compares the 2x2 mode (Heisenberg) transformation implied by the Fock-space
generator (paths 1/2) against the symplectic get_bs_symplectic (path 3), via:
  (a) the mode-operator matrix extracted from the Fock unitary, and
  (b) propagation of a coherent-state mean through both (exact, no truncation).
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from scipy import linalg as sla
from frontend.independent_verifier import _fock_bs_unitary, _build_gaussian_moments
from frontend.gaussian_decomposition import get_bs_symplectic
from thewalrus.quantum import state_vector

HBAR = 2.0


def fock_mode_matrix(theta, phi, cutoff=12):
    """Extract 2x2 M with a_out_k = sum_j M[k,j] a_j from the Fock BS unitary.
    a_out = U^dag a U. We read M from action on single-photon states:
    U|1,0> etc. Actually we compute M[k,j] = <vac| a_k U a_j^dag |vac>."""
    a1 = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    eye = np.eye(cutoff)
    a = np.kron(a1, eye); b = np.kron(eye, a1)
    ops = [a, b]
    U = _fock_bs_unitary(theta, phi, cutoff)
    vac = np.zeros(cutoff * cutoff); vac[0] = 1.0
    M = np.zeros((2, 2), dtype=complex)
    for k in range(2):
        for j in range(2):
            # <vac| a_k U a_j^dag |vac>
            ket = U @ (ops[j].T @ vac)
            bra = ops[k].T @ vac
            M[k, j] = np.vdot(bra, ket)
    return M


def symp_mode_matrix(theta, phi):
    """The 2x2 complex U that get_bs_symplectic encodes (a_out = U a)."""
    t = np.cos(theta); r = np.sin(theta)
    U = np.array([[t, -np.exp(-1j * phi) * r],
                  [np.exp(1j * phi) * r, t]], dtype=complex)
    return U


def coherent_mean_check(theta, phi):
    """Propagate a coherent state's mean through both conventions (exact)."""
    # mode A coherent alpha=1.0, mode B coherent alpha=0.5j
    alphaA, alphaB = 1.0 + 0j, 0.5j
    # xp mean, hbar=2: x=sqrt(2hbar)Re, p=sqrt(2hbar)Im
    s = np.sqrt(2 * HBAR)
    mu = np.array([s * alphaA.real, s * alphaB.real, s * alphaA.imag, s * alphaB.imag])
    S = get_bs_symplectic(theta, phi, 2, 0, 1)
    mu_symp = S @ mu
    alpha_symp = (mu_symp[:2] + 1j * mu_symp[2:]) / s

    # Fock generator mode transform: a_out = M a  -> coherent alpha_out = M alpha
    M = fock_mode_matrix(theta, phi)
    alpha_fock = M @ np.array([alphaA, alphaB])
    return alpha_symp, alpha_fock, M, symp_mode_matrix(theta, phi)


if __name__ == "__main__":
    for (th, ph) in [(np.pi/4, 0.0), (0.7, 0.0), (np.pi/4, 0.5), (1.0, 1.3)]:
        a_s, a_f, M_fock, M_symp = coherent_mean_check(th, ph)
        print(f"theta={th:.4f} phi={ph:.3f}")
        print(f"   alpha_out symp = {np.round(a_s,4)}")
        print(f"   alpha_out fock = {np.round(a_f,4)}")
        print(f"   M_fock=\n{np.round(M_fock,4)}")
        print(f"   M_symp=\n{np.round(M_symp,4)}")
        print(f"   |M_fock - M_symp| = {np.max(np.abs(M_fock-M_symp)):.4f}",
              f" |M_fock - M_symp^*| = {np.max(np.abs(M_fock-M_symp.conj())):.4f}",
              f" |M_fock - M_symp^T| = {np.max(np.abs(M_fock-M_symp.T)):.4f}")
