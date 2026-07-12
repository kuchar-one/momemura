"""Minimal check: does thewalrus state_vector(post_select) lose accuracy for
large displacement? Compare a 2-mode herald computed (a) directly via
state_vector(post_select) vs (b) in a displaced frame (center -> herald ->
re-apply displacement analytically). Sweep displacement magnitude.
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from scipy import linalg as sla
from thewalrus.quantum import state_vector
from frontend.independent_verifier import _build_gaussian_moments

HBAR = 2.0
CUT = 30


def disp_op(alpha, cutoff):
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1); ad = a.T
    return sla.expm(alpha * ad - np.conj(alpha) * a)


def direct(mu, cov, n, cutoff):
    sv = np.asarray(state_vector(mu, cov, post_select={1: n}, cutoff=cutoff,
                                 hbar=HBAR, normalize=False, check_purity=False)).ravel()
    return sv


def disp_frame(mu, cov, n, cutoff):
    N = 2; scale = np.sqrt(2 * HBAR)
    alpha = (mu[:N] + 1j * mu[N:]) / scale
    H0 = np.asarray(state_vector(np.zeros(2 * N), cov, cutoff=cutoff, hbar=HBAR,
                                 normalize=False, check_purity=False))  # [s, c]
    Dc = disp_op(alpha[1], cutoff)        # control mode 1
    v = H0 @ Dc[n, :]                      # contract control axis with row n
    v = disp_op(alpha[0], cutoff) @ v      # signal displacement
    return v


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    ph = rng.uniform(0, 2 * np.pi, 4)
    for d in (0.5, 1.0, 1.5, 2.2, 3.0):
        mu, cov = _build_gaussian_moments(np.array([0.6, 0.5]), ph,
                                          np.array([d + 0.3j, -0.7 * d + 0.2j]), 2)
        a = direct(mu, cov, 1, CUT); a /= np.linalg.norm(a) + 1e-300
        b = disp_frame(mu, cov, 1, CUT); b /= np.linalg.norm(b) + 1e-300
        F = abs(np.vdot(a, b)) ** 2
        print(f"disp~{d}: F(direct, disp_frame)={F:.5f}  norm_direct={np.linalg.norm(direct(mu,cov,1,CUT)):.2e}")
