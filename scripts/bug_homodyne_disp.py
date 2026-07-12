"""Direct test of measure_homodyne WITH displacement (nonzero means), which the
earlier (disp=0) nanos never exercised. Build a displaced 2-mode squeezed
Gaussian, homodyne mode 1 at x_val, compare Fock projection vs symplectic
measure_homodyne. Sweep x_val and displacement.
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from thewalrus.quantum import state_vector
from frontend.independent_verifier import _build_gaussian_moments, _hermite_phi
from frontend.gaussian_decomposition import measure_homodyne

HBAR = 2.0
CUT = 22


def ov(a, b):
    a = a / (np.linalg.norm(a) + 1e-300); b = b / (np.linalg.norm(b) + 1e-300)
    return abs(np.vdot(a, b)) ** 2


def test(r0, r1, dispA, dispB, xval, seed=0):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, 4)
    mu, cov = _build_gaussian_moments(np.array([r0, r1]), phases,
                                      np.array([dispA, dispB]), 2)
    # Fock: full 2-mode state, project mode1 onto |x_val>, keep mode0
    sv = np.asarray(state_vector(mu, cov, cutoff=CUT, hbar=HBAR, normalize=False,
                                 check_purity=False)).reshape((CUT, CUT))
    psi_fock = sv @ _hermite_phi(xval, CUT)
    # Symp: homodyne mode 1, then 1-mode state
    Vn, mun = measure_homodyne(cov, mu, 1, 2, xval)
    psi_symp = np.asarray(state_vector(mun, Vn, cutoff=CUT, hbar=HBAR,
                                       normalize=False, check_purity=False)).ravel()
    return ov(psi_fock, psi_symp)


if __name__ == "__main__":
    print("disp=0 (baseline):")
    for x in (0.0, 0.5, 1.5):
        print(f"   x={x}: F={test(0.6, 0.5, 0j, 0j, x):.5f}")
    print("with displacement:")
    for x in (0.0, 0.5, 1.5):
        print(f"   x={x}: F={test(0.6, 0.5, 1.0 + 0.5j, -0.8 + 0.3j, x):.5f}")
    print("strong squeeze + displacement:")
    for x in (0.0, 1.5):
        print(f"   x={x}: F={test(1.8, 1.5, 1.5 - 0.7j, 1.2 + 0.9j, x):.5f}")
