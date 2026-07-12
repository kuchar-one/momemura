"""Isolate the final-Gaussian bug. Compare:
  Fock:  herald(control) then _apply_final_gaussian on signal Fock vector
  Symp:  apply_final_gaussian_symplectic on moments then herald(control)
on a 2-mode (1 signal + 1 control) Gaussian. Also test 1-mode (no control).
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from thewalrus.quantum import state_vector
from frontend.independent_verifier import _build_gaussian_moments, _apply_final_gaussian
from frontend.gaussian_decomposition import apply_final_gaussian_symplectic
from frontend.gbs_optimizer import align_states

HBAR = 2.0
CUT = 20
FG = {"r": 0.441, "phi": -1.476, "varphi": -0.417, "disp": 0.143 - 0.383j}


def herald_signal(mu, cov, N, post=None):
    sv = np.asarray(state_vector(mu, cov, post_select=post, cutoff=CUT, hbar=HBAR,
                                 normalize=False, check_purity=False)).ravel()
    return sv / (np.linalg.norm(sv) + 1e-300)


def test_1mode():
    mu, cov = _build_gaussian_moments(np.array([0.5]), np.array([0.3]), np.array([0.2 + 0.1j]), 1)
    # Fock: herald (none) then final gauss
    psi = herald_signal(mu, cov, 1)
    psi_fock = _apply_final_gaussian(psi, FG, CUT); psi_fock /= np.linalg.norm(psi_fock) + 1e-300
    # Symp: final gauss on moments then herald
    Vn, mun = apply_final_gaussian_symplectic(cov, mu, FG, 0, 1)
    psi_symp = herald_signal(mun, Vn, 1)
    direct = abs(np.vdot(psi_fock, psi_symp)) ** 2
    F, _ = align_states(psi_fock, psi_symp, CUT, align_cut=CUT)
    print(f"1-mode: direct={direct:.4f}  align={F:.4f}")


def test_2mode(post_val=1):
    # mode0 = signal, mode1 = control. Build a correlated 2-mode Gaussian.
    r = np.array([0.5, 0.4]); phases = np.random.default_rng(1).uniform(0, 2 * np.pi, 4)
    disp = np.array([0.2 + 0.1j, -0.1 + 0.05j])
    mu, cov = _build_gaussian_moments(r, phases, disp, 2)
    # Fock: herald control(mode1) -> signal Fock, then final gauss
    psi = herald_signal(mu, cov, 2, post={1: post_val})
    psi_fock = _apply_final_gaussian(psi, FG, CUT); psi_fock /= np.linalg.norm(psi_fock) + 1e-300
    # Symp: final gauss on signal(mode0) then herald control(mode1)
    Vn, mun = apply_final_gaussian_symplectic(cov, mu, FG, 0, 2)
    psi_symp = herald_signal(mun, Vn, 2, post={1: post_val})
    direct = abs(np.vdot(psi_fock, psi_symp)) ** 2
    F, _ = align_states(psi_fock, psi_symp, CUT, align_cut=CUT)
    print(f"2-mode (herald n={post_val}): direct={direct:.4f}  align={F:.4f}")


if __name__ == "__main__":
    test_1mode()
    for nv in (0, 1, 2):
        test_2mode(nv)
