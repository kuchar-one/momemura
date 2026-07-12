"""_apply_gaussian_fast (eigendecomposition path) must reproduce the original
scipy.linalg.expm implementation of G(dr, di, r, phi, varphi)|psi> exactly."""
import os, sys
import numpy as np

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "scripts")]


def _apply_gaussian_expm(params, psi):
    import scipy.linalg as sla
    psi = np.asarray(psi, complex).ravel()
    n = len(psi)
    a = np.diag(np.sqrt(np.arange(1, n)), k=1); ad = a.T
    dr, di, r, phi, varphi = params
    v = sla.expm(0.5 * (r * np.exp(-2j * phi) * a @ a
                        - r * np.exp(2j * phi) * ad @ ad)) @ psi
    v = np.exp(1j * np.arange(n) * varphi) * v
    disp = dr + 1j * di
    v = sla.expm(disp * ad - np.conj(disp) * a) @ v
    return v / (np.linalg.norm(v) + 1e-300)


def test_fast_gaussian_matches_expm():
    from run_hanamura_all import _apply_gaussian_fast
    rng = np.random.default_rng(11)
    for trial in range(8):
        n = int(rng.integers(20, 49))
        psi = rng.normal(size=n) + 1j * rng.normal(size=n)
        psi = psi / np.linalg.norm(psi)
        params = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                  rng.uniform(-0.8, 0.8), rng.uniform(-np.pi, np.pi),
                  rng.uniform(-np.pi, np.pi))
        v_ref = _apply_gaussian_expm(params, psi)
        v_new = _apply_gaussian_fast(params, psi)
        # states equal up to global phase
        ov = abs(np.vdot(v_ref, v_new))
        assert ov > 1 - 1e-10, f"trial {trial}: overlap {ov}"


if __name__ == "__main__":
    test_fast_gaussian_matches_expm()
    print("fast Gaussian application == expm reference: OK")
