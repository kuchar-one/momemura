"""Verify the fix hypothesis: whiten the signal mode (undo its single-mode
squeeze+displacement) BEFORE the Fock herald, so state_vector's recurrence is
well-conditioned, then compare to path 2 via align (which absorbs the removed
single-mode Gaussian). If F jumps to ~1, the diagnosis + fix are confirmed.
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from scripts.bug_prod_regime import build_params
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.independent_verifier import verify_circuit
from frontend.gbs_optimizer import heralded_output, align_states

HBAR = 2.0


def signal_whitening_symplectic(cov, mu, s, N):
    """Single-mode symplectic M (2x2) that maps the signal marginal cov to
    vacuum (hbar/2 I). Embedded into 2N. Also returns the displacement removed."""
    Vs = np.array([[cov[s, s], cov[s, s + N]],
                   [cov[s + N, s], cov[s + N, s + N]]])
    # Vs = (hbar/2) S S^T (pure). Eigendecompose: Vs = U diag(l1,l2) U^T
    w, U = np.linalg.eigh(Vs)
    # whitening: M = sqrt(hbar/2) diag(1/sqrt(l)) U^T  (symplectic up to rotation)
    M = np.sqrt(HBAR / 2.0) * (np.diag(1.0 / np.sqrt(w)) @ U.T)
    S = np.eye(2 * N)
    S[s, s] = M[0, 0]; S[s, s + N] = M[0, 1]
    S[s + N, s] = M[1, 0]; S[s + N, s + N] = M[1, 1]
    return S


def heralded_output_whitened(cov, mu, s, control_idx, pnr, cutoff):
    N = cov.shape[0] // 2
    S = signal_whitening_symplectic(cov, mu, s, N)
    covw = S @ cov @ S.T
    muw = S @ mu
    # also remove signal mean so the recurrence is centered
    muw2 = muw.copy(); muw2[s] = 0.0; muw2[s + N] = 0.0
    return heralded_output(covw, muw2, s, control_idx, pnr, cutoff=cutoff)


if __name__ == "__main__":
    CMP = 24
    cases = {
        "2leaf strong+disp": ([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], {0: [3], 1: [2]}, 7, 1.8, 2.2, 0.0),
        "2leaf +hx":          ([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], {0: [3], 1: [2]}, 7, 1.8, 2.2, 1.4),
        "4leaf":              ([1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], {0: [3], 1: [2], 2: [4], 3: [1]}, 3, 1.8, 2.2, 1.4),
    }
    for tag, (act, nc, pnr, seed, rs, ds, hx) in cases.items():
        p = build_params(act, nc, pnr, seed, rs, ds, hx, {})
        psi2 = np.asarray(verify_circuit(p, cutoff=CMP, pnr_max=15)["state"]).ravel()
        eq = compute_equivalent_gaussian(p)
        psi3, _ = heralded_output(eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"], eq["pnr_outcomes"], cutoff=CMP)
        psi3w, _ = heralded_output_whitened(eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"], eq["pnr_outcomes"], cutoff=CMP)
        f_old, _ = align_states(psi2, np.asarray(psi3).ravel(), CMP, align_cut=CMP)
        f_new, _ = align_states(psi2, np.asarray(psi3w).ravel(), CMP, align_cut=CMP)
        print(f"{tag}: F_old={f_old:.4f}  F_whitened={f_new:.4f}")
