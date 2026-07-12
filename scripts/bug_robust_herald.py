"""Robust heralded reconstruction from a correct (high-n_bar) equivalent
Gaussian: fully whiten the signal mode (single-mode symplectic -> vacuum
marginal + zero mean) so the state_vector recurrence stays well-conditioned,
herald controls via post_select at an internal cutoff with headroom, then align
(absorbs the removed single-mode Gaussian). Compare to path 2.
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from thewalrus.quantum import state_vector
from scripts.bug_prod_regime import build_params
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.independent_verifier import verify_circuit
from frontend.gbs_optimizer import heralded_output, align_states

HBAR = 2.0


def whiten_signal_symplectic(cov, s, N):
    Vs = np.array([[cov[s, s], cov[s, s + N]], [cov[s + N, s], cov[s + N, s + N]]])
    w, U = np.linalg.eigh(Vs)
    M = np.sqrt(HBAR / 2.0) * (np.diag(1.0 / np.sqrt(w)) @ U.T)  # det=1 (pure)
    S = np.eye(2 * N)
    S[s, s] = M[0, 0]; S[s, s + N] = M[0, 1]
    S[s + N, s] = M[1, 0]; S[s + N, s + N] = M[1, 1]
    return S


def robust_herald(cov, mu, s, cidx, pnr, cutoff, internal):
    N = cov.shape[0] // 2
    S = whiten_signal_symplectic(cov, s, N)
    cw = S @ cov @ S.T
    mw = S @ mu
    mw[s] = 0.0; mw[s + N] = 0.0   # remove signal mean (align absorbs it)
    psi, p = heralded_output(cw, mw, s, cidx, pnr, cutoff=internal)
    return np.asarray(psi).ravel()[:cutoff], p


if __name__ == "__main__":
    CMP = 24
    cases = {
        "2leaf strong+disp": ([1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],{0:[3],1:[2]},7,1.8,2.2,0.0),
        "2leaf +hx":          ([1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0],{0:[3],1:[2]},7,1.8,2.2,1.4),
        "4leaf":              ([1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],{0:[3],1:[2],2:[4],3:[1]},3,1.8,2.2,1.4),
    }
    for tag,(act,nc,pnr,seed,rs,ds,hx) in cases.items():
        p = build_params(act,nc,pnr,seed,rs,ds,hx,{})
        psi2 = np.asarray(verify_circuit(p,cutoff=CMP,pnr_max=15)["state"]).ravel()
        eq = compute_equivalent_gaussian(p)
        old,_ = heralded_output(eq["cov"],eq["mu"],eq["signal_idx"],eq["control_idx"],eq["pnr_outcomes"],cutoff=CMP)
        fo,_ = align_states(psi2, np.asarray(old).ravel(), CMP, align_cut=CMP)
        for internal in (CMP, 40, 60):
            new,_ = robust_herald(eq["cov"],eq["mu"],eq["signal_idx"],eq["control_idx"],eq["pnr_outcomes"],CMP,internal)
            fn,_ = align_states(psi2, new, CMP, align_cut=CMP)
            print(f"{tag}: F_old={fo:.4f}  F_robust(int={internal})={fn:.4f}")
