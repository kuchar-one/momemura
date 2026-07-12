"""Prototype + verify the robust heralding fix: herald in a displaced frame.

|psi_her> ∝ <n_c| Psi>, Psi = D_total|Psi_0> (centered).  Since D on the signal
commutes with the control projector, and <n_c|D_c(α) = Σ_m D_c(α)_{n_c,m}<m_c|:

  psi[s] = D_sig(α_s) · Σ_m ( Π_c D_c(α_c)_{n_c,m_c} ) · H0[s, m...]

where H0 is the heralded amplitude tensor of the CENTERED Gaussian (zero means →
stable thewalrus recurrence). This removes the large-displacement instability
that makes the herald-last path collapse a displaced state toward even parity.
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
from scripts.bug_prod_regime import build_params
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.independent_verifier import verify_circuit
from frontend.gbs_optimizer import heralded_output, align_states

HBAR = 2.0


def disp_op(alpha, cutoff):
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1); ad = a.T
    return sla.expm(alpha * ad - np.conj(alpha) * a)


def heralded_output_disp_frame(cov, mu, s, control_idx, pnr, cutoff, internal=None):
    N = cov.shape[0] // 2
    ic = internal or cutoff
    scale = np.sqrt(2 * HBAR)
    alpha = (mu[:N] + 1j * mu[N:]) / scale          # per-mode complex displacement
    mu0 = np.zeros_like(mu)                          # centered
    H0 = np.asarray(state_vector(mu0, cov, cutoff=ic, hbar=HBAR,
                                 normalize=False, check_purity=False))
    cutoff = ic
    # H0 axes are in mode order 0..N-1. Contract each control axis with the
    # displacement-operator row D(α_c)[n_c, :].
    order = list(range(N))
    # process controls; track current axis positions
    psi = H0
    axes = order.copy()  # axes[k] = current position of original mode k
    for c, n_c in zip(control_idx, pnr):
        Dc = disp_op(alpha[c], cutoff)               # <row|: D[n_c, m]
        row = Dc[n_c, :]
        ax = axes.index(c)
        psi = np.tensordot(psi, row, axes=([ax], [0]))
        axes.pop(ax)
    # psi now has a single remaining axis = signal mode s
    psi = np.asarray(psi).ravel()
    # apply signal displacement D(α_s)
    psi = disp_op(alpha[s], cutoff) @ psi
    nrm = np.linalg.norm(psi)
    return (psi / nrm if nrm > 0 else psi), float(nrm) ** 2


if __name__ == "__main__":
    CMP = 24
    cases = {
        "2leaf strong+disp": ([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], {0: [3], 1: [2]}, 7, 1.8, 2.2, 0.0),
        "2leaf +hx":          ([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], {0: [3], 1: [2]}, 7, 1.8, 2.2, 1.4),
    }
    for tag, (act, nc, pnr, seed, rs, ds, hx) in cases.items():
        p = build_params(act, nc, pnr, seed, rs, ds, hx, {})
        psi2 = np.asarray(verify_circuit(p, cutoff=CMP, pnr_max=15)["state"]).ravel()
        eq = compute_equivalent_gaussian(p)
        old, _ = heralded_output(eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"], eq["pnr_outcomes"], cutoff=CMP)
        new, _ = heralded_output_disp_frame(eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"], eq["pnr_outcomes"], CMP, internal=60)
        new = np.asarray(new).ravel()[:CMP]
        f_old, _ = align_states(psi2, np.asarray(old).ravel(), CMP, align_cut=CMP)
        f_new, _ = align_states(psi2, new, CMP, align_cut=CMP)
        print(f"{tag}: F_old={f_old:.4f}  F_dispframe(int=60)={f_new:.4f}")
