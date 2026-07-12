"""Bisect the second bug by building params by hand and comparing
verify_circuit (path 2) vs compute_equivalent_gaussian+heralded_output (path 3)
for chosen active-leaf / control layouts."""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import qutip  # noqa
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
from frontend.independent_verifier import verify_circuit
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.gbs_optimizer import heralded_output, align_states

CUT = 16
NMODES = 3   # 1 signal + up to 2 controls


def build_params(active, nctrl, seed=0, hx=0.2):
    rng = np.random.default_rng(seed)
    L = 8
    n_ctrl = np.zeros(L, dtype=np.int32)
    r = np.zeros((L, NMODES)); phases = np.zeros((L, NMODES * NMODES))
    disp = np.zeros((L, NMODES), dtype=complex); pnr = np.zeros((L, NMODES - 1), dtype=np.int32)
    for i in range(L):
        if not active[i]:
            continue
        nc = nctrl[i]; n_ctrl[i] = nc
        N = nc + 1
        r[i, :N] = rng.uniform(0.2, 0.5, N)
        phases[i, :N * N] = rng.uniform(0, 2 * np.pi, N * N)
        disp[i, :N] = rng.uniform(-0.3, 0.3, N) + 1j * rng.uniform(-0.3, 0.3, N)
        for c in range(nc):
            pnr[i, c] = rng.integers(0, 3)
    leaf_params = {"n_ctrl": n_ctrl, "pnr": pnr, "r": r, "phases": phases,
                   "disp": disp, "pnr_max": np.full(L, 15, dtype=np.int32)}
    mix = np.zeros((7, 3)); mix[:, 0] = np.pi / 4
    return {
        "homodyne_x": np.full(7, hx),
        "homodyne_window": 0.0,
        "mix_params": mix,
        "leaf_active": np.array(active, dtype=bool),
        "leaf_params": leaf_params,
        "final_gauss": {},
    }


def compare(active, nctrl, seed=0):
    params = build_params(active, nctrl, seed)
    res2 = verify_circuit(params, cutoff=CUT, pnr_max=15)
    psi2 = np.asarray(res2["state"]).ravel(); psi2 /= (np.linalg.norm(psi2) + 1e-300)
    eq = compute_equivalent_gaussian(params)
    psi3, _ = heralded_output(eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"],
                              eq["pnr_outcomes"], cutoff=CUT)
    psi3 = np.asarray(psi3).ravel(); psi3 /= (np.linalg.norm(psi3) + 1e-300)
    F, _ = align_states(psi2, psi3, len(psi2), align_cut=len(psi2))
    return F


if __name__ == "__main__":
    cases = {
        "seed4-full   ": ([1, 0, 1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 2]),
        "two-noctrl mix": ([1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]),
        "mix w/ctrl(B) ": ([1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
        "mix w/ctrl(A) ": ([1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
        "both ctrl     ": ([1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]),
        "L2 cross 0,2  ": ([1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]),
        "L2 cross ctrl ": ([1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]),
        "layer1 23+ctrl": ([0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]),
    }
    for name, (a, nc) in cases.items():
        F = compare(a, nc)
        print(f"{name}: F = {F:.4f}")
