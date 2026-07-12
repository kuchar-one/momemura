"""Production-regime reproduction with hand-built params: strong squeezing,
large displacement, large homodyne x, final Gaussian. Compares path2 vs path3
and reports parity (odd-Fock mass) to chase the even-parity fingerprint.
"""
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

CUT = int(os.environ.get("PROD_CUT", "24"))
NMODES = 3


def odd_mass(psi):
    p = np.abs(psi) ** 2
    return float(p[1::2].sum() / (p.sum() + 1e-300))


def build_params(active, nctrl, pnr_vals, seed, rscale, dscale, hx, fg):
    rng = np.random.default_rng(seed)
    L = 8
    n_ctrl = np.zeros(L, np.int32)
    r = np.zeros((L, NMODES)); phases = np.zeros((L, NMODES * NMODES))
    disp = np.zeros((L, NMODES), complex); pnr = np.zeros((L, NMODES - 1), np.int32)
    for i in range(L):
        if not active[i]:
            continue
        nc = nctrl[i]; n_ctrl[i] = nc; N = nc + 1
        r[i, :N] = rng.uniform(0.5, rscale, N)
        phases[i, :N * N] = rng.uniform(0, 2 * np.pi, N * N)
        disp[i, :N] = rng.uniform(-dscale, dscale, N) + 1j * rng.uniform(-dscale, dscale, N)
        for c in range(nc):
            pnr[i, c] = pnr_vals[i][c]
    leaf_params = {"n_ctrl": n_ctrl, "pnr": pnr, "r": r, "phases": phases,
                   "disp": disp, "pnr_max": np.full(L, 15, np.int32)}
    mix = np.zeros((7, 3)); mix[:, 0] = np.pi / 4
    return {"homodyne_x": np.full(7, hx), "homodyne_window": 0.0, "mix_params": mix,
            "leaf_active": np.array(active, bool), "leaf_params": leaf_params,
            "final_gauss": fg}


def run(params, tag):
    r2 = verify_circuit(params, cutoff=CUT, pnr_max=15)
    psi2 = np.asarray(r2["state"]).ravel(); psi2 /= np.linalg.norm(psi2) + 1e-300
    eq = compute_equivalent_gaussian(params)
    psi3, p3 = heralded_output(eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"],
                               eq["pnr_outcomes"], cutoff=CUT)
    psi3 = np.asarray(psi3).ravel(); psi3 /= np.linalg.norm(psi3) + 1e-300
    F, _ = align_states(psi2, psi3, len(psi2), align_cut=len(psi2))
    print(f"{tag}: F23={F:.4f}  odd2={odd_mass(psi2):.3f} odd3={odd_mass(psi3):.3f}  "
          f"prob2={r2['probability']:.2e} prob3={p3:.2e}")


if __name__ == "__main__":
    fg = {"r": 0.9, "phi": 0.6, "varphi": -0.4, "disp": 1.2 - 0.8j}
    nofg = {}

    # --- isolate: controls vs no controls, strong vs weak squeeze, hx=0, no fg ---
    A2 = [1, 1, 0, 0, 0, 0, 0, 0]
    print("# isolation (hx=0, no final gauss)")
    run(build_params(A2, [0, 0, 0, 0, 0, 0, 0, 0], {}, 7, 1.8, 2.2, 0.0, nofg), "  noctrl strong sq ")
    run(build_params(A2, [0, 0, 0, 0, 0, 0, 0, 0], {}, 7, 0.5, 2.2, 0.0, nofg), "  noctrl weak   sq ")
    run(build_params(A2, [1, 1, 0, 0, 0, 0, 0, 0], {0: [3], 1: [2]}, 7, 1.8, 2.2, 0.0, nofg), "  ctrl   strong sq ")
    run(build_params(A2, [1, 1, 0, 0, 0, 0, 0, 0], {0: [3], 1: [2]}, 7, 0.5, 0.3, 0.0, nofg), "  ctrl   weak   sq ")
    # control but NO displacement (pure squeezed leaves)
    run(build_params(A2, [1, 1, 0, 0, 0, 0, 0, 0], {0: [3], 1: [2]}, 7, 1.8, 0.0, 0.0, nofg), "  ctrl strongsq nodisp")
    print("# original sweeps")
    # two active leaves, each 1 control heralded at moderate matched n
    active = [1, 1, 0, 0, 0, 0, 0, 0]
    nctrl = [1, 1, 0, 0, 0, 0, 0, 0]
    pnr = {0: [3], 1: [2]}
    for hx in (0.0, 0.5, 1.4):
        run(build_params(active, nctrl, pnr, 7, 1.8, 2.2, hx, nofg), f"2leaf nofg hx={hx}")
    for hx in (0.0, 1.4):
        run(build_params(active, nctrl, pnr, 7, 1.8, 2.2, hx, fg), f"2leaf  +fg hx={hx}")
    # bigger tree
    active = [1, 1, 1, 1, 0, 0, 0, 0]
    nctrl = [1, 1, 1, 1, 0, 0, 0, 0]
    pnr = {0: [3], 1: [2], 2: [4], 3: [1]}
    run(build_params(active, nctrl, pnr, 3, 1.8, 2.2, 1.4, fg), "4leaf  +fg hx=1.4")
