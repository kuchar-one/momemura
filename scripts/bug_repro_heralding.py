"""Reproduce + isolate the three-path heralding disagreement.

Generates random 00B genotypes (no cluster cache needed) and compares:
  path 1: frontend.utils.compute_state_with_jax        (JAX breeding sim)
  path 2: frontend.independent_verifier.verify_circuit (thewalrus + scipy)
  path 3: gaussian_decomposition + gbs_optimizer.heralded_output (moment space)
"""
import os, sys, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# qutip is only used by frontend.utils for Wigner plotting; stub it if absent.
try:
    import qutip  # noqa: F401
except ImportError:
    sys.modules["qutip"] = types.ModuleType("qutip")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.genotypes.genotypes import get_genotype_decoder
from frontend.utils import compute_state_with_jax
from frontend.independent_verifier import verify_circuit
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.gbs_optimizer import heralded_output, align_states

CUT = int(os.environ.get("REPRO_CUT", "20"))
PNR_MAX = int(os.environ.get("REPRO_PNRMAX", "15"))
NSEED = int(os.environ.get("REPRO_NSEED", "5"))
RS = float(os.environ.get("REPRO_RSCALE", "1.2"))
CFG = {"genotype": "00B", "depth": 3, "modes": 3, "pnr_max": PNR_MAX, "cutoff": 30,
       "r_scale": RS, "d_scale": RS * 0.5, "hx_scale": 1.5}


def tn(o):
    if isinstance(o, dict):
        return {k: tn(v) for k, v in o.items()}
    if hasattr(o, "tolist"):
        return np.asarray(o)
    if isinstance(o, (list, tuple)):
        return [tn(x) for x in o]
    return o


def make_params(seed):
    dec = get_genotype_decoder("00B", depth=3, config=CFG)
    L = dec.get_length(3)
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(L) * 0.8
    return tn(dec.decode(jnp.asarray(g), CUT)), dec


def run_all(params):
    psi1, _ = compute_state_with_jax(params, cutoff=CUT, pnr_max=PNR_MAX)
    psi1 = np.asarray(psi1).ravel(); psi1 = psi1 / (np.linalg.norm(psi1) + 1e-300)

    res2 = verify_circuit(params, cutoff=CUT, pnr_max=PNR_MAX)
    psi2 = np.asarray(res2["state"]).ravel(); psi2 = psi2 / (np.linalg.norm(psi2) + 1e-300)

    eq = compute_equivalent_gaussian(params)
    psi3, _ = heralded_output(eq["cov"], eq["mu"], eq["signal_idx"],
                              eq["control_idx"], eq["pnr_outcomes"], cutoff=CUT)
    psi3 = np.asarray(psi3).ravel(); psi3 = psi3 / (np.linalg.norm(psi3) + 1e-300)
    return psi1, psi2, psi3


if __name__ == "__main__":
    S0 = int(os.environ.get("REPRO_S0", "0"))
    for seed in range(S0, S0 + NSEED):
        params, dec = make_params(seed)
        active = np.asarray(params["leaf_active"]).astype(bool)
        nctrl = np.asarray(params["leaf_params"]["n_ctrl"])
        psi1, psi2, psi3 = run_all(params)
        F12, _ = align_states(psi1, psi2, len(psi1), align_cut=len(psi1))
        F13, _ = align_states(psi1, psi3, len(psi1), align_cut=len(psi1))
        F23, _ = align_states(psi2, psi3, len(psi2), align_cut=len(psi2))

        def odd(psi):
            p = np.abs(psi) ** 2
            return float(p[1::2].sum() / (p.sum() + 1e-300))
        print(f"seed {seed} | active={active.astype(int)} nctrl={nctrl} | "
              f"F12={F12:.4f} F13={F13:.4f} F23={F23:.4f} | "
              f"odd1={odd(psi1):.3f} odd3={odd(psi3):.3f}")
