"""Regression test: the three heralded-output paths must agree.

For a decoded 00B circuit, all three implementations should yield the same
heralded single-mode signal state (up to a single-mode Gaussian unitary, which
`align_states` absorbs):

  path 1  frontend.utils.compute_state_with_jax        (JAX breeding sim, reference)
  path 2  frontend.independent_verifier.verify_circuit (thewalrus + scipy)
  path 3  gaussian_decomposition.compute_equivalent_gaussian
          + gbs_optimizer.heralded_output              (symplectic moment space)

This guards the bug where `get_bs_symplectic` stored the TRANSPOSE of the
beam-splitter mode transformation used by the Fock-space paths (a reflection
flip, not a single-mode unitary), which made path 3 disagree with paths 1/2.

Note on conditioning: a randomly-decoded genotype can herald a near-impossible
PNR pattern (e.g. detecting 13-14 photons from a weakly-squeezed mode). The
heralded state is then numerically ill-defined (herald probability underflows),
so such cases are skipped — the equivalence is a statement about a well-defined
physical state. We therefore use a modest pnr_max and skip vanishing-probability
heralds.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

jax = pytest.importorskip("jax")
pytest.importorskip("thewalrus")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from src.genotypes.genotypes import get_genotype_decoder  # noqa: E402
from frontend.utils import compute_state_with_jax  # noqa: E402
from frontend.independent_verifier import verify_circuit  # noqa: E402
from frontend.gaussian_decomposition import compute_equivalent_gaussian  # noqa: E402
from frontend.gbs_optimizer import heralded_output, align_states  # noqa: E402

# cutoff=24 (>= the criterion's 20) gives convergence headroom: paths 1/2 share
# the same per-leaf Fock truncation, while path 3 is exact in moment space and
# only truncates at the final herald, so the three agree to ~3 nines once the
# cutoff comfortably covers the heralded state's support.
CUT = 24
PNR_MAX = 3  # keep heralds well-conditioned for random genotypes
PROB_FLOOR = 1e-9  # below this the heralded state is numerically undefined
CFG = {
    "genotype": "00B", "depth": 3, "modes": 3, "pnr_max": PNR_MAX, "cutoff": 30,
    "r_scale": 1.0, "d_scale": 0.5, "hx_scale": 1.5,
}


def _to_numpy(o):
    if isinstance(o, dict):
        return {k: _to_numpy(v) for k, v in o.items()}
    if hasattr(o, "tolist"):
        return np.asarray(o)
    if isinstance(o, (list, tuple)):
        return [_to_numpy(x) for x in o]
    return o


def _decode(seed):
    dec = get_genotype_decoder("00B", depth=3, config=CFG)
    g = np.random.default_rng(seed).standard_normal(dec.get_length(3)) * 0.8
    return _to_numpy(dec.decode(jnp.asarray(g), CUT))


def _three_paths(params):
    psi1, prob1 = compute_state_with_jax(params, cutoff=CUT, pnr_max=PNR_MAX)
    psi1 = np.asarray(psi1).ravel()

    res2 = verify_circuit(params, cutoff=CUT, pnr_max=PNR_MAX)
    psi2 = np.asarray(res2["state"]).ravel()

    eq = compute_equivalent_gaussian(params)
    psi3, _ = heralded_output(eq["cov"], eq["mu"], eq["signal_idx"],
                              eq["control_idx"], eq["pnr_outcomes"], cutoff=CUT)
    psi3 = np.asarray(psi3).ravel()
    return psi1, psi2, psi3, float(prob1)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_three_paths_agree(seed):
    params = _decode(seed)
    psi1, psi2, psi3, prob1 = _three_paths(params)

    if prob1 < PROB_FLOOR or np.linalg.norm(psi1) == 0:
        pytest.skip(f"herald probability {prob1:.2e} too small to define a state")

    f12, _ = align_states(psi1, psi2, len(psi1), align_cut=len(psi1))
    f13, _ = align_states(psi1, psi3, len(psi1), align_cut=len(psi1))

    assert f12 > 0.999, f"path1 vs path2 (Fock cross-check) F={f12:.4f}"
    assert f13 > 0.999, f"path1 vs path3 (equivalent-GBS) F={f13:.4f}"


if __name__ == "__main__":
    for s in range(6):
        p = _decode(s)
        a, b, c, pr = _three_paths(p)
        if pr < PROB_FLOOR:
            print(f"seed {s}: SKIP (prob={pr:.2e})")
            continue
        f12, _ = align_states(a, b, len(a), align_cut=len(a))
        f13, _ = align_states(a, c, len(a), align_cut=len(a))
        print(f"seed {s}: prob={pr:.2e} F12={f12:.4f} F13={f13:.4f}")
