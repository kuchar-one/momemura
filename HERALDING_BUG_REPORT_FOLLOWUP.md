# Follow-up: BS-transpose fix is partial; path 3 still diverges on production solutions

Thanks for the BS-symplectic transpose fix in
`frontend/gaussian_decomposition.get_bs_symplectic` — paths 1 (JAX
breeding sim) and 2 (`independent_verifier`) now agree to F=1.000, which
they did not before. But on production-archive solutions, path 3
(`compute_equivalent_gaussian` + `heralded_output`) still diverges from
1/2. The regression test added in
`tests/test_heralding_path_consistency.py` happens to miss this because
it uses gentle scales and random genotypes; the optimized production
genotypes — with strong squeezing, large displacements and large PNR
outcomes — exercise a code path that is still broken.

## Concrete evidence (post-fix, with `__pycache__` cleared)

Run, in the repo's venv, on a cached optimized `|+_L>` Pareto point
(saved at `outputs/chosen_genotypes.npz` key `plus_3`):

```
p1 top|c|^2: [0.274 0.005 0.123 0.129 0.03  0.102]   # JAX breeding sim
p2 top|c|^2: [0.274 0.005 0.123 0.129 0.03  0.102]   # independent_verifier
p3 top|c|^2: [0.486 0.001 0.184 0.001 0.105 0.001]   # compute_equivalent_gaussian + heralded_output
F12 = 1.0000   F13 = 0.5313   F23 = 0.5313
```

So:

- Paths 1 and 2 match perfectly (your fix held there).
- Path 3 is a **strictly even-Fock state** (|c₁|²≈|c₃|²≈|c₅|²≈10⁻³), while
  paths 1/2 have visibly mixed parity (|c₃|² = 0.129).
- The mismatch is therefore larger than a single-mode Gaussian unitary,
  so `align_states` still cannot bridge it — same symptom as before but
  with a different fingerprint (parity asymmetry rather than a generic
  rotation).

## Why the regression test didn't catch it

`tests/test_heralding_path_consistency.py` uses

```python
CFG = {"genotype": "00B", "depth": 3, "modes": 3, "pnr_max": 3, ...,
       "r_scale": 1.0, "d_scale": 0.5, "hx_scale": 1.5}
```

and freshly random genotypes. Production runs use `pnr_max=15`,
`r_scale≈1.87` (with `--dynamic-limits`), `d_scale≈2.24`, `hx_scale≈1.37`,
and **optimized** genotypes that exercise large PNR outcomes (up to 15
per detector) and the high-squeezing regime of the leaf moments. The gentle
test does not exercise this regime, so it passes while real solutions
still disagree.

## What to do

1. **Extend the regression test** to use the actual production
   configuration and at least one cached optimized genotype (not just
   random ones). I've saved a few keys in
   `outputs/chosen_genotypes.npz` (`plus_3`, `plus_4`, `H_4`, `T_2`,
   `T_4`) together with their configs in `outputs/chosen_configs.json`;
   please copy them into `tests/data/` (small files, ≤30 KB total) and
   include `plus_3` as a fixture in the test. A minimal reproducer:

```python
# tests/test_heralding_path_consistency_production.py
import os, json, numpy as np, pytest, jax.numpy as jnp
from src.genotypes.genotypes import get_genotype_decoder
from frontend.utils import compute_state_with_jax
from frontend.independent_verifier import verify_circuit
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.gbs_optimizer import heralded_output, align_states

CUT = 24

def _to_numpy(o):
    if isinstance(o, dict): return {k: _to_numpy(v) for k, v in o.items()}
    if hasattr(o, "tolist"): return np.asarray(o)
    if isinstance(o, (list, tuple)): return [_to_numpy(x) for x in o]
    return o

@pytest.mark.parametrize("key", ["plus_3", "plus_4", "H_4", "T_2", "T_4"])
def test_three_paths_agree_on_production(key):
    base = os.path.join(os.path.dirname(__file__), "data")
    g = np.load(os.path.join(base, "chosen_genotypes.npz"))[key]
    cfg = json.load(open(os.path.join(base, "chosen_configs.json")))[key]
    cfg.pop("_meta", None)
    dec = get_genotype_decoder(cfg["genotype"], depth=3, config=cfg)
    params = _to_numpy(dec.decode(jnp.asarray(g), int(cfg["cutoff"])))
    pnr_max = int(cfg.get("pnr_max", 15))

    p1, _ = compute_state_with_jax(params, cutoff=CUT, pnr_max=pnr_max)
    p1 = np.asarray(p1).ravel()
    p2 = np.asarray(verify_circuit(params, cutoff=CUT, pnr_max=pnr_max)["state"]).ravel()
    eq = compute_equivalent_gaussian(params)
    p3, _ = heralded_output(eq["cov"], eq["mu"], eq["signal_idx"],
                            eq["control_idx"], eq["pnr_outcomes"], cutoff=CUT)
    p3 = np.asarray(p3).ravel()

    f12, _ = align_states(p1, p2, len(p1), align_cut=len(p1))
    f13, _ = align_states(p1, p3, len(p1), align_cut=len(p1))
    assert f12 > 0.999, f"{key}: paths 1/2 disagree, F={f12:.4f}"
    assert f13 > 0.999, f"{key}: paths 1/3 disagree, F={f13:.4f}"
```

Today this test fails on every key with F13 ≈ 0.53.

2. **The parity asymmetry is a strong clue.** A heralded state's parity
   in the Fock basis is preserved by Gaussian-unitary alignment up to a
   complex phase — so a state that is "almost exactly parity-even" cannot
   be aligned to a state with substantial odd-Fock support by a 5-param
   single-mode unitary. The two states are physically different. Likely
   culprits in path 3 (now that BS is consistent):

   - **Homodyne sign / quadrature choice in
     `gaussian_decomposition.measure_homodyne`.** It projects mode `idx`
     onto `x_val` using
     `V_new = V_BB − c cᵀ / V_AA`,
     `mu_new = mu_B + (x_val − mu_A) · c / V_AA`,
     where `c = V[:, x_idx]` (the column at the measured mode's
     x-quadrature) and the dropped row is `[idx, idx+N]`. The JAX
     `jax_superblock` does the projection in Fock space via
     `jax_hermite_phi_matrix`, and `independent_verifier._mix_pair` does
     it the same way. Compare the **sign of `(x_val − mu_A)`** and the
     **dropped quadrature** (x vs p) between the two; a sign flip on the
     conditional mean gives precisely a parity-of-output difference for
     non-zero homodyne `x`.

   - **Which BS output is homodyned away.** In
     `gaussian_decomposition.compute_equivalent_gaussian` the loop does
     `measure_homodyne(..., idxB, ...)` after `get_bs_symplectic(...,
     idxA, idxB)`. Confirm `jax_superblock`/`_mix_pair` also homodyne the
     second (B) output of the BS — and not the first one. Now that
     `get_bs_symplectic` is no longer transposed, an inconsistent
     "which-output-is-measured" choice would appear precisely as the
     parity-asymmetry symptom we still see (the heralded mode then comes
     out of a different port of the corrected BS).

   - **The final single-mode Gaussian operation.** Compare
     `jax_apply_final_gaussian` (Fock-space, builds `U_disp @ U_rot @
     U_squeeze`) with
     `gaussian_decomposition.apply_final_gaussian_symplectic` (which
     constructs `S_2x2 = R_mat @ S_sq` and then adds `scale *
     disp` to the mean with `scale = sqrt(2*HBAR)`). The displacement
     scale and the rotation convention (`varphi` vs `phi`) must match
     bit-for-bit. With production `r_scale≈1.87` the squeezing in the
     final Gaussian is large, so a sign error here would no longer
     average out.

3. **Nano-repro at the production regime.** Re-run your nano-isolation
   under
   `r_scale=2.0`, `d_scale=2.5`, large homodyne value (say `x_val=2.0`)
   and a leaf heralding `n=10`. Iterate through the diagnostics until
   path 3 again matches paths 1/2. The current BS-only test won't
   re-flag this because BS is now correct in isolation; the remaining
   bug shows up only when BS + homodyne + (possibly) the final Gaussian
   are composed in a regime with large quadrature means.

## Note re: Streamlit "Input matrix is not positive definite"

I suspect this is the *same* underlying issue: a path-3 covariance that
already disagrees with the physical one feeds into `purify_control →
williamson`, which then rejects it as non-PSD. Fixing the remaining
convention mismatch in path 3 should silence the crash and let the
"Optimized GBS Architecture" panel render Hanamura's before/after Wigner
on the optimized production solutions.

## Cached test data (for convenience)

Already prepared:
- `outputs/chosen_genotypes.npz` (15 keys: `{plus,H,T}_{0..4}`)
- `outputs/chosen_configs.json` (matching configs)

These are the actual 5-per-target Pareto representatives the thesis
table uses. Drop them into `tests/data/` and reference from the test
above.

## Acceptance criterion (unchanged)

`F_align(p1, p2) > 0.999` and `F_align(p1, p3) > 0.999` for every
cached production genotype above, at cutoff 24, in the production
configuration (`pnr_max=15`, `r_scale≈1.87`, `d_scale≈2.24`,
`hx_scale≈1.37`).
