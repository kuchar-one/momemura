# Bug: three heralded-output paths in `momemura` disagree

## TL;DR

The `momemura` codebase exposes three implementations that should produce
**the same heralded single-mode signal state** for the same circuit
parameters and the same PNR pattern; they do not. Pairwise
`fidelity_up_to_(single-mode)-Gaussian-unitary` is 0.5–0.7 for production
plus/H/T-target solutions, which is impossible for the same physical state
(it must be 1.0 up to a single-mode unitary). The discrepancy is therefore
**larger than a single-mode Gaussian**, so it is not a phase/displacement
convention but a real implementation bug somewhere in the mixing-tree
simulation. We need it found and fixed.

## Repo and how it runs

- Local repo: `/Users/kuchar/Nextcloud/vojtech/python/code/momemura/`
  (cluster mirror: `/cluster/home/kuchar/code/momemura/`).
- Production runs use exclusively the `00B` genotype on a depth-3 binary
  breeding tree with 3 modes per leaf (1 signal + 2 control), Fock cutoff
  30, `pnr_max=15`, `dynamic_limits`, hybrid emitter, point homodyne. The
  canonical command is in `chapter 4` of the thesis (see `Run
  configuration` paragraph in `mgr/parts/4chapter.tex`).
- **Point homodyne is mandatory.** Any finite homodyne window destroys
  purity and breaks the Gaussian-unitary-equivalence assumption that the
  Hanamura optimizer (`frontend/gbs_optimizer.py`) and the equivalent-GBS
  reduction (`frontend/gaussian_decomposition.py`) rely on.

## The three paths

For a decoded circuit `params` (output of
`get_genotype_decoder('00B', depth=3, config=cfg).decode(g, cutoff)`),
all three of the following should yield **the same heralded single-mode
signal state** (up to global phase):

1. **`frontend.utils.compute_state_with_jax(params, cutoff, pnr_max)`** —
   the JAX breeding-tree simulation used by the Streamlit frontend
   ("Output Wigner Function" panel). Heralds each leaf via
   `src.simulation.jax.runner.jax_get_heralded_state`, then runs
   `src.simulation.jax.composer.jax_superblock` (Fock-space BS via
   `jax_bs`, point homodyne via `jax_hermite_phi_matrix`), then
   `jax_apply_final_gaussian`. **This is the path the user trusts as the
   reference**, since it is what the frontend displays.
2. **`frontend.independent_verifier.verify_circuit(params, cutoff,
   pnr_max)`** — the CPU/numpy thewalrus cross-check
   (`_herald_leaf` via `thewalrus.quantum.state_vector` with
   `post_select`, then `_mix_pair` using `_fock_bs_unitary` and
   `_hermite_phi`).
3. **`frontend.gaussian_decomposition.compute_equivalent_gaussian(params)`
   + `frontend.gbs_optimizer.heralded_output(eq.cov, eq.mu,
   eq.signal_idx, eq.control_idx, eq.pnr_outcomes, cutoff)`** — the
   moment-space reduction the Hanamura optimizer operates on. Builds the
   full pre-PNR Gaussian state by composing leaf Gaussian moments and
   applying BS + Gaussian homodyne projection (`get_bs_symplectic`,
   `measure_homodyne`) in symplectic moment space, then heralds with
   `thewalrus.quantum.state_vector(post_select=...)`.

## Reproduction

Run the script below in the repo's venv (which has jax, thewalrus, qutip
installed). The genotype + config for one production-archive Pareto point
are pre-cached at
`/Users/kuchar/Library/Application Support/Claude/local-agent-mode-sessions/9b2092a1-2e06-475a-8981-2d49910900e7/28f99f3a-0c4a-4dfe-b80c-1a8f8a2b931e/local_f4c0361d-9e68-4508-aa4b-8d86332946dc/outputs/chosen_genotypes.npz`
and
`.../chosen_configs.json`; the key `plus_3` is one of the chosen
representatives for the logical-|+_L> archive
(`experiments/00B_c30_a1p00_b1p00`).

```python
# scripts/bug_repro_heralding.py
import sys, json, numpy as np
sys.path.insert(0, '<REPO_ROOT>')           # adjust
import jax, jax.numpy as jnp
from src.genotypes.genotypes import get_genotype_decoder
from frontend.utils import compute_state_with_jax           # path 1
from frontend.independent_verifier import verify_circuit    # path 2
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.gbs_optimizer import heralded_output, align_states  # path 3

CUT, PNR_MAX = 24, 15

def tn(o):
    if isinstance(o, dict): return {k: tn(v) for k, v in o.items()}
    if hasattr(o, 'tolist'): return np.asarray(o)
    if isinstance(o, (list, tuple)): return [tn(x) for x in o]
    return o

g = np.load('<...>/chosen_genotypes.npz')['plus_3']
cfg = json.load(open('<...>/chosen_configs.json'))['plus_3']; cfg.pop('_meta', None)

dec = get_genotype_decoder(cfg['genotype'], depth=3, config=cfg)
params = tn(dec.decode(jnp.asarray(g), int(cfg['cutoff'])))

# Path 1: JAX breeding sim (frontend reference)
psi1, _ = compute_state_with_jax(params, cutoff=CUT, pnr_max=PNR_MAX)
psi1 = np.asarray(psi1).ravel(); psi1 /= np.linalg.norm(psi1)

# Path 2: independent_verifier
res2 = verify_circuit(params, cutoff=CUT, pnr_max=PNR_MAX)
psi2 = np.asarray(res2['state']).ravel(); psi2 /= np.linalg.norm(psi2)

# Path 3: equivalent-GBS + thewalrus state_vector
eq = compute_equivalent_gaussian(params)
psi3, _ = heralded_output(eq['cov'], eq['mu'], eq['signal_idx'],
                          eq['control_idx'], eq['pnr_outcomes'], cutoff=CUT)
psi3 = np.asarray(psi3).ravel(); psi3 /= np.linalg.norm(psi3)

for a, b, name in [(psi1, psi2, '1-vs-2 (breed vs iv)'),
                   (psi1, psi3, '1-vs-3 (breed vs equiv-GBS)'),
                   (psi2, psi3, '2-vs-3 (iv vs equiv-GBS)')]:
    F, _ = align_states(a, b, len(a), align_cut=len(a))
    print(f'{name}:  F_align = {F:.4f}')
```

What I see on `plus_3` (and similarly on every plus_*, H_*, T_* tested):

```
1-vs-2 (breed vs iv)        :  F_align ≈ 0.70
1-vs-3 (breed vs equiv-GBS) :  F_align ≈ 0.52
2-vs-3 (iv vs equiv-GBS)    :  F_align ≈ 0.53
```

These should all be 1.000. `align_states` searches over single-mode
Gaussian unitaries (5 params: dx, dp, r, phi, varphi); if it cannot reach
≈1 then the residual is **larger than a single-mode Gaussian**, which is
impossible if all three paths are simulating the same heralded
single-mode state.

## Where to look

The three paths differ in how they perform **beam-splitter mixing** and
**point homodyne projection** on signal modes between leaves. Likely
suspects:

- BS angle/phase sign convention. Compare the 2×2 BS matrices:
  - `src/simulation/jax/composer.py` `jax_bs(theta, phi)` (Fock-space
    unitary built from `jax_beamsplitter_2x2`),
  - `frontend/independent_verifier.py` `_fock_bs_unitary(theta, phi,
    cutoff)`,
  - `frontend/gaussian_decomposition.py` `get_bs_symplectic(theta, phi,
    N_modes, modeA, modeB)`.
  All three claim to encode the same $\hat U_{\rm BS}(\theta,\phi)$ — verify on
  a tiny analytic case (one squeezed mode + vacuum, BS, no homodyne, no PNR)
  by comparing the output Gaussian moments to the Fock-space unitary
  applied to the same state.

- Point homodyne convention. Three implementations:
  - JAX uses `jax_hermite_phi_matrix(x, cutoff)` ⊗ inner product on the
    signal modes' Fock vector.
  - `independent_verifier` uses `_hermite_phi(x, cutoff)` followed by the
    same inner-product idiom.
  - `gaussian_decomposition.measure_homodyne` applies the Gaussian
    conditional formula
    `V_new = V_BB - V_BA V_AA^{-1} V_AB`,
    `mu_new = mu_B + (x_val - mu_A) V_BA / V_AA`
    on the SYMPLECTIC covariance + mean (point-x projection on mode
    `idx`). Check the sign of `(x_val - mu_A)`, the choice of `x` vs `p`
    quadrature (`x_idx = idx`, `p_idx = idx + N_modes` — xp-ordered),
    and that the same mode (`B` = the one homodyned away) is dropped in
    all three implementations.

- Which mode of the BS pair is kept and which is homodyned. In
  `gaussian_decomposition.compute_equivalent_gaussian` the second input
  (`idxB`) is homodyned and `sigA_leaf` is propagated up; the JAX
  `jax_superblock` and `independent_verifier._mix_pair` must use the
  same orientation. A swapped pair changes the output by a beam-splitter
  reflection that is **not** a single-mode unitary, which matches the
  symptom (`align_states` cannot bridge the disagreement).

- hbar / scale convention. The repo uses ħ=2 throughout (xp-ordering,
  vacuum covariance = I). `compute_equivalent_gaussian` uses `HBAR=2.0`
  for displacement scaling in `apply_final_gaussian_symplectic` (`scale =
  sqrt(2*HBAR)`). Confirm the displacement scale is consistent with the
  ħ used by `jax_apply_final_gaussian` and `_apply_final_gaussian` in
  `independent_verifier`.

- Final-Gaussian operator ordering. JAX `jax_apply_final_gaussian` builds
  `U_disp @ U_rot @ U_squeeze` (squeeze, then rotate, then displace). The
  symplectic in `gaussian_decomposition.apply_final_gaussian_symplectic`
  uses `S_2x2 = R_mat @ S_sq` followed by `mu += scale * disp`. Verify
  the order matches and `phi` (squeezing angle) vs `varphi` (rotation
  angle) are not swapped.

## Suggested investigation order

1. **Minimal repro.** Build a "nano breeding": two leaves, leaf 0 = single
   squeezed signal (no controls), leaf 1 = single squeezed signal with
   one control mode heralded on `n=1`, one BS mixing them, one homodyne
   at `x=0.3`, no final Gaussian. Run all three paths at `cutoff=15` and
   diff the resulting state vectors. Fix the orientation by hand until
   they agree.
2. Turn on the final Gaussian; verify they still agree.
3. Scale to a 4-leaf depth-2 tree, then to the full depth-3 8-leaf
   superblock.
4. Once `compute_state_with_jax` and `independent_verifier` agree on the
   full superblock, fix `compute_equivalent_gaussian` against them. The
   bug is most likely in `get_bs_symplectic` or `measure_homodyne` (or
   their orientation in the mixing loop).

## Acceptance criterion

Add a test in `tests/` that, for a handful of randomly-decoded `00B`
genotypes at `cutoff=20`, asserts

```python
F_align(compute_state_with_jax, verify_circuit) > 0.999
F_align(compute_state_with_jax, equivalent_GBS_herald) > 0.999
```

with `align_states(..., align_cut=cutoff)`. Once that passes, the
Hanamura before/after Wigner figure (figure script:
`mgr/scripts/gen_wigner_pareto.py`, data extraction:
`outputs/gen_psi1_lift.py`) will produce visually-identical before/after
pairs for moderate reductions and the Streamlit frontend's "Optimized
GBS Architecture" view will stop crashing on the `Input matrix is not
positive definite` path (root cause likely the same convention bug
feeding a non-PSD covariance into `purify_control → williamson`).

## Don't touch

- `experiments/**` — read-only optimisation results downloaded from the
  cluster. Loading is fine; don't modify.
- `mgr/parts/4chapter.tex` and `mgr/figures/*` — thesis content, owned by
  the user. The fix should land in the `momemura` code only.
