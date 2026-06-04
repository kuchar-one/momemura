# Hanamura figure/table regeneration — runbook

Regenerates the Hanamura **before/after** data and the `tab:hanamura` numbers for
the canonical trio {|+>, |H>, |T>} with the **bug-fixed** heralding code, then
feeds the existing thesis plotter. Split into a GPU step (cluster) and a CPU step
(plot + LaTeX) because the figure data needs JAX + thewalrus.

## Why this was needed
- `wigner_pareto_pairs.npz` and `tab:hanamura` predate the `compute_equivalent_gaussian`
  fix (transposed BS / parity), so their squeezings + Hanamura gains came from buggy moments.
- The old "before" states were rebuilt via `heralded_output` on the moment-reduced
  Gaussian ("path-3"), which collapses high-energy displaced-squeezed states onto their
  even-parity core (visible in the cache: `T_1` before piled at Fock 20–23).
- Fix: **before** = trusted path-1 breeding sim (`utils.compute_heralded_state`).
  **after** = the **Hanamura core state** `(a^dag + s0' a + delta0')^{n'}|0>` built directly
  from the reduced/damped control parameters (well-conditioned, no thewalrus herald). This
  replaced the earlier `heralded_output` route, which still mis-reconstructed the highly-squeezed
  (11–12 dB) generators — even no-reduction rows aligned at only ~0.7 instead of ~1.0, the tell
  that the herald (not the photon count) was the ill-conditioned step. Per Sec. V (single signal
  mode => control Schmidt rank r=1; Theorem 9), a control mode detecting `n_m=0` contributes only a
  Gaussian filter (absorbed by alignment), so the core form is **exact whenever exactly one control
  mode fires** (`k_eff = #{m : n_m>=1} == 1`), regardless of how many vacuum-detection modes inflate
  `k_control`. The script keys on `k_eff` (recorded per row, with per-mode `n0`/`n1`). For `k_eff>1`
  the multimode output has no closed form (Theorem 11 is recursive), so the script falls back to the
  architecture-rule herald and flags it (`after_source = "herald_fallback"`).
- Built-in check: the script aligns the core "before" against the trusted path-1 "before" and
  records `core_validation_fid` per row. Near 1.0 confirms the core reconstruction is faithful
  (so the core "after" is trustworthy); a low value flags a generator that isn't single-mode and
  needs the multimode treatment.
- The Pareto **selection** is preserved exactly: each row of the existing
  `wigner_pareto_data.json` is pinned to its genotype by full-precision probability,
  so `tab:breeding_pareto` and the Pareto figure are unaffected.

## Data the script needs
- `experiments/` repertoires — **gitignored and huge, NOT in this commit**; they already live on
  the cluster. Run from a checkout that sits next to (or symlinks) `experiments/`.
- The Pareto **selection spec** is bundled in this commit at
  `scripts/data/hanamura_selection_spec.json` (a copy of the thesis `wigner_pareto_data.json`),
  so the script pins the same genotypes without needing the `mgr` repo. Override with
  `--select-from <path>` or ignore it with `--reselect`.

## Step 1 — on the A5000 node (needs jax + thewalrus + the repertoires)
```bash
cd <repo-root>                       # the momemura repo
python scripts/gen_hanamura_data.py --out outputs       # or --out ../mgr/scripts if mgr is a sibling
```
Writes into `../mgr/scripts` (where `gen_wigner_pareto.py` reads them):
- `chosen_genotypes.npz` (+ `chosen_genotypes_meta.json`) — the pinned genotypes (this is the
  cache that was missing; keep it).
- `wigner_pareto_data.json` — per-target rows with **refreshed** `gbs_sq_db` + Hanamura columns.
- `wigner_pareto_pairs.npz` / `wigner_pareto_pairs_meta.json` — before/after state vectors.
- `hanamura_table.csv` — the `tab:hanamura` numbers (Nc→Nc′, P→P′, gain, r_max→r_max′).

Flags: `--reselect` re-derives the front from scratch (only if you *want* to change the
selection — the front has grown since the figure was made); `--reduction-factor` (default 3.0);
`--herald-cap` (default 48).

Sanity checks to eyeball in the log:
- each target prints `reused 5/5 existing Pareto rows`;
- "before" states for the magic states should **not** be parity-collapsed (compare even/odd mass);
- Hanamura may report degraded gain (`x<1`) or invalid (`han_ok=False`) on asymmetric
  magic configs — that is expected and should be reflected honestly in the table/text.

## Step 2 — plot + LaTeX (CPU; numpy/scipy/matplotlib only)
```bash
python ../mgr/scripts/gen_wigner_pareto.py     # -> figures/wigner_hanamura.pdf (+ wigner_pareto.pdf)
```
Then update `parts/4chapter.tex`:
- replace the `tab:hanamura` body from `hanamura_table.csv`;
- re-check the Hanamura subsection prose against the new gains (note any degraded/invalid magic cases).

I (Claude) can do Step 2 and the tex edits here once the six files are synced back into
`mgr/scripts`.

## Note on the figure's row selection
`gen_wigner_pareto.py` currently shows, per target, the highest-gain pair with `prob_gain >= 1.5`.
With bug-free numbers the magic states may have no row above that threshold (gain can degrade),
which would blank their figure row. If that happens, switch the selection to "largest valid
Nc reduction with a valid before+after pair" so all three rows render — decide this after seeing
the regenerated `wigner_pareto_pairs_meta.json`.
