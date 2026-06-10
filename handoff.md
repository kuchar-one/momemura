# Handoff — Hanamura Wigner figure / pipeline validation (momemura)

> **UPDATE 2026-06-10: root cause FOUND — see `HANAMURA_VALIDATION_FINDINGS.md`.**
> No reconstruction bug: the breeding sim is cutoff-truncated (11–12 dB
> squeezing, tails ~0.1 at cutoff 30) and `plus_0`'s photon is physically
> decoupled (true output exactly Gaussian → the rank-1 check could never pass).
> New exact/stable herald: `gbs_optimizer.reduced_herald` (+13 passing tests).
> Remaining step: run `scripts/validate_convergence.py` on the cluster to
> confirm F(path-1 → exact) → 1 at high cutoff. The doc below is the OLD state.

**Status: blocked on a verification question, not on a feature.** The rigorous,
moment-space result (the `tab:hanamura` table) is done and trustworthy. What is
**not** yet established is that the *Fock-space reconstructions* of the heralded
states agree end-to-end, and a single-photon sanity check is currently failing.
Until that's understood, do **not** ship the before/after Wigner figure.

This doc explains the goal, what's solid, the exact open problem, why the latest
validation script hung, the leading hypotheses, and concrete next steps.

---

## 1. Goal

Finalize Chapter 4's Hanamura analysis for the canonical trio
**{|+⟩_L, |H⟩_L, |T⟩_L}** of the generalized-breeding ("momemura") optimizer:

1. A `tab:hanamura` table: photon reduction `N_c → N_c'`, probability gain, and
   squeezing `r_max → r_max'` for each Pareto-selected protocol. **DONE.**
2. A before/after **Wigner figure** (`figures/wigner_hanamura.pdf`) showing the
   heralded output before vs. after the Hanamura control-parameter optimization,
   visually preserved up to a Gaussian unitary. **BLOCKED — see §4.**

The user's acceptance criterion (correct and non-negotiable): produce **three
Wigner functions that visibly match** —
**optimizer result → equivalent-GBS reconstruction → Hanamura-optimized GBS** —
for a few representative solutions. If those three agree, the pipeline is proven
working. They currently cannot be shown to agree.

The canonical trio maps to `00B` (depth-3, balanced-BS, point-homodyne window=0)
experiment groups:
- `|+⟩`: `00B_c30_a1p00_b1p00`  (α=1, β=1; u=(1,0,0))
- `|H⟩`: `00B_c30_a1p41_b1p41`  (α=√2, β=1+i; u=(0.707,0.707,0))
- `|T⟩`: `00B_c30_a2p73_b1p41`  (α=2.732, β=1+i; u=(0.577,0.577,0.577))

---

## 2. Architecture: the four reconstruction paths

A breeding solution is scored by the JAX/QDax MOME optimizer, then "deconstructed"
into an equivalent Gaussian-boson-sampling (GBS) generator (vacuum → Gaussian
unitary → PNR pattern on `k` control modes + 1 heralded signal mode). The Hanamura
method (PRX 16, 021034; arXiv 2509.06255, PDF read in full) post-processes the
generator's **moments** to reduce detected photons and boost probability.

There are four ways the single-mode heralded output can be reconstructed in Fock
space; the validation hinges on whether they agree:

| Path | What | Code | Trust |
|---|---|---|---|
| **A** | optimizer output (breeding sim) | `frontend/utils.compute_heralded_state` (JAX, path-1) | ground truth the optimizer actually scored |
| **B** | independent reconstruction | `frontend/independent_verifier.verify_circuit` (thewalrus+scipy, path-2) | independent cross-check of A |
| **C** | equivalent-GBS herald | `frontend/gbs_optimizer.heralded_output` on `compute_equivalent_gaussian` output (thewalrus `state_vector`, path-3) | **numerically fragile** (see §4, §5) |
| **D** | Hanamura-reduced output | core state if one control mode fires, else `reduced_full_state` herald | depends on C's machinery |

**Key fact established by the previous bug-fix agent (in memory / bug reports):**
the equivalent-GBS *moments* (`compute_equivalent_gaussian` cov/mu) are bit-for-bit
correct, and paths A and B agree to F≥0.999 *in the production test config*. Path C
(`heralded_output` heralding the full collapsed generator) is the one shown to be
ill-conditioned at high energy/squeezing. **But A==B was never confirmed on the
actual `chosen_genotypes` solutions** — only on synthetic production-config states.
Confirming A==B on the real solutions is step 1 of the open work.

---

## 3. What is DONE and trustworthy

- **Heralding/moment bug fix** (transposed beam-splitter + parity in
  `frontend/gaussian_decomposition.py`) — committed, on `origin/master`, verified.
  See `HERALDING_BUG_REPORT*.md` and `tests/test_heralding_path_consistency*.py`.
- **`tab:hanamura` numbers** — regenerated bug-free in moment space
  (`outputs/hanamura_table.csv`). `han_ok` correctly flags degenerate cases
  (`plus_1` numerical blow-up → nan; `H_2`/`T_0` invalid no-gain). This table is
  independent of any Fock reconstruction and is the rigorous result. **Ready for
  the thesis.**
- **Pareto selection preserved exactly.** `scripts/gen_hanamura_data.py` pins each
  row of the existing `wigner_pareto_data.json` to its genotype by full-precision
  probability (do NOT re-select; the front grew and would drift). 5/5 rows per
  target reproduce, all `00B`.
- **`chosen_genotypes.npz` regenerated** (it was missing at the project start;
  re-derivable from the repertoires via the pinning above).
- **Multimode Hanamura formalism understood** (paper Sec. V): single signal mode ⇒
  control covariance Schmidt rank `r=1`; a control mode detecting `n_m=0`
  contributes only a Gaussian filter (Theorem 6, wave form); there is **no
  closed-form multimode core state** (Theorem 11 is recursive). So the single-mode
  core state `(â†+s₀â+δ₀)ⁿ|0⟩` is exact only when exactly one control mode fires
  *and* it is uncorrelated with the others.

### Files / commits

`origin/master` (cluster has these):
- `frontend/gaussian_decomposition.py`, `frontend/gbs_optimizer.py`, `frontend/app.py` — bug-fixed
- `pareto_report.py` — Pareto extraction + selection (imported by the scripts)
- `scripts/gen_hanamura_data.py` — regenerate `chosen_genotypes.npz`, refreshed
  `wigner_pareto_data.json`, `wigner_pareto_pairs.npz`(+meta), `hanamura_table.csv`
- `scripts/data/hanamura_selection_spec.json` — bundled selection spec
- `scripts/HANAMURA_REGEN_RUNBOOK.md` — how to run on the cluster
- `scripts/validate_pipeline.py` — the validation harness (commit `2a96c36`; **has a
  performance bug, see §5**)
- `mgr/scripts/gen_wigner_pareto.py` (separate thesis repo) — the thesis-style
  renderer (numpy/scipy/matplotlib; inferno + `PlateauTwoSlopeNorm(0,0.03,±0.23)`,
  axes in x/√π). This is the plotting reference for any new figure.

Untracked clutter left in `scripts/`: `bug_*.py`, `debug_*.py` — diagnostic
scrap from the bug-fix phase, safe to ignore/delete.

---

## 4. The open problem: reconstructions don't agree, and a rank-1 sanity check fails

From the latest good run (`outputs/wigner_pareto_pairs_meta.json`), the core-state
"after" was validated by aligning the **core "before"** against the trusted
**path-1 "before"** (`core_validation_fid`, should be ≈1.0 when the core form
applies):

```
plus_0  Nc 1->1   k_eff=1   coreFid=0.69
plus_3  Nc 2->2   k_eff=1   coreFid=0.79
H_0     Nc 10->4  k_eff=1   coreFid=0.51
H_1     Nc 1->1   k_eff=1   coreFid=0.53
T_0     Nc 2->2   k_eff=1   coreFid=0.86   (s0≈4190, degenerate near-pure ctrl mode)
T_1     Nc 2->2   k_eff=1   coreFid=0.27
T_2     Nc 14->6  k_eff=1   coreFid=0.66
(35->13 rows: k_eff=3 -> no closed form -> herald_fallback)
```

**The smoking gun:** `plus_0` is a **single-photon** herald (`N_c=1`), so its output
*must* be a rank-1 core state up to a Gaussian unitary. Yet under exhaustive
alignment (120 random restarts, squeezing-seeded) its path-1 state caps at **0.69**
against *any* rank (0/1/2/3) core — flat. That is physically impossible for a clean
rank-1 state. So one of these is true:

1. The **path-1 reconstruction itself is wrong/imprecise** for these solutions
   (e.g. float32 truncation — see §6), so its "before" isn't actually the clean
   heralded output; or
2. The single-mode **core framework genuinely doesn't apply** because the firing
   control mode is **correlated** with the `n_m=0` spectator modes (the output then
   depends on the full joint `(C,β,n)`, not one mode's `(s₀,δ₀)`); or
3. The state genuinely carries **more structure than `N_c` implies** (the `photons`
   descriptor ≠ stellar rank for these breeding circuits).

I ruled OUT the alignment metric being the culprit: `align_states` correctly scores
Fock |2⟩ at 0.19 and a genuinely-near-Gaussian small cat at ~0.90, and a synthetic
Gaussian-unitary round-trip recovers ≈0.99 up to ~1.3 nat of squeezing (degrading to
~0.93 by 1.6 nat). So `align_states` is usable but **loses accuracy above ~1.4 nat
of squeezing** — relevant because these generators sit at 11–12 dB ≈ 1.3–1.4 nat.

**Net:** we cannot currently demonstrate optimizer ≈ GBS ≈ Hanamura in Fock space,
and a trivial `N_c=1` case fails the rank check. This must be resolved before any
Wigner figure is credible.

---

## 5. Why `validate_pipeline.py` hung for days (and how to fix it)

The run printed the `|+⟩` selection line, a JAX x64 warning, then hung inside the
**first** solution before any `[plus_0]` line. Root cause: **path C
(`heralded_output` → thewalrus `state_vector`) is computed on the *full collapsed
generator* at `cutoff=40`, plus a second `cutoff2=60` "conditioning probe."**
`state_vector` fills the free signal mode's amplitudes `0..cutoff-1`, and each
amplitude is a loop hafnian whose size grows with the Fock index — so the cost is
roughly factorial in `cutoff`. At cutoff 40–60 on an 11–12 dB generator this is
effectively non-terminating. (The earlier `gen_hanamura_data.py` rarely hit this
because it used path-1 for "before" and the core state for "after"; it only called
`heralded_output` in the `k_eff>1` fallback.)

**Fixes the next agent should make to the harness:**
- Cap the path-C cutoff hard (≤ ~16–20) and **delete the `cutoff2=60` probe** (or
  make it ≤ 20). Better: time-box/skip path C when `gbs_sq_db` is high or
  `sum(n0)`/cutoff make the loop hafnian intractable.
- Skip high-`N_c` rows by default (`--max-nc 16` exists but path C is still the
  bottleneck even for low `N_c` because cutoff drives the cost, not `N_c`).
- The `--rank` effective-rank scan is ~60 `align_states` calls per row (minutes,
  not days) — fine once path C is bounded, but consider gating it.
- Set **`JAX_ENABLE_X64=1`** before running (see §6).

---

## 6. Leading hypotheses for the root cause (ranked, with how to test)

1. **float32 precision in path-1 (most actionable).** The cluster run warned:
   `complex128 ... truncated to complex64 ... set jax_enable_x64`. The breeding sim
   (and thus the trusted "before" state) ran in **single precision**. This can
   plausibly degrade reconstruction fidelities and the rank-1 check.
   **Test:** `JAX_ENABLE_X64=1 python ...` and re-check `plus_0` rank-1 fidelity and
   A-vs-B. If it jumps toward 1.0, much of the mystery is just precision.

2. **A vs B never confirmed on the real solutions.** Run path-1 and path-2
   (`independent_verifier.verify_circuit`, with `pnr_max ≥ max(n0)+1`) on `plus_0`
   and compare. If **A≠B**, there's a reconstruction bug specific to these
   genotypes; if **A==B** but neither is rank-1, the state genuinely isn't rank-1
   ⇒ investigate the `photons` descriptor vs. true stellar rank, and whether the
   point-homodyne (window=0) conditioning is being treated as exactly Gaussian.

3. **Correlated control modes break the single-mode core.** Even with `k_eff=1`,
   the `n_m=0` spectator modes are correlated, so the firing mode's local
   `(s₀,δ₀)` need not equal the output's. **Test:** for `plus_0`, compute the joint
   control covariance `C` (2k×2k) and check whether the firing mode's block is
   (nearly) decoupled. If correlated, the per-mode core is only approximate and the
   faithful reconstruction needs a multimode method (below).

4. **Path C conditioning (known).** `heralded_output` on the full generator is
   ill-conditioned at high energy/squeezing — documented. Not a physics error
   (moments are correct), but it means path C cannot be used to *prove* the
   equivalence by naive heralding.

---

## 7. Recommended next steps

1. **Re-run with `JAX_ENABLE_X64=1`** and the harness perf-fixes from §5. This is
   cheap and may dissolve much of the problem.
2. **Confirm A==B on `plus_0`** (the `N_c=1` case) at a modest cutoff (≤24).
   This single comparison decides between "reconstruction bug" and "state genuinely
   not rank-1." Everything downstream depends on it.
3. If A==B and faithful: the obstacle is only path C/D's Fock reconstruction of
   high-squeezing generators. The paper itself reconstructs actual states with
   **MrMustard** (their `[59]`), not closed forms. Consider using MrMustard (or a
   displaced/whitened-frame herald — a prototype reached ~0.87) to produce C and D
   for the figure. The Wigner renderer to match is `mgr/scripts/gen_wigner_pareto.py`.
4. If a faithful 3-Wigner agreement cannot be shown for these specific multimode
   11–12 dB solutions, the honest fallback is to present Hanamura via the
   (rigorous) table only and drop the standalone before/after Wigner figure — but
   exhaust steps 1–2 first, because the rank-1 failure suggests something fixable.

---

## 8. How to run things (and the git gotcha)

```bash
# regenerate the Hanamura data (cluster; needs jax + thewalrus + experiments/):
JAX_ENABLE_X64=1 python scripts/gen_hanamura_data.py --out outputs
# render the thesis figures (CPU; numpy/scipy/matplotlib), from the mgr repo:
python ../mgr/scripts/gen_wigner_pareto.py
# validation harness (FIX §5 perf bug first):
JAX_ENABLE_X64=1 python scripts/validate_pipeline.py --out outputs/validate
```

**Conventions (must stay consistent with thewalrus):** ℏ=2, xp-ordering
(x₀..x_{N-1}, p₀..p_{N-1}), vacuum covariance = Identity. thewalrus gotcha: the
`Amat` block equals **conj(B)** of the Bargmann `B`; `Q=(I−A·X)⁻¹` (order matters).

**Git note:** this repo was edited from a sandbox on a Nextcloud-fuse mount that
**won't let git delete its lock files**, leaving stale `.git/index.lock` and
`.git/HEAD.lock`. If a normal `git commit` fails with "Another git process seems to
be running," just `rm -f .git/*.lock` and retry. All commits are pushable normally
from a real checkout. `origin/master` is at `2a96c36`.

## 9. Pointers

- Paper: arXiv 2509.06255 (Hanamura et al.), full PDF in the project uploads.
  Key eqs: control params s₀,δ₀ (35–36 two-mode; 90–91 multimode); particle/core
  form (Thm 5, Eq 38); wave form (Thm 6, Eq 39); multimode canonical form (Thm 9,
  Eq 79); recursive multimode output (Thm 11, Eq 94); damping (Thm 10, Eq 81).
- Auto-memory (persists across sessions): `reference_hanamura.md`,
  `project_thesis_ch4.md`, `project_gbs_optimizer.md` — read these; they carry the
  conventions and the bug history.
- The previous bug-fix agent's writeups: `HERALDING_BUG_REPORT.md`,
  `HERALDING_BUG_REPORT_FOLLOWUP.md`.
