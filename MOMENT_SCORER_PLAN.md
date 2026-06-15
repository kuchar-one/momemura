# Plan — moment-space optimizer scorer (`src/simulation/jax/moment_scorer.py`)

> Goal: replace the truncated Fock breeding sim **inside the optimization loop**
> with the exact moment-space path already used for rescoring/validation, so the
> optimizer scores states that are correct at any squeezing — no truncation
> artifacts to reap, no dual-cutoff sweep needed.
>
> Status as of 2026-06-13: scoped + locally validated oracle. Not yet built.

## 0. Why (one paragraph)

The in-loop scorer (`runner._score_batch_shard`) builds the whole breeding tree
in Fock space at cutoff 30 (`jax_superblock`): intermediate two-mode kron states
are truncated, corrupting high-squeezing outputs. The moment path
(`compute_equivalent_gaussian` → `reduced_herald`) composes the entire circuit on
a ≤24×24 covariance, conditions homodynes + n=0 detections analytically, and runs
ONE Hermite recurrence on the fired modes — exact at any squeezing, the Fock
object only ever being the final single-mode state. `reduced_herald` is numpy and
was deliberately kept out of the loop; this plan ports the path to JAX, batched
for the A5000s, differentiable end-to-end.

## 1. Local validation already done (the oracle)

`thewalrus`, `jax`, (stubbed) `qutip` all run in-sandbox, so the numpy reference
runs locally. Two checks (scripts in `/tmp`, fold into `tests/`):

- **Equivalence on well-behaved rows** (user's test): moment ⟨O⟩ vs the stored
  Fock ⟨O⟩ in the repertoires. `0_c5` (low cutoff): |Δ|≤0.016. `00B_c30`
  survivors: |Δ|~0.001–0.01 for genuine rows; diverges on the wrong ones
  (idx 4: 0.834→0.891; the n_det=35 descriptor-padded rows). ⟨O⟩ agreement is
  expected even at high dB because O is bounded — this is the regression anchor,
  NOT evidence the states agree (they don't; that's the point).
- **Fired-mode sizing** (sets the static tensor): coupled fired modes are mostly
  0–2, ≤3 common, 4–5 rare (one group); max photon/coupled-mode up to 15.
  Full cov ≤ 24×24.

## 2. What to port (three pieces, all JAX, fixed shapes)

### 2a. `jax_equivalent_gaussian(params) -> (cov, mu, dens)`
JAX mirror of `frontend/gaussian_decomposition.compute_equivalent_gaussian`.
- Build at **fixed** `N_MAX` modes (depth-3 ⇒ 8 leaves × (1+2) = 24; cov 48×48).
  Inactive leaves / unused control slots = vacuum, masked.
- Leaf moments: reuse `jax_get_gaussian_moments` (already in `runner.py`).
- Beam-splitter symplectics: `get_bs_symplectic` → JAX, static tree schedule
  (layers (1,4),(2,2),(3,1)), `theta/phi` from `mix_params`.
- Homodyne nodes: **do not delete modes** (dynamic shape). Apply the Schur-
  complement update (`measure_homodyne`) writing the kept block back in place and
  retiring the measured mode to decoupled-vacuum. Record the Gaussian density
  `p(x=hx)` per node (for probability). Static layout ⇒ signal/control indices
  are compile-time constants.
- Final Gaussian: `apply_final_gaussian_symplectic` → JAX.
- All ops are smooth in continuous params ⇒ exact gradients (BS/sq/disp/homodyne).

### 2b. `jax_reduced_herald(cov, mu, n, ...) -> (psi[L], prob)`
JAX mirror of `gbs_optimizer.reduced_herald` + `_gaussian_amplitudes`.

**Budget on tensor amplitudes `∏(n_j+1)·L`, NOT on mode count.** Measured driver
(local, 2026-06-13): the cost is total photon number Σn, not the number of fired
modes `k`. `k=5` solutions have median `∏(n_j+1)≈70` (trivial); the expensive tail
is Σn≈45–65 (few modes × ~15 photons), 1–8% of genotypes, where every method
(incl. loop hafnian) is intractable. A 2M-amp budget covers 92% of the hardest
group, 8M covers 97%, 100% elsewhere. **High mode count is first-class and cheap.**

- Vacuum-condition every n=0 / inactive control mode analytically (Schur
  complement, measurement cov = I; closed-form `p_vac`). Only n≥1 modes enter the
  tensor.
- Box = `(L) × (n_1+1) × ... × (n_kf+1)` using each fired mode's ACTUAL `n_j+1`
  (this exact sizing is what makes high `k` feasible — padding every axis to
  pnr_max+1 is the thing that blows up and must be avoided).
- Renormalized multidimensional-Hermite recurrence: `lax.scan` over the L signal
  axis (carry = previous slabs), inner fired-axis recurrence over the small box.
- Slice each fired axis at `n_j`, ravel → `psi`; `prob = p_vac · Σ|psi|²`.

**Static-shape strategy (no `k` cap): bucket the population by fired-shape.**
Genotypes sharing a fired pattern are `vmap`'d together; XLA compiles once per
distinct shape and caches it (~10–30 shapes/generation, compiled once, reused
across all iters). Exact per-genotype sizing, arbitrary `k`, zero padding waste.
The rare Σn≳45 over-budget tail routes to a per-genotype path (sequential GPU, or
the existing exact CPU `reduced_herald`) — still exact, just not batched.

### 2c. scoring wrapper in `_score_batch_shard`
- Replace steps 1–3 (leaf herald + `jax_superblock` + final gaussian) with 2a+2b.
- Keep the SAME fitness/descriptor/penalty assembly (QDax/MOME interface
  unchanged): ⟨O⟩ = `real(vdot(psi, O@psi))`, fitness `[-exp,-logP,-active,-n_eff]`,
  descriptor `[active, max_pnr_eff, n_eff]`.
- Effective photons (`n_eff`) become native: the coupling norm is read straight
  off `cov` (no separate leaf pass).
- Drop the truncation-only guards (leakage/structure penalties, base-vs-correction
  recompute). KEEP the physics artifact guard (sub-Gaussian ∧ no effective
  photons).
- Gate behind config `"scorer": "moment" | "fock"` (default `fock` until the
  cluster A/B passes, then flip).

## 3. GPU / A5000 utilization

- **x64 mandatory.** The moment path at 11–14 dB needs complex128 (current run is
  f32 — itself a latent error). Force `JAX_ENABLE_X64=1`; tiny matrices make f64
  cost negligible except the Hermite scan.
- Per-fired-shape compile + cache; `vmap` within a shape bucket, `pmap`/sharding
  over both A5000s as the Fock scorer does today.
- Tensor budget on amplitudes `∏(n_j+1)·L` (NOT mode count): ~2M-amp budget covers
  92–100% of genotypes batched; rare Σn≳45 tail per-genotype/CPU. No thewalrus, no
  loop hafnians.
- Expected: cheaper than the Fock tree (which builds c×c kron states at 7 nodes)
  AND correct. Benchmark script reports throughput + peak VRAM vs Fock.

## 4. Tests (`tests/test_moment_scorer.py`)

1. `jax_equivalent_gaussian` cov/mu == `compute_equivalent_gaussian` (light), per
   leaf config (n_ctrl 0/1/2), random genotypes — atol 1e-9 (x64).
2. `jax_reduced_herald` psi & prob == numpy `reduced_herald` across squeezings
   (0–16 dB), fired counts 0–3, pnr 0–15 — atol 1e-8.
3. Full JAX score == numpy `rescore_genotype` (exp, P) on canonical rows
   (plus_*, H_*, T_*) — we have ground truth.
4. **Equivalence-to-Fock on well-behaved genotypes** (user's test, promoted to a
   regression): moment ⟨O⟩ ≈ stored/recomputed Fock ⟨O⟩ on low-squeezing rows
   (|Δ|<0.02); assert divergence flagged on the known-bad rows.
5. Gradient: finite-difference vs autodiff of ⟨O⟩ w.r.t. squeeze/BS/disp/homodyne.
6. Over-budget sentinel + `effective_photons` guard behavior.
7. Regression: existing `tests/` still pass (`reduced_herald`, archive_validator).
8. GPU smoke + benchmark (cluster): batch vmap/pmap, throughput & VRAM vs Fock.

## 5. Sequencing

1. `moment_scorer.py` + 2a, unit test (1).  2. 2b + tests (2).  3. wrapper + tests
(3,4,6).  4. gradient + regression (5,7).  5. local CPU vmap smoke.  6. cluster
A/B + benchmark (8): same `run_pipeline` invocation with `--scorer moment`,
compare fronts/throughput/VRAM to Fock, confirm the periodic sweep removes ~0.
7. flip default; retire the dual-cutoff sweep (becomes redundant).

## 6. Decisions (resolved 2026-06-13)
- **No mode-count cap.** Budget on tensor amplitudes `∏(n_j+1)·L`; shape-bucketed
  vmap sizes each tensor exactly so high-`k` solutions (the interesting ones) are
  first-class. Only the Σn≳45 tail is special-cased (exact CPU/sequential).
- Probability fitness[1] = **`leaf`** (truncation-free, matches current optimizer
  ⇒ existing repertoires stay valid seeds). Record `physical` in extras/CSV too.
- First PR = **scorer behind `--scorer moment|fock`, default `fock`**; keep the
  dual-cutoff sweep as a safety net; flip default + retire sweep after cluster A/B.

## 6b. Status (2026-06-14)
- **2a DONE + validated**: `jax_equivalent_gaussian` == numpy ref to ~1e-14
  (cov/mu/densities/indices, all groups incl. high-`k` `a1p00_b1p41`); autodiff
  == finite-diff to 1e-9. Self-contained tests pass on Mac + cluster.
- **2b DONE + validated**: `jax_reduced_herald` == numpy `reduced_herald`
  BIT-exact (psi + prob, ~1e-16–1e-11) across fired counts kf=0–3; general-rank
  so kf≥4 use the identical path.
- **3a DONE + validated**: `moment_score_one` (per-genotype kernel) — `<O>` and
  `P_leaf` match the numpy rescore to ~1e-12; structure is static (jit/vmap/
  bucket-ready). Forward value test in the suite (fast); gradient test gated
  behind `MOMENT_SLOW=1`.
- **AD blocker RESOLVED (2026-06-14)**: the base slab was an O(∏(n_j+1))
  `lax.fori_loop` — forward fine, but reverse-mode AD compiled in tens of s and
  blew up for larger ∏. Replaced with **anti-diagonal `lax.scan`**: fill the
  fired box in T=Σn layers of constant total photons, each layer's <=W entries
  filled in parallel (`_base_slab_schedule`, cached per static `sub`). Now
  bit-exact (dpsi ~1e-16–1e-13) and `jax.grad` compiles in ~1–8 s, runs <1 s,
  even at kf=3 / ∏≈3500 / Σn≈43 (which previously hung). Suite: 13 passed incl.
  the gradient test (`MOMENT_SLOW=1`). `REDUCED_HERALD_PROD_BUDGET` (16384) still
  routes the extreme-Σn tail to the exact CPU fallback.
- **3b + 3c DONE + validated (2026-06-14)**: `moment_score_population` —
  structure-bucketed `vmap(value_and_grad(moment_score_one))` (one compile per
  fired/active signature, cached), exact CPU `reduced_herald` fallback (zero grad)
  for the over-budget Σn tail, scatter back to population order. Same
  (fitnesses[N,4], descriptors[N,3], extras{gradients,...}) contract as
  `_score_batch_shard`. Native effective-photon descriptor from the final cov
  (closes the dud exploit); physics artifact guard kept; truncation-only
  penalties dropped. Wired into `jax_scoring_fn_batch` behind config
  `scorer="moment"`; CLI `--scorer moment|fock` + `--moment-cutoff` in run_mome;
  L-cutoff `<O>` operator cached (`moment_operator`).
  Local validation: moment vs Fock <O> agree to ~1e-4 on well-behaved genotypes,
  log-prob agree, gradients finite; wired `jax_scoring_fn_batch(scorer=moment)`
  returns correct shapes & finite grads.
- **Acceptance gate**: `scripts/consistency_moment_vs_fock.py` (cluster) — moment
  vs Fock through the real wired path, PASS if median |Δ<O>| on well-behaved
  (<8 dB) solutions < tol and all moment grads finite. After it passes, full runs
  with `--scorer moment` are valid (do a short smoke run first; then task 5
  benchmark / retire the dual-cutoff sweep).

## 6c. Option 2 -- single-compile static scorer (2026-06-14)

The structure-bucketed scorer recompiled per genotype (~60 distinct structures /
64-genotype population) -> XLA compile storm, fatal for QDax. Replaced with a
ONE-compile, fully-masked, fixed-shape scorer:

- **Piece A** `jax_equivalent_gaussian_static`: fixed 8x3=24-mode layout; inactive
  leaves / variable n_ctrl via masked Clements gates (build N=1/2/3 Clements and
  select by n_real); tree routing via masked BS angles (identity when one child
  dead, theta=pi/2 swap when only the right child is live). Fixed loop counts ->
  unrolls into one graph. **Bit-exact vs per-structure (dcov 0, dmu 1e-14)** incl.
  inactive-leaf routing.
- **Piece B** `jax_reduced_herald_static` / `_flat_amplitudes`: vacuum-condition
  eff==0 controls (masked sequential Schur), then Hermite box over (signal + <=MAXF
  fired) in a FIXED flat buffer `MOMENT_BF`, with mixed-radix strides / predecessor
  indices computed at RUNTIME from the pnr-derived radii. Base slab = level-scan
  (<=T_MAX steps); signal axis = lax.scan over L. **Bit-exact vs numpy
  reduced_herald (psi 1e-12) for kf=0..7**, incl. high-kf synthetic states.
- `_leaf_prob_product_static` reuses `_herald_static` at MAXF=2/BF=256 per leaf.
  Full forward chain (exp + P_leaf) == numpy rescore to 1e-16 / 1e-12.
- `moment_score_population_static`: single `vmap(value_and_grad)` over the whole
  population (ONE compile, no bucketing); over-budget tail (kf>MAXF or
  prod(n_j+1)>BF) -> exact CPU fallback (zero grad). Wired into
  `jax_scoring_fn_batch` behind `scorer="moment"`.
- Knobs: `MOMENT_MAXF=8`, `MOMENT_BF=4096` (raise to cover more of the ∏ tail if
  VRAM allows). Tests in `tests/test_moment_scorer.py` (static equiv + static
  reduced_herald, self-contained). The jit+vmap+grad COMPILE is only verifiable on
  the cluster (CPU sandbox window too small) -- that's the smoke run.

## 6d. Performance knobs + exact re-validation (2026-06-15)

The single-compile scorer is correct but f64 + the level/signal scans are heavy
(~18 GB, ~2.9 s/iter at L=100, BF=4096). Knobs to trade speed vs coverage/accuracy:
- `--moment-bf` (default 4096): fired-box flat buffer. Memory ∝ BF. 1024 ⇒ ~4×
  less VRAM (kills the 18 GB rematerialization), drops more high-Σn genotypes.
- `--moment-cutoff` (L, default 100): final signal Fock cutoff. 50 halves the
  signal scan; mildly truncates high-squeezing signal states (caught by the gate
  + re-validation).
- `--moment-fast`: skip the exact per-leaf probability (8 sub-heralds) — big
  speedup. Auto-on when `alpha_probability==0`. Sets prob objective flat, so use
  only for exp-focused search (recover true prob via re-validation), NOT for a
  Pareto/probability experiment.

Workflow (fast search + exact validation):
1. Search fast: `... --scorer moment --moment-cutoff 50 --moment-bf 1024 [--moment-fast]`.
2. Re-validate the archive exactly:
   `JAX_ENABLE_X64=1 python scripts/validate_moment_archive.py --group <g> \
       --l-search 50 --l-high 120 --write`
   Re-scores every point at high L/BF, flags low-L artifacts (|Δ<O>|>tol or
   low-L herald norm<0.99), prints the best EXACT <O>, and (`--write`) emits an
   exact-rescored repertoire. This is the moment dual-cutoff sweep, both ends
   exact.

VRAM cycling during QDax = `XLA_PYTHON_CLIENT_PREALLOCATE=false` + chunked
MAP-Elites (scan→Python→scan), not reinitialisation. Each watchdog restart
recompiles (~5 min) — inherent to the watchdog.

## 7. Risks
- Moment path is exact only for **point** homodyne (window=0). 00B uses window=0;
  guard against finite-window configs (would need mixed-state hybrid — out of
  scope).
- `MAX_FIRED` clamp approximates rare high-fired solutions in-loop; final archive
  is exact via CPU rescore, so the figure/table stay exact.
