# Hanamura pipeline validation — root cause found (2026-06-10)

**TL;DR: there is no reconstruction bug.** The "rank-1 failure" of `plus_0` and
the path disagreements were caused by **Fock-cutoff truncation of the breeding
sim** (path 1) — the canonical-trio solutions carry 11–12 dB of internal
squeezing and are *nowhere near converged at the config cutoff of 30*
(tail amplitudes ~0.1). On top of that, `plus_0`'s single detected photon is
**physically inert** (its control mode is decoupled from everything), so the
true `plus_0` output is *exactly Gaussian* — it could never match a rank-1 core.

## Chain of evidence (all reproducible in minutes, scripts below)

1. **Leaf-level A==B is exact.** For every active leaf of `plus_0`, the JAX
   leaf herald (`jax_get_heralded_state`) and the independent thewalrus
   reconstruction (`_herald_leaf`) agree to F=1.000000 with identical
   probabilities. Paths 1 and 2 share no code, so the circuit semantics and
   conventions (post-BS-fix) are confirmed correct.

2. **`plus_0`'s photon is a dud.** In the (bit-for-bit correct) equivalent-GBS
   covariance, the fired control mode is locally PURE (ν=1.000000) and its
   cross-covariance to the signal and to all other modes is ~4e-4. Its leaf BS
   angle decodes to θ≈π (cos≈−1, sin≈3e-4): the leaf interferometer never
   couples signal and control. Heralding n=1 on a decoupled pure mode only
   rescales probability. Verified independently: the leaf state with PNR=1 vs
   PNR=0 is the *same state* (overlap 1.0000).

3. **The exact `plus_0` output is a pure Gaussian** (10.8 dB squeezed,
   nbar=2.696, det of its Wigner covariance = 1.0000000): obtained analytically
   by vacuum-conditioning the n=0 modes of the equivalent-GBS moments.

4. **Path 1 converges to that exact Gaussian as the cutoff grows** (x64):

   | cutoff L | 24 | 40 | 56 |
   |---|---|---|---|
   | F(A_L, exact) | 0.576 | 0.732 | 0.877 |
   | nbar(A_L) | 6.36 | 4.49 | 3.84 (→2.70) |
   | tail amp | 1.7e-1 | 5.5e-2 | 5.5e-3 |

   Same trend for `T_1`: F = 0.24@40 → 0.74@56. The sandbox CPU cannot go
   past L≈56; **run `scripts/validate_convergence.py` on the cluster** to
   carry this to L≈160 and confirm F→1 for all canonical rows.

5. **f32 vs f64 was a red herring**: x64 path-1 at L=30 reproduces the stored
   cluster (f32) state (same nbar 4.861). The truncation, not precision,
   dominates.

## Why everything downstream failed

- The optimizer **scored truncated states**. The cutoff-30 "ground truth" used
  for `coreFid`, the rank scans, and the cached `psi_before` is not the
  physical output of the circuit for high-squeezing solutions.
- The old `heralded_output` (path C) was both intractable (one loop hafnian per
  amplitude on the full 6-mode generator) and ill-conditioned. It has been
  replaced (see below).
- `align_states` against core states compared a truncation artifact with exact
  states — the flat ~0.69 scans follow.

## The descriptor exploit (thesis-relevant!)

Scanning all 15 chosen rows for fired-mode coupling
(`scripts/validate_convergence.py` prints this):

- `plus_0`: fired mode fully decoupled (|C|₂=0.0004) → photon does nothing.
- `plus_1` (n=5 mode, 0.016), `H_2` (n=5, 0.0003), `T_3` (n=5, 0.046): the
  Nc=35 rows pad their photon count with (near-)decoupled detections.
- The MOME "photons" descriptor counts *detected* photons, not *used* photons —
  adding decoupled PNR modes moves a solution to a different MAP-Elites cell
  without changing the physical state. The optimizer exploited this to fill
  niches. `Nc` ≠ stellar rank ≠ non-Gaussian resources.
- The genuinely-coupled low-Nc rows: `plus_2`, `plus_3`, `H_0`, `H_1`, `T_0`
  (weak, 0.21), `T_1`, `T_2` — these are the candidates for the Wigner figure.

## What changed in the code

- **`frontend/gbs_optimizer.reduced_herald`** (new): exact heralded output via
  analytic vacuum conditioning (Schur complement, measurement cov = I) of all
  n=0 control modes + thewalrus Hermite-recurrence `state_vector` (no
  post-select, no hafnians) on the small remaining (signal+fired) system, then
  slicing the fired Fock indices. Exact match to `heralded_output` (state and
  probability) on tractable generators — `tests/test_reduced_herald.py`
  (13 tests, passing). Fast and stable at any squeezing/cutoff.
- **`scripts/validate_pipeline.py`**: forces x64; path C and the D
  herald-fallback now use `reduced_herald` at full cutoff; the
  `cutoff2=60` probe that hung the harness for days is disabled by default.
- **`scripts/gen_hanamura_data.py`**: `reduced_full_state` now heralds via
  `reduced_herald`.
- **`scripts/validate_convergence.py`** (new): the cluster job that closes the
  loop — ramps path-1 cutoff to 160 per canonical row and reports F against
  the exact `reduced_herald` reference + the fired-coupling audit.
- **`scripts/check_path_consistency.py`** (new): single-solution A/B/core/
  coupling diagnostic.

## Consequences for Chapter 4

1. **`tab:hanamura` stays valid** (pure moment space) — but annotate `plus_0`:
   its detected photon is physically inert (output Gaussian); arguably drop the
   row or use it as a cautionary example.
2. **The before/after Wigner figure is now buildable**: use `reduced_herald`
   for the "before" (exact, replaces the truncated path-1 state) and the core
   state / `reduced_full_state` for "after". For rows where the cluster
   convergence run confirms F(A_L→∞, reduced_herald)→1, the three-Wigner
   acceptance criterion (optimizer = GBS = Hanamura) is met by construction —
   show A at the largest converged cutoff alongside if desired.
3. **Bigger caveat to flag in the thesis**: optimizer scores (⟨O⟩, P) of
   high-squeezing solutions include truncation error and the photons
   descriptor can be gamed by decoupled detections. Recommend re-scoring the
   Pareto front's ⟨O⟩ from `reduced_herald` states (cheap) before quoting
   numbers, and/or adding a fired-coupling penalty/filter to the optimizer.

## Cluster command

```bash
JAX_ENABLE_X64=1 python scripts/validate_convergence.py --out outputs/convergence
# optional A==B spot check (slow):  --check-b
```

---

# Addendum (2026-06-10, same day): rescoring + exploit fix

**Cluster confirmation came in:** F(A_L, exact) for plus_0 = 0.732@40, 0.877@56,
0.973@72, **0.994@96** — monotone convergence to the exact moment-space herald.
Verdict closed. (L=128 OOMs the tree sim; irrelevant.)

## New finding: the scored probability never included homodyne acceptance

`jax_superblock` DISCARDS each mixing node's `p_measure` (the `_` in its
unpack): the optimizer's `joint_probability` is the product of LEAF herald
probabilities only. Verified numerically: prod of exact leaf probs = 0.166 vs
sim joint 0.170 for plus_0 (residual = leaf truncation). The physical
acceptance — density(x_i)·window per homodyne node — is ~1e-8 for plus_0, i.e.
the experimentally meaningful success probability is ~8 orders of magnitude
below the quoted one. Flag this in the thesis when quoting P. The rescorer
reports both conventions (`P_leaf` exact-old-convention, `P_phys` physical).

## New tooling

- **`scripts/rescore_repertoires.py`** — exact, cutoff-free re-scoring of whole
  repertoires in moment space: `exp` on the reduced_herald state (final-state
  cutoff 100, cheap), exact `P_leaf` + `P_phys`, effective photons `n_eff`
  (fired-mode coupling audit, `--coupling-eps`). Writes a seedable mirror tree
  (`results.pkl` + `config.json` per run, SimpleRepertoire format readable by
  `scan_results_for_seeds` and `pareto_report`) + `rescore_summary.csv`.
  Sandbox spot-check found e.g. a point with scored exp 0.566 whose true exp is
  0.977 (worse than the 0.667 Gaussian bound) — pure truncation artifact — and
  points whose *every* detected photon is a dud.
- **`compute_equivalent_gaussian(..., light=True)`** — skips Bloch-Messiah for
  bulk rescoring; also now returns `homodyne_densities` (exact per-node
  Gaussian density p(x=hx), same hbar=2 x-units as the Fock sim's phi_n).
- **Dud-photon guard in the scorer** (`src/simulation/jax/runner.py`):
  `_leaf_effective_pnr` weights each leaf detection by a smooth gate
  w = c²/(c²+eps²) on the control mode's covariance coupling to the rest of its
  leaf (built at the leaf's TRUE mode count — the Clements layout depends on
  it). Effective totals replace `leaf_total_pnrs`/`leaf_max_pnrs` before the
  superblock, so the photons descriptor, the photon fitness AND the
  sub-Gaussian artifact guard all see only contributing photons. Differentiable;
  default ON, opt out with config `"effective_photons": false`, threshold via
  `"coupling_eps"` (default 0.05). Verified: plus_0 raw Nc=1 → eff 0.001 (now
  trips the artifact guard); plus_2 4→4.0; T_2 14→14.0; H_2 35→30.0 (the n=5
  dud removed). Tests: `tests/test_effective_photons.py`.

## Updated runbook (cluster)

```bash
# 1. exact re-scoring of everything (honest fronts + seeds):
JAX_ENABLE_X64=1 python scripts/rescore_repertoires.py \
    --root experiments --out experiments_rescored
# rescored trees are drop-in for seeding (scan_results_for_seeds-compatible);
# swap/point the seed scan at experiments_rescored as desired.

# 2. re-run optimization (dud-photon guard is now default-on):
#    use your usual run_mome invocation with --seed-scan against the rescored
#    tree; add '"effective_photons": false' to a config to reproduce old runs.
```

## Post-launch fixes (first cluster attempt OOM'd at >100 GB RAM)

Root cause: thewalrus `state_vector` only takes a SCALAR cutoff, so a solution
with k fired control modes allocated `cutoff^(k+1)` amplitudes (100^4 = 1.6 GB,
100^5 = 160 GB). Fixed by implementing the renormalized multidimensional-
Hermite recurrence with PER-MODE cutoffs inside `reduced_herald`
(`_gaussian_amplitudes`): tensor is now `cutoff x prod(n_j+1)` — the worst
real case (signal + fired (15,15,5) @ L=100) is 153k amplitudes / 27 ms.
Verified bit-exact against `state_vector` (max diff 3e-17) and all 13
exactness tests. A memory budget (2e8 amplitudes) shrinks the signal axis
rather than OOM-ing on pathological PNR patterns.

Also: the rescorer now sets `JAX_PLATFORMS=cpu` (JAX is only used for decoding;
a GPU backend pointlessly preallocates ~75% of every card) and runs run-dirs in
parallel (`--workers`, default half the cores).
