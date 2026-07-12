# Task brief for Claude Code: re-validate & mine ALL past experiments

You are working in the **momemura** repo (generalized breeding-protocol optimization for
GKP-state preparation, JAX + thewalrus). Your job is **data archaeology**: the repo holds
a large pile of past optimization runs, many of which were polluted by numerical artifacts
(Fock-truncation, placeholder probabilities, dropped metadata). Re-decode and **re-score
every recoverable result with the current, exact validation pipeline**, throw out the
artifacts, and surface the genuine optima that actually exist across all searches so far.

Work on a branch (`rescore-archaeology`). Do not modify the optimizer; you add a new
analysis script + tests + outputs. Commit frequently.

---

## 0. Environment

- Python venv at `.venv` (activate it). Install whatever you need: `jax` (CPU is fine; use
  GPU if the box has one), `thewalrus`, `qutip`, `qdax`, `numpy`, `scipy`, `pandas`,
  `pyarrow`, `matplotlib`. `qdax` is needed to unpickle the MOME repertoires.
- **The moment scorer is only correct in 64-bit.** Always set `JAX_ENABLE_X64=1` (env var)
  AND `jax.config.update("jax_enable_x64", True)` at the top of every entry point.
- Everything below already exists in the repo — **reuse it, don't reinvent**.

---

## 1. The data

Past runs live under three roots (each contains an `experiments/` dir):

- `experiments/`            (repo-root; the "current" runs)
- `output_old/experiments/`
- `output_oldold/experiments/`
- also check `output/experiments/` if it exists.

Each **group** folder is named `<design>_c<cutoff>_a<alpha>_b<betamag>`, e.g.
`00B_c30_a2p73_b1p41`, `A_c6_a2p00_b0p00`, `B30_c30_a1p41_b1p41`. Inside are timestamped
**run** folders `<YYYYMMDD-HHMMSS>_p<pop>_i<iters>/`, each with `config.json` + `results.pkl`.

Designs seen: `00B`, `0`, `A`, `B1`, `B2`, `B3`, `C1`, `C2`, `B30`, `B30B`. Targets vary:
α ∈ {1.00, 1.41, 2.00, 2.73}, |β| ∈ {0.00, 1.00, 1.41}, cutoffs ∈ {4,5,6,30}. **Expect high
variance** — different genotype layouts, depths, modes, cutoffs, scorer backends, and
several config-schema versions. Many old runs will NOT be decodable by the current code;
that's fine — **account for every run, recover what you can, log the rest.**

`results.pkl` contains either a `SimpleRepertoire` (single-objective; loads without qdax)
or a QDax `MapElitesRepertoire` (MOME; needs qdax). Use `pareto_report.load_repertoire(pkl)`
which already abstracts both and returns an object with `.fitnesses`, `.descriptors`,
`.genotypes` (shapes `(cells, n_obj)`, `(cells, n_desc)`, `(cells, D)` after reshape).

**Conventions (verify against code, do not trust this blindly):**
- `⟨O⟩ = -fitnesses[:, 0]` (lower ⟨O⟩ is better).
- `fitnesses[:, 1] = log10(P)` (≤ 0); displayed "LogProb"/NegLog10P `= -fitnesses[:,1]`.
- `descriptors = [active_modes, max_pnr, total_photons]`.

---

## 2. The current correct pipeline (reuse these)

- `scripts/validate_moment_archive.py` — **the reference single-run validator. Read it
  first and generalize it.** It re-scores one run's archive at a high cutoff, drops
  L-truncation artifacts, and (now) refreshes the exact probability. Your script is
  essentially "run this logic over EVERY run in EVERY root, then aggregate."
- `src/simulation/jax/moment_scorer.py`:
  - `jax_equivalent_gaussian_static(params, depth)` → `(cov, mu, eff_pnr, densities)`,
    the exact equivalent-Gaussian generator (1 signal + 2·2^depth control slots).
  - `jax_reduced_herald_static(cov, mu, eff_pnr, L, BF, depth, maxf)` → `(psi[L], prob)`,
    the exact (truncation-free) heralded signal state + PNR herald prob.
  - `_leaf_prob_product_static(params, L, depth)` → exact 'leaf' herald probability.
  - `moment_operator(L, alpha_str, beta_str)` → cached L×L GKP target operator (jax).
- `src/genotypes/genotypes.py: get_genotype_decoder(name, depth, config)` → decoder with
  `.decode(g, cutoff)` and `.get_length(depth)`. **Build it with the run's OWN config**
  (genotype name, `depth`, `modes`, `pnr_max`, and the scales `r_scale/d_scale/hx_scale`
  if present — they affect decoding).
- `frontend.gbs_optimizer.reduced_herald(cov, mu, signal_idx, control_idx, n, cutoff)` —
  an **independent thewalrus** heralding (different code path). Use it as ground-truth to
  catch scorer/decoder mismatches (fidelity to the moment state must be ≈ 1).
- `src/utils/gkp_operator.py: construct_gkp_operator(L, alpha, beta, backend)` (needs qutip).
- Per-target reference values: reuse run_mome's own imports (around line ~715–723 it does
  `from ... import gaussian_limit as get_gaussian_limit, get_u_vec_from_alpha_beta`):
  `ux,uy,uz = get_u_vec_from_alpha_beta(alpha,beta); gaussian_limit = get_gaussian_limit(ux,uy,uz)`.
  Ground-state eigenvalue `gs_eig = float(jnp.linalg.eigvalsh(moment_operator(L,a,b))[0])`.
  **These are target-specific — recompute per group, do NOT hardcode 0.30935 / 1.08932.**

---

## 3. Known gotchas you MUST handle (this is why naive loading fails)

1. **`target_beta` is missing from many configs.** `--target-beta` is parsed as a Python
   `complex`, which an old save-path silently dropped (and stored `target_alpha` as a bare
   number). ~26% of configs lack β. **Resolve the target robustly:**
   - Prefer `config['target_alpha']` / `config['target_beta']` when present, parsed with the
     robust parser in `validate_moment_archive.py` (`_as_complex`: handles numbers,
     `'(1+1j)'`, `i`→`j`, spaces).
   - Else fall back to the **group-folder name** (the authoritative record of what was
     targeted): `a2p73` → α≈2.7320508, `b1p41` → |β|≈1.41. **But |β| loses the phase.**
   - **Reverse-engineer the |β| → complex-β map empirically:** scan ALL configs that DO
     have `target_beta`, group them by their folder's `b`-string, and learn the mapping
     (you will find `b1p41 → (1+1j)`, `b0p00 → 0`, etc.). Apply it to configs missing β.
     For any `b`-string with no example anywhere, log it as `target_unresolved` and skip
     (do not guess a phase). **`b0p00` β=0 is legitimate** (pure-α GKP target), not a bug.
2. **`--moment-fast` stored a prob=1 placeholder.** Many archives have `fitnesses[:,1] = 0`
   for every cell (the probability objective is meaningless). **Always recompute the exact
   probability** via `_leaf_prob_product_static`; never trust the stored prob.
3. **Genotype-layout / schema drift.** Old genotypes may not match the current decoder's
   `get_length(depth)`. **Gate every decode on `len(g) == decoder.get_length(depth)`**; if
   it mismatches, the run used an incompatible layout → mark `undecodable` (reason
   `length_mismatch`) and move on. Try the run's own `genotype`/`depth`/`modes`; if depth
   isn't in config, infer it as the `d` whose `get_length(d)` matches `len(g)`.
4. **Two repertoire formats.** SimpleRepertoire loads bare; QDax needs qdax installed. If a
   pickle fails to load even with qdax, fall back to raw `pickle` + structural introspection
   to pull `genotypes/fitnesses/descriptors` arrays; if that also fails, log `unloadable`.
5. **Fock-truncation artifacts (the headline problem).** Old runs were scored with a
   truncated Fock breeding sim (e.g. cutoff 30/45). A high-photon heralded state simulated
   at low cutoff is distorted and can show a falsely-low ⟨O⟩ that *looks* like it beats the
   Gaussian limit. The moment scorer is truncation-free, so **re-scoring already fixes ⟨O⟩**,
   but you must additionally apply the physical-validity filters in §5.
6. **x64 or it's wrong** (see §0). complex64 underflows the high-squeezing recurrence.

---

## 4. Methodology

Write `scripts/rescore_all_experiments.py` (CLI, parametrized, GPU-aware). Pipeline:

1. **Discover** every `results.pkl` under all roots in §1. Record `(root, group, run, path)`.
2. **Per run:** load config, load repertoire (handle both formats + failures, §3.4).
   Resolve target `(α, β)` (§3.1). Determine `design, depth, modes, pnr_max, scales`.
3. **Select cells to re-score.** Re-scoring every cell of every run is millions of evals —
   too much. Per run, take: (a) the run's **non-dominated set** (Pareto front over
   `(⟨O⟩, logP)` among valid cells), plus (b) the **best-⟨O⟩ cell per descriptor bin**
   (the MAP-Elites elites). De-duplicate. Make the per-run cap a CLI flag (default e.g. 256)
   and log how many were skipped. (If a run is small, just take all valid cells.)
4. **Re-score each selected genotype exactly** (vmap/batch on GPU where possible, x64):
   - decode with the run's decoder; build `(cov, mu, eff_pnr)` via
     `jax_equivalent_gaussian_static(params, depth)`.
   - herald at `L_lo` (e.g. 50) and `L_hi` (e.g. 120) with `BF_hi` ≥ 8192, `maxf` ≥ 10:
     `psi_lo/psi_hi, _ = jax_reduced_herald_static(...)`.
   - `⟨O⟩_lo, ⟨O⟩_hi = ⟨psi|O_L|psi⟩` using `moment_operator(L, a, b)` for the **resolved**
     target (NOT β=0 unless the target really is β=0).
   - `P = _leaf_prob_product_static(params, L_hi, depth)`; `logP = log10(clip(P,1e-45,1))`.
   - `herald_norm = ⟨psi_lo|psi_lo⟩` (should be ≈ 1).
   - independent cross-check on a **subsample**: `reduced_herald(cov, mu, 0, range(1,2·2^depth+1),
     eff_pnr, cutoff=L_hi)` and `fidelity = |⟨psi_indep|psi_moment⟩|²` (must be ≈ 1).
   - descriptors recomputed: active modes, max fired PNR, total detected photons.
5. **Apply the artifact filters in §5** → mark each row valid / artifact(+reason).
6. **Aggregate:** concatenate all rescored rows; build the **global artifact-free Pareto
   front per target group** (different groups = different targets, never mix them).
7. **Write deliverables (§6).**

---

## 5. Artifact filters (the "is this a real optimum?" logic)

A re-scored solution is **VALID** only if all hold; otherwise record `is_artifact=True` with
the failing `artifact_reason`:

- `herald_norm` ∈ [0.99, 1.01] at `L_lo` — state essentially complete within the search
  cutoff (else `unnormalized`; optionally retry at higher L before failing).
- `|⟨O⟩_hi − ⟨O⟩_lo| ≤ tol` (default 0.02) — no residual L-truncation bias (`l_truncation`).
- `P` finite, `> 1e-40`, and `≤ 1` (`bad_prob`).
- in-budget: fired control modes `≤ maxf` and `∏(n_j+1) ≤ BF` (`over_budget`).
- independent-fidelity ≥ 0.999 on the checked subsample (`scorer_mismatch`).
- **Hudson / Wigner-negativity gate (critical, this is the filter motivated by the bug
  hunt):** the heralded signal state is *pure*, so by **Hudson's theorem** a pure state has a
  non-negative Wigner function **iff it is Gaussian**. Therefore **any state claiming
  `⟨O⟩_hi < gaussian_limit` (a non-Gaussian advantage) MUST have Wigner negativity.** Compute
  the Wigner negative volume of `psi_hi`; if `⟨O⟩_hi < gaussian_limit − margin` but
  `negative_volume < neg_tol` → `fake_subgaussian` artifact (impossible / numerical ghost).
  - Wigner via displaced-parity (no qutip needed, vectorize with
    `scipy.sparse.linalg.expm_multiply`): `W(z) = (2/π)·⟨ψ|D(z)Π D(-z)|ψ⟩`, `Π=diag((-1)^n)`,
    `z=(x+ip)/√2`; `negative_volume = Σ_{W<0}|W|·dxdp`. A coarse grid (~25², span ±5) suffices
    for a yes/no negativity verdict. Sanity-check it on a Fock `|1⟩` (negative at origin) and
    a coherent/squeezed state (non-negative).

Compute `vs_gs = ⟨O⟩_hi − gs_eig` and `vs_gaussian = ⟨O⟩_hi − gaussian_limit` per row.

---

## 6. Deliverables (write under `recompute/`)

1. `recompute/all_solutions.parquet` (+ a `.csv` sample) — one row per re-scored cell, with
   at least: `root, group, run, cell_idx, design, depth, modes, pnr_max, target_alpha,
   target_beta, exp_stored, exp_lo, exp_hi, herald_norm, prob, logP, active_modes, max_pnr,
   total_photons, fired_modes, fp_budget, indep_fidelity, wigner_negvol, gs_eig,
   gaussian_limit, vs_gs, vs_gaussian, scorer_backend(if known), is_artifact, artifact_reason`.
2. `recompute/pareto_fronts/<group>.csv` — the artifact-free Pareto front (⟨O⟩ vs logP) for
   each target group, with provenance (which run/cell each point came from).
3. `recompute/best_states/<group>/` — for the top genuine states per group: save the raw
   genotype (`.npy`), decoded params, `psi_hi`, ⟨O⟩, prob, and a Wigner PNG.
4. `recompute/REPORT.md` — the human-readable summary:
   - totals: #roots, #groups, #runs, #cells discovered; #runs loadable / unloadable /
     undecodable (with reason breakdown); #cells re-scored; #valid vs #artifact (by reason).
   - **per target group:** best genuine ⟨O⟩ (with `vs_gaussian`, `vs_gs`, prob, photons,
     provenance), and how it compares to the run's originally-stored best (to quantify how
     much was artifact). A table sorted by group.
   - a short narrative of what's actually recoverable vs lost, and any `target_unresolved`
     groups.
5. `recompute/undecodable.csv` — every run that couldn't be used, with the precise reason.
6. Plots: per-group Pareto fronts; a histogram of `artifact_reason`; the best Wigner(s).
7. The reusable script `scripts/rescore_all_experiments.py` with `--roots`, `--groups`
   (glob/filter), `--l-search`, `--l-high`, `--bf-high`, `--maxf`, `--tol`, `--neg-tol`,
   `--per-run-cap`, `--fidelity-subsample`, `--out recompute/`, `--limit` (for smoke runs).

---

## 7. Tests (add under `tests/`, must pass)

- **Target resolver:** group-name → (α,β) for representative names (`a2p73_b1p41 → (2.732…, 1+1j)`,
  `a2p00_b0p00 → (2.0, 0)`); and the empirical |β|→β map agrees with every config that has
  an explicit `target_beta` (cross-validation: no contradictions).
- **Round-trip consistency:** pick a recent **non-fast, non-artifact** moment-scored run;
  its recomputed `⟨O⟩_hi` matches the stored `⟨O⟩` within `tol` for the cells that pass the
  filters. (Don't assert on fast-mode prob — that's expected to differ.)
- **Independent-fidelity:** moment-scorer `psi` vs `frontend.gbs_optimizer.reduced_herald`
  `psi` on a sample → fidelity ≥ 0.999.
- **Hudson sentinels:** a constructed Gaussian pure state (squeezed/displaced vacuum) →
  `wigner_negvol ≈ 0` and `⟨O⟩ ≥ gaussian_limit`; a constructed photon-added/Fock state →
  `wigner_negvol > 0`. The `fake_subgaussian` filter fires on a hand-made
  "below-limit-but-positive-Wigner" row and does NOT fire on a genuine negative-Wigner row.
- **Robustness:** the sweep does not crash on (a) a corrupt/garbage pickle, (b) a config
  missing `target_beta`, (c) a genotype with the wrong length — each is logged and skipped.
- **Determinism:** re-running the script on the same inputs yields identical `all_solutions`
  (sort + compare).

---

## 8. Mandates

- **Never let one bad run abort the sweep.** Wrap per-run work in try/except, log the
  reason, continue. The final accounting must add up: every discovered run is in exactly one
  of {re-scored, undecodable, unloadable}.
- **Be honest about recoverability.** It's expected that only the more recent runs decode;
  report that fraction plainly rather than forcing matches.
- **Correctness over coverage.** A wrongly-resolved target or a skipped Hudson check produces
  fake optima — exactly what we're trying to purge. When unsure about a target, mark
  `target_unresolved` and exclude rather than guess.
- Keep it reasonably fast: batch the JAX re-scoring (one compiled graph per
  `(design, depth, modes)` bucket), use GPU if present, x64 always.
- Commit the script, tests, and `recompute/REPORT.md` + CSVs. (You can omit the large
  `.parquet`/PNGs from git if they're big — but always produce them on disk.)

**Definition of done:** `pytest tests/ -k rescore` green; `recompute/REPORT.md` exists and
gives, per target group, the genuine artifact-free best state and how much of the old
"record" was artifact; `recompute/undecodable.csv` accounts for everything that didn't make
it; and the global artifact-free Pareto fronts are written per group.
