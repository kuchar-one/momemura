# HANDOFF: Validate NG-pipeline results + run full Pareto-front Hanamura optimizations

You are picking up after a major optimizer refactor (2026-07-02) and its first
weekend production campaign (finished ~2026-07-06). The campaign produced
**deeply sub-Gaussian results for BOTH targets** — the stabilizer/plus state
(α=1, β=1) and the magic state (α=2.7320508, β=1+1j). Your job, in order:

1. **Audit the runs**: read the pipeline logs and archives, flag anything weird.
2. **Validate the results**: prove the sub-Gaussian winners are physical, not
   numerical artifacts (this codebase has a history of truncation exploits).
3. **Run the full Pareto-front Hanamura control-parameter optimizations** on the
   validated results, for both targets.

Work locally for code/tests; production analysis runs on the cluster:
`/cluster/home/kuchar/code/momemura`, `.venv` (python 3.11), 2× RTX A5000
(24 GB). JAX x64 is REQUIRED for the moment scorer (`run_mome.py` self-enforces
when "moment" is in argv). Conventions everywhere: **ħ=2, xp-ordering
(x_0..x_{N-1}, p_0..p_{N-1}), vacuum covariance = Identity**; thewalrus gotcha:
`Amat = conj(B)`.

---

## 1. What the optimizer is

`run_mome.py` searches over "breeding protocol" circuits: a binary tree of
depth d with 2^d **leaves**, each leaf a 3-mode general Gaussian state (1
signal + up to 2 PNR-heralded controls), combined pairwise by beamsplitter
nodes each followed by an x-homodyne condition, plus a final Gaussian. The
objective is the expectation ⟨O⟩ of a GKP/witness operator built from the
target (α, β) — **minimize**; sub-Gaussian means beating the analytic Gaussian
limit (e.g. 2/3 for α=β=1; printed as "Gaussian Limit" at startup).

Scoring is the exact **moment-space scorer** (`--scorer moment`,
`src/simulation/jax/moment_scorer.py`): composes the whole circuit as a
covariance/mean pair, heralds analytically, and materialises only the final
single-mode Fock state at cutoff L (`--moment-cutoff`, campaign used L=50
in-loop, validated at L_high=120). Search is QDax MOME (MAP-Elites with
per-cell Pareto fronts, max length 5).

### Genotypes (B30 family, `src/genotypes/genotypes.py`)
- **B30**: semi-tied design — shared continuous leaf block, per-leaf discrete
  genes (active flag, n_ctrl, PNR pattern), per-node homodyne + mixing, final
  Gaussian. Layout: `hom(nodes) | shared(18) | unique(L×4) | mix(nodes×3) | final(5)`.
- **B30F** (new): same length/continuous mapping as B30 but **forced
  heralding** — leaf 0 always active, every active leaf has n_ctrl ≥ 1 and its
  first detector fires ≥ 1 photon. The Gaussian manifold is *unrepresentable*:
  discovery runs live entirely on the non-Gaussian side of the barrier.
  `converter.convert_b30f_to_b30()` re-encodes losslessly to B30 (unit-tested
  to 1e-6 in state space).
- `converter.upgrade_depth()` embeds a depth-d genotype into depth d+1/d+2
  exactly (extra leaves inactive, θ=0 pass-through nodes) — this is how depth
  growth transfers progress. Also unit-tested state-exact.

### 2026-07-02 refactor — what changed and why (read before judging logs)
- **Fitness is now 2 objectives**: `fit = [-⟨O⟩, log10(P)]`. The old objectives
  3/4 (-active, -photons) made every proto-non-Gaussian candidate
  Pareto-dominated by the Gaussian corner. **Legacy archives are 4-wide** —
  any tool reading `fitnesses[:, 2:]` must be shape-tolerant.
- **Descriptors: 4 axes** = (active leaves, max effective PNR, effective
  photons, **δ_ng** = relative-entropy non-Gaussianity of the heralded signal
  state). D4 is **log-binned** (centroids 0, 0.02 … 2.0, 9 bins).
- **Sweep-trigger bugfix**: the periodic dual-L validation sweep
  (`clean_archive_moment`) previously *never* fired mid-run (`completed % 250`
  is unreachable with chunk 64); it now fires on every crossing of
  `--moment-validate-every`, runs **incrementally** (only cells whose fitness
  changed since the last sweep; the function returns a 3-tuple
  `(repertoire, n_removed, fingerprint)`), and always runs once more before
  saving. Consequence: **saved archives are already L=120-exact** — but verify
  anyway (§3).
- **In-loop tail gate**: states with > `--moment-tail-tol` (0.05) of their
  normalised mass in the top decile of the L-box are invalid at insertion (the
  old `norm_tol` check was a silent no-op).
- **ng-hybrid emitter** (`src/optimization/emitters.py`): exploration stream +
  fitness-elite (softmax on -⟨O⟩, T=0.05) + **NG-elite** (softmax on δ_ng,
  T=0.2). Env knobs `NG_ELITE_RATIO`, `NG_ELITE_TEMP`.
- **Physics macro-mutations** (`src/optimization/macro_mutations.py`, 18 ops:
  photon-subtract leaf, breed-level/all, symmetrize sibling/all, mirror
  half-tree, click ±1, boost squeeze, hom-zero, …), applied with
  `--macro-prob` inside exploration.
- **NG-stratified seeding** (`--seed-metric pareto_ng`,
  `src/utils/result_scanner.py`): candidates bucketed by δ_ng, per-stratum
  (exp, prob) Pareto fronts, round-robin selection, high-NG strata first;
  cross-design acceptance via `--seed-accept` (B30F↔B30 conversion + depth
  embedding happen automatically in `run_mome.py` seeding).
- **PNR-pattern seeds** (`src/genotypes/pnr_seeds.py`): canonical click
  combinations under the fired-box budget (kf ≤ moment_maxf,
  ∏(n_j+1) ≤ moment_bf) with breeding-friendly continuous defaults.
- **"vs G_N"** (`src/utils/gaussian_reference.py`): clamped-Gaussian optimum
  under the search's squeezing clamp, printed at startup (G_N and G_N2 at
  2×r_scale) and in every progress line. G_N = 0.0000 means "merely matched
  the best clamped Gaussian"; **negative = genuine non-Gaussian advantage**.
- **STE decode twins** (`pnr_ste`, `n_ctrl_ste`): forward-exact float copies of
  the discrete genes. Known limitation: PNR feeds integer indexing downstream,
  so these carry ~no gradient today — Adam (single mode) polishes continuous
  parameters with the discrete structure frozen. That is intentional.
- Unit tests: `tests/test_ng_pipeline_units.py` (10 tests; run piecewise on
  CPU, compiles are slow).

### The pipeline that produced the results (`run_pipeline_ng.py`)
Per cycle (2 cycles), per depth (3 → 4 → 5), launched via
`watchdog_restart.py` with `STAGNATION_LIMIT=100000 WATCHDOG_START_LONG=1`:

- **Phase A — discovery**: qdax, **B30F**, ng-hybrid, macro-prob 0.35,
  pnr-seeds 32, α_ng (NG exploration reward, optimization-objective-only)
  = 0.3, seed-accept B30.
- **Phase B — consolidation**: qdax, **B30**, ng-hybrid, macro-prob 0.2,
  α_ng = 0.1, seed-accept B30F.
- **Phase C — polish**: 2 parallel short Adam runs (`--mode single`, 300
  iters, lr 0.02), `--seed-fill jitter` (whole population = jittered copies of
  the NG-stratified Pareto set), α_ng = 0.
- Common: `--seed-scan --global-seed-scan --moment-ng-descriptor
  --seed-metric pareto_ng`, pop 64, cutoff 30, pnr-max 10, modes 3,
  moment-cutoff 50, moment-bf 1024, moment-maxf 10, moment-chunk 64.

Logs: master `ngpipe_master_a1b1.log` (and the magic-state analogue), per-phase
`ngpipe_c{cycle}_d{depth}_{A|B|C}*_<ts>.log`. Archives:
`output/experiments/B30F_c30_a1p00_b1p00/`, `output/experiments/B30_c30_a1p00_b1p00/`,
`output/experiments/B30F_c30_a2p73_b1p41/`, `output/experiments/B30_c30_a2p73_b1p41/`
(run dirs contain `results.pkl`, `config.json`, `checkpoint_latest.pkl`,
`final_plot.png`, `history.gif`).

---

## 2. Task 1 — log & archive audit ("anything weird?")

Go through the per-phase logs chronologically. Specifically check:

- **Sweep lines** `=== Moment Validation Sweep (gen N, L 50->120, incremental) ===`
  should appear every ~256 gens. Plot/track `Removed k L-truncation artifacts`
  per sweep: after the first (full) sweep, counts should be small
  (≲ few hundred). A late spike = an exploit family the tail gate misses —
  investigate which cells (descriptor region) were removed.
- **Progress lines**: `Exp: … (vs GS: …, vs G: …, vs G_N: …)`. Confirm
  monotone non-increase within a run and that reported bests survive
  subsequent sweeps (a best that jumps back UP right after a sweep line was an
  artifact that led the elite stream astray — note where).
- **Startup blocks**: `G_N`/`G_N2` values sane (analytic G ≤ … ≤ G_N2 ≤ G_N);
  `Physics macro-mutations: ON (prob=…, 18 operators)`; `Injected N seeds
  (metric=pareto_ng, …)` with nonzero depth-embedded counts at depths 4/5;
  `Injected N PNR-pattern seeds`; the ng-hybrid banner WITHOUT the
  "[WARN] ng-hybrid without --moment-ng-descriptor" line.
- **Coverage** trends: sudden coverage collapses right after sweeps = tail
  gate or sweep too aggressive at that depth.
- **Phase C (Adam)**: `Best Exp` should improve at least in the 3rd/4th
  decimal from its seeds. Flat-to-machine-precision from iteration 1 in ALL
  polish runs would suggest the jitter-fill seeding failed (check the
  `Injected … (metric=pareto_ng, fill=jitter)` line).
- **Prob axis**: phase A/B ran WITHOUT `--moment-fast`, so log10 P values
  should be real (≤ 0, not all exactly 0). In `--mode single` output,
  `Best Prob: 1.000000` is a placeholder — ignore it there.
- **Watchdog**: note any `IMPROVEMENT DETECTED` restarts, crashes, OOM, or
  NaN warnings; also XLA recompile storms (compile lines mid-run = cache
  misses worth reporting).
- **Archive-level stats** (write a small script or extend `pareto_report.py`):
  per target, per genotype (B30F vs B30), per depth: number of valid cells,
  number of sub-Gaussian cells (⟨O⟩ < analytic G AND < G_N), δ_ng distribution,
  photon/PNR distributions of the winners, and the global best with its
  descriptors. Sanity: winners should have δ_ng > 0 and fired detectors;
  a "sub-Gaussian" cell with 0 effective photons is by definition an artifact
  (the scorer marks those invalid — finding one saved means a bug).

## 3. Task 2 — validity of the sub-Gaussian winners

History lesson (why we're paranoid): pre-refactor runs produced thousands of
fake sub-Gaussian states via Fock/L-truncation exploits; a 2026-06-10
investigation also found a **decoupled-photon descriptor exploit** (detectors
firing on modes decoupled from the signal inflate photon descriptors while the
output stays exactly Gaussian). All Fock reconstructions must go through
`reduced_herald` (`frontend/gbs_optimizer.py`) — never the truncated Fock
breeding tree.

For the top candidates (say, the global best + the full sub-Gaussian set of
each target):

1. **Independent high-L rescore**: recompute ⟨O⟩ at L = 200–240 with
   `moment_bf_high ≥ 8192` (use `clean_archive_moment` with L_lo=120,
   L_hi=200+, or `_revalidate_jit` directly) and confirm |Δ⟨O⟩| < 0.02 vs the
   stored value. The stored archives are L=120-refreshed; this is the
   independent check at yet-higher L.
2. **Numpy cross-check**: for the top ~10 per target, reconstruct the state
   via the numpy reference path (`frontend.gaussian_decomposition.
   compute_equivalent_gaussian` → `frontend.gbs_optimizer.reduced_herald`) and
   compare ⟨O⟩ and the state vector against the JAX moment scorer
   (should agree to ~1e-8; `tests/test_moment_scorer.py` shows the pattern).
3. **Coupling audit**: confirm effective photons (descriptor axis 2) ≳ 1 and
   that fired detectors are genuinely coupled to the signal (the
   `_effective_photons_static` gate uses coupling_eps=0.05 — recompute
   offline for the winners; a winner whose n_eff ≈ 0 is the decoupled exploit).
4. **Physicality**: herald probability finite and sane (log10 P from the
   archives; anything < -40 is suspicious), state norm complete at L=120
   (top-decile tail mass ≪ 0.05), Wigner negativity present (δ_ng > 0 and a
   quick Wigner plot for the champions — `scripts/run_hanamura_pareto.py`
   already renders Wigner functions).
5. **Structure sanity**: decode the winners
   (`get_genotype_decoder(...).decode`) and eyeball: active leaves, PNR
   patterns, mixing angles, homodyne x's. Deeply sub-Gaussian winners should
   look breeding-like (several fired leaves, near-balanced nodes, small |x|
   homodyne). Document the discovered structure — this is thesis material:
   the whole point of the campaign was learning the structure of the
   magic-state protocol, which was previously unknown.

Deliverable: a validation report (markdown in repo root, follow the style of
`HANDOFF_rescore_archaeology.md` / `HANAMURA_VALIDATION_FINDINGS.md`) with a
per-candidate table: stored ⟨O⟩, L=200 ⟨O⟩, numpy ⟨O⟩, P, δ_ng, n_eff,
verdict.

## 4. Task 3 — full Pareto-front Hanamura optimizations

Once validity is established, run the Hanamura control-parameter
optimization over the **full validated (exp, prob) Pareto front** of each
target (not just the champion):

- Tool: `scripts/run_hanamura_pareto.py` (built 2026-06-30; produces full
  before/after archive comparisons + Wigner plots). Check its `--help`; it
  loads QDax archives robustly, dedupes rows, and skips modes≠3/invalid
  entries.
- **Magic-state gotcha**: target-matching may drop the a2p73 group because of
  how (α, β) are stored in configs — if the magic-state front comes back
  empty, use the `--force-alpha 2.7320508 --force-beta 1+1j` override (this
  exact failure happened before with `pareto_report.py`).
- Legacy-format note: these runs' archives are 2-objective + 4-descriptor —
  if the script assumes 4-objective fitnesses anywhere, patch it
  shape-tolerantly (`fitnesses.shape[-1]`), don't reindex blindly.
- pnr_max=10 was chosen deliberately so the Hanamura stage has photon-number
  headroom: part of the goal is **photon reduction** — find Hanamura
  control-parameter re-optimizations of the winners that keep ⟨O⟩
  sub-Gaussian with fewer detected photons / higher probability. The frontend
  GBS-optimizer feature (two-step photon-reduction + probability-maximization)
  exists for exactly this; consider chaining it after the Pareto runs.
- Run per target, per depth stratum if instructive; both GPUs are available
  (pin with `CUDA_VISIBLE_DEVICES` if you parallelise the two targets).

Deliverable: before/after Pareto fronts (⟨O⟩ vs log10 P) for both targets,
Wigner plots of the champions, and a short summary of how much the Hanamura
stage improved probability/photon count at fixed sub-Gaussianity.

## 5. Ground rules

- Never "fix" a validation failure by loosening the check; find the cause.
- Anything you script, keep and commit (`scripts/` or repo root), with a
  short docstring; follow the existing HANDOFF/report style.
- x64 always; watch VRAM at depth 5 (validation at high L/BF is the peak —
  shrink the validation chunk first, then BF).
- The user (Vojta) values honest negative findings as much as positive ones:
  if a "deeply non-Gaussian" result dies under §3, that IS a result — write
  it up with the mechanism.
