# NG-campaign results validation + Hanamura Pareto optimization (2026-07-11)

Follow-up to `HANDOFF_ng_results_validation.md`. Covers the full 2026-07-02 →
2026-07-11 ng-pipeline campaign: **α=1, β=1** (plus), **α=√2, β=1+1j**
(H-type, finished 07-10, added to scope), **α=2.7320508, β=1+1j** (magic),
and the unfinished **α=0, β=1** (margin-checked and dropped, §3). All
analysis ran on the downloaded archives with the exact moment pipeline
(JAX x64, CPU); every script referenced below is committed under `scripts/`.

## TL;DR

- **The campaign's sub-Gaussian results are real.** 2,395 of 2,499 unique
  sub-Gaussian archive states survive an independent rescore at L=200,
  BF=8192 (per-target survival 92 / 99 / 96 %); every casualty is an
  L-truncation artifact from the **unswept phase-C (Adam) stream**, none
  from the swept A/B archives. Zero decoupled-photon exploits; zero
  zero-photon or δ_ng≤0 sub-Gaussian cells anywhere.
- **Honest champions (L=200-exact ⟨O⟩, confirmed by the independent numpy
  path to ≤1e-14):**

  | target | ⟨O⟩ | vs G | vs G_N | log10 P | provenance |
  |---|---|---|---|---|---|
  | α=1, β=1 | **0.389324** | −0.277 | −0.321 | −4.43 | B30 d4, c1 phase C |
  | α=√2, β=1+1j | **0.502563** | −0.457 | −0.495 | −7.73 | B30 d5, c1 phase C |
  | α=2.732, β=1+1j | **0.491360** | −0.598 | −0.634 | −5.44 | B30 d5, c1 phase C |

- **The headline a1b1 number in the logs (0.3754) was truncation optimism.**
  Phase-C Adam runs score at L=50 and never sweep; the 0.3754 family is
  exactly 0.3917 at any L ≥ 120. The genuine d4 champion 0.389324 stands
  (stored 0.378182, drift +0.011 — inside the 0.02 sweep tolerance and
  confirmed by numpy). Net: Adam polish contributed ~0.002 of real
  improvement for a1b1, ~1e-4 for the other targets — most of its apparent
  gain was the L=50 truncation error it was chasing.
- **Hanamura control-parameter optimization over the validated fronts (84
  states, 100 % success)**: detected photons cut by ~55–60 % (mean 14.6→6.1,
  15.9→6.6, 14.5→7.5 per target), success probability up by a median
  ×13 / ×18 / ×27 (max ×2.2e5 for the a141 champion, whose raw P was 6e-9),
  max squeezing down ~2–3 dB, Wigner negativity preserved at the ensemble
  level. Fronts + Wigner panels under `hanamura_ng/`.
- **α=0, β=1 dropped**, as expected: best 0.4125 vs a1b1's 0.3893 at the
  same Gaussian limit (2/3), with the run killed mid cycle-0 depth-5.

---

## 1. Log audit (`scripts/audit_ngpipe_logs.py` → `ngpipe_audit_report.md`)

All four master logs parsed per-phase and per-watchdog-launch; checks per
HANDOFF §2:

- **Sweeps fired as designed**: 400 / 428 / 336 / 120 sweeps (a1b1 / a141 /
  a273 / a0b1), every ~256 gens plus the always-before-save sweep. The
  sweep-trigger bugfix works.
- **Removal counts** sit at a steady background of ~100–400 per sweep
  (fresh L=50 candidates cleaned at L=120) with **no late spikes** above
  3× the stream median anywhere except one benign 328-vs-median-97 event
  (a141 c1_d3_A, gen 512). No new exploit family emerged mid-campaign.
- **Artifact-led elites (the one real pathology, logged locations in the
  audit report):** in a few streams the reported best reverted upward right
  after a sweep — worst case `a1b1 c0_d5_B_b30`, where a persistent artifact
  family at ⟨O⟩≈0.453–0.457 was re-discovered by the elite stream and
  killed by three successive sweeps (best jumping 0.4534→0.7207). The saved
  archive is post-final-sweep and clean (§4 confirms), but at depth 5 the
  elite stream demonstrably wasted generations chasing L=50 mirages —
  an argument for running the elite stream at a higher in-loop L, or
  sweeping more often at depth ≥ 5.
- **Watchdog / stability**: no crashes, OOM, NaN, or recompile storms in
  any log. One cosmetic race: the two 2026-07-02 21:06 `a273 c0_d3_A`
  processes interleave lines in the master log (both run dirs saved fine).
- **Phase C caveat** (matters for §4): `--mode single` output has no sweep
  machinery, `Best Prob: 1.000000` is a placeholder, and `Injected Global
  Best` values are L=50 numbers. Phase-C bests must never be quoted
  unrescored.

## 2. Archive stats (`scripts/ng_archive_stats.py` → `ng_archive_stats.md`)

297 campaign run dirs (read from the masters' `Created output directory`
lines, so co-resident legacy runs are excluded). Per target × genotype ×
depth × phase tables in `ng_archive_stats.md`; key sanity results:

- **0** sub-Gaussian cells with < 0.5 effective photons (decoupled-photon
  exploit) and **0** with δ_ng ≤ 0, in every stratum.
- **0** cells with log10 P > 0; the A/B prob axis is real (–38 … –0.3).
- Depth trend: at fixed budget, d4/d5 dominate d3 for a1b1 (0.643 → 0.406
  best per B30F stratum), while a141/a273 saturate by d4 — their champions'
  active-leaf counts (§4.5) explain why: the discovered protocols only use
  3–4 leaves.

## 3. α=0, β=1: checked and dropped

Same witness limit as a1b1 (G = 2/3, G_N = 0.7108). Best archived state
0.4125 (margin −0.254) vs a1b1's 0.3893 (−0.277); the run died mid
`c0_d5_A` and its best cells are ordinary B30F d5 states (L=120-swept,
δ_ng≈4.2, 18–22 photons — honest but strictly dominated). Not carried into
Hanamura; no thesis value beyond this note.

## 4. Validity of the sub-Gaussian winners
   (`scripts/validate_ng_winners.py` → `recompute_ng/`)

Every unique sub-Gaussian state (deduped by (⟨O⟩, log P) signature across
the 297 runs: **2,499 states**) was independently rescored at **L=200,
BF=8192** with exact leaf-probability recomputation, top-decile tail mass,
and a numpy coupling audit (`n_eff` at coupling_eps=0.05, mirroring
`_effective_photons_static`).

| target | scored | valid | artifact | artifact phases | max valid |drift| |
|---|---|---|---|---|---|
| a1p00_b1p00 | 699 | 643 | 56 | C only | 0.0199 |
| a1p41_b1p41 | 871 | 860 | 11 | 10 C + 1 A | 0.0156 |
| a2p73_b1p41 | 929 | 892 | 37 | C only | 0.0191 |

- **All artifacts but one** came from unswept phase-C rows (max drift 0.79);
  the single A-phase escapee (a141 d3) slipped the in-loop tail gate but
  died here — 1/1,779 swept rows, i.e. the sweep+tail-gate combination is
  ~99.94 % tight at L=120 against an L=200 referee.
- **No decoupled states** (min n_eff among valid = 3.9; champions 17–24)
  and no suspicious probabilities (valid set log10 P ∈ (−14.9, −1.5) —
  nothing near the −40 red line); top-decile tail mass ≤ 6.7e-6 at L=200
  for every valid state (champions ≤ 1e-12).
- **Numpy cross-check (§3.2)**: top 10 per target reconstructed through
  `compute_equivalent_gaussian` → `reduced_herald` (thewalrus path,
  independent of the JAX scorer): worst |Δ⟨O⟩| = 1.1e-14, worst state
  fidelity 1 − 1.3e-15. Full table: `recompute_ng/numpy_crosscheck.jsonl`.
- **Bug found & fixed on the way**: `frontend/gaussian_decomposition.py::
  compute_equivalent_gaussian` was hard-coded to depth 3 (`range(8)` leaves,
  7 homodyne nodes) and silently truncated depth-4/5 genotypes — the numpy
  reference path and everything downstream of it (incl.
  `run_hanamura_pareto.py`) produced garbage for deep states. Now
  depth-general (leaf count from `leaf_active`, layers generated in
  `_static_tree` node order); the depth-3 unit tests
  (`tests/test_moment_scorer.py`, 12 numpy-vs-JAX cases) pass unchanged.

### Per-candidate table (top 10 per target)

⟨O⟩ stored vs L=200 vs numpy, exact log10 P, δ_ng (descriptor), n_eff:

**α=1, β=1** (G=0.6667):

| stored | L=200 | numpy | log10P | δ_ng | n_eff | depth/phase | verdict |
|---|---|---|---|---|---|---|---|
| 0.378182 | 0.389324 | 0.389324 | −4.43 | 4.29 | 24 | d4/C | valid |
| 0.375581 | 0.391599 | 0.391599 | −3.20 | 4.20 | 20 | d5/C | valid |
| 0.391604 | 0.391604 | 0.391604 | −3.20 | 4.20 | 20 | d5/B | valid |
| 0.375438 | 0.391734 | 0.391734 | −3.20 | 4.21 | 20 | d5/C | valid |
| (…6 more copies of the 0.3917 family, all valid at 0.3917…) | | | | | | | |

**α=√2, β=1+1j** (G=0.9596): top 10 all 0.50256–0.50269, drift −0.0013 …
0 (stored values slightly *pessimistic*), δ_ng≈3.92, n_eff=18, d3–d5 B/C —
all valid.

**α=2.732, β=1+1j** (G=1.0893): top 10 all 0.49136–0.49145, |drift| ≤
1.1e-4, δ_ng≈3.96, n_eff=17, d5 B/C — all valid.

### 4.5 Structure of the champions (§3.5 — thesis material)

All three are textbook breeding trees with near-balanced nodes and small
homodyne offsets, discovered without any structural prior:

- **plus (d4)**: 3 active leaves, one detector each, PNR (9, 8, 7);
  fired-node mixing θ ≈ 43–47°; |hx| ≤ 0.13.
- **H-type (d5)**: 3 active leaves, PNR (6, 6, 6) — perfectly symmetric;
  θ ≈ 43–48°; |hx| small except one −1.32 node.
- **magic (d5)**: **asymmetric 4-leaf cascade, PNR (8, 6, 2, 1)**, θ ≈ 45–49°,
  |hx| ≤ 0.34. This graded-photon-number breeding hierarchy is the
  previously-unknown magic-state protocol structure the campaign was run
  to find.

## 5. Hanamura optimization over the validated fronts
   (`scripts/run_hanamura_pareto.py --pareto-dir recompute_ng/pareto_fronts`
   → `hanamura_ng/`)

Validated (⟨O⟩, log P) Pareto fronts: 15 (a1b1) + 35 (a141) + 34 (a273)
states; all 84 optimized successfully (`han_valid` 84/84, no
ill-conditioned reductions). Aggregates (means unless noted):

| target | Nc before→after | P gain (median / max) | max-sq dB | negvol before→after |
|---|---|---|---|---|
| plus | 14.6 → 6.1 | ×13.1 / ×15.4 | 14.6 → 12.6 | 2.52 → 2.54 |
| H-type | 15.9 → 6.6 | ×17.8 / ×2.2e5 | 15.6 → 12.8 | 2.74 → 2.52 |
| magic | 14.5 → 7.5 | ×26.5 / ×97.5 | 16.0 → 12.6 | 3.49 → 2.85 |

Champions: plus 24→10 photons, P ×6.8; H-type 18→6, P 6.4e-9 → 4.1e-3
(×2.2e5 — its raw probability was pathological, which is exactly what the
pnr_max=10 headroom + Hanamura stage were for); magic 17→9, P ×97.5.

Deliverables: `hanamura_ng/ng_fronts.png` (before/after fronts, all three
targets), per-state `<group>/<run>_cell<idx>.json` (full before/after
architecture: U_passive, squeezings, displacements, PNR), `_wigner.png`
panels (champions all clearly negative), `states/*.npz` state vectors,
`hanamura_summary.csv` + `REPORT.md`.

**Caveats:** (i) 81/84 "after" states use the multi-detector
`architecture_rule` reconstruction (only k_eff=1 states get the exact core
state) — after-negvol numbers are indicative, not exact; the *probability*
and *architecture* numbers are exact. (ii) "P before" here is the
reduced-herald PNR probability at the herald cutoff, consistent within the
before/after comparison but not identical to the archives' leaf-product
log10 P (both are quoted where used).

## 6. Ground-rule notes

- No validation thresholds were loosened; the depth-3 truncation bug was
  fixed at the cause (`gaussian_decomposition.py`), verified against the
  unchanged depth-3 unit tests and 30/30 machine-precision cross-checks.
- Honest negatives: the phase-C Adam stage's headline gains were mostly
  truncation error (its stored bests are 0.013–0.016 optimistic for a1b1);
  depth 5 added nothing over depth 4 for a1b1 (honest d5 best 0.3916 vs d4
  0.3893); the a1b1 depth-5 consolidation stream repeatedly chased one
  artifact family (§1).

## 7. Reproduction

```bash
# 1. log audit
python scripts/audit_ngpipe_logs.py ngpipe_master_a1b1.log ngpipe_master_a273b11.log \
    ngpipe_master_a141b11.log ngpipe_master_a0b1.log --out ngpipe_audit_report.md
# 2. archive stats (resumable)
python scripts/ng_archive_stats.py --logs ngpipe_master_*.log \
    --state recompute_ng/ng_scan_state.jsonl --out-prefix ng_archive_stats
# 3. high-L revalidation of every unique subG state (resumable; CPU-safe)
JAX_ENABLE_X64=1 python scripts/validate_ng_winners.py \
    --scan-state recompute_ng/ng_scan_state.jsonl --out recompute_ng
# 4. numpy cross-check, top-10 per target
JAX_ENABLE_X64=1 python scripts/numpy_crosscheck_ng.py --top 10
# 5. Hanamura over the validated fronts (resumable via --skip-existing)
JAX_ENABLE_X64=1 python scripts/run_hanamura_pareto.py \
    --pareto-dir recompute_ng/pareto_fronts --out hanamura_ng --skip-existing
python scripts/plot_ng_fronts.py
```
