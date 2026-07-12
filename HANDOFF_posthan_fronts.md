# Post-Hanamura fronts over ALL validated states (runbook)

Follow-up to `NG_VALIDATION_REPORT.md`; read `HANAMURA_REDUCTION_DIAGNOSIS.md`
first — it explains why the naive photon-reduction scoring produced
above-Gaussian ⟨O⟩ (the reduction preserves the output only *up to a Gaussian
unitary*, which must be re-applied before scoring) and why aggressive reduction
genuinely can't help photon-efficient champions.

The 2026-07-11 Hanamura run (`hanamura_ng/`) optimized only the 84 pre-Hanamura
front states, at one reduction factor, and **never recomputed ⟨O⟩ after**. This
pipeline runs Hanamura over **all 2,395 validated sub-Gaussian states**, sweeps
reduction factors, **recomputes ⟨O⟩ correctly (up to a final Gaussian)**, and
rebuilds the fronts from every optimized point. Key design points:

1. **Fronts merged per target** (B30F+B30) and **sorted by post-Hanamura
   values**, not the prior ones.
2. **⟨O⟩_after is minimized over the final single-mode Gaussian** (the reduction
   is only defined up to one — paper Eqs. 1/38, thesis Alg. 1 l.335). Same
   freedom the MOME final-Gaussian gene used for ⟨O⟩_before. The stale-frame
   value is kept as `exp_after_raw` for contrast. The Gaussian search is
   **warm-started from the nearest already-solved neighbour** in objective space
   (~2.5× fewer evals; neighbours share a similar optimal Gaussian).
3. **Reduction factor sweep `1.0,2.0,3.0,4.0`**:
   - **1.0 = damping-only**: exactly output-preserving (⟨O⟩ unchanged, fid≈1),
     the *lossless* probability-boost front — this is the correct basis for the
     "dominated-before, promoted-after" question (states promoted purely by the
     P gain, no quality loss).
   - **≥2.0**: a genuine quality-vs-(photons, squeezing) tradeoff. Expect real
     degradation on efficient champions — a legitimate negative result.
4. Two fronts per target: **probability** (⟨O⟩_after vs P_after) and
   **squeezing** (⟨O⟩_after vs max necessary squeezing dB).

## Feasibility

2,395 valid states × 4 factors. The heavy before-`reduced_herald` is computed
once per state and shared across factors; factor 1.0 needs no Gaussian
alignment; factors ≥2 add a warm-started 5-parameter Gaussian minimization
(~1–3 s each). Rough cost ~10–15 s/state ⇒ ~7–10 h single-proc, **~50–75 min at
8-way**. CPU/`thewalrus`/`scipy`-bound, not VRAM-bound.

## Scripts (committed under `scripts/`, resumable, unit-checked)

- **`run_hanamura_all.py`** — per state: before via `reduced_herald`; per factor:
  Hanamura via `optimize_gbs_architecture`, after-state via
  `reduced_full_state`, then `exp_after` = **min ⟨O⟩ over final Gaussian**
  (`min_exp_over_gaussian`, warm-started), `fidelity_after_before` = fidelity up
  to that Gaussian, `prob_after` = damping-optimized success prob
  (`han["prob_after"]`, the real boost; herald value kept as
  `prob_after_herald`), `max_sq_after`. Records `exp_after_raw`/`fidelity_raw`
  (stale frame) alongside. Flags: `--reduction-factors`, `--nshards/--shard`,
  `--skip-existing` (keyed state@factor), `--max-seconds`, `--align-cut`,
  `--no-align` (A/B with the old scoring), `--wigner`.
- **`build_posthan_fronts.py`** — pools all points, builds the per-target
  probability + squeezing fronts from the frame-corrected values, flags
  PROMOTED. Emits `posthan_fronts/{all_points.csv, posthan_summary.md,
  fronts_probability.png, fronts_squeezing.png, front_csvs/prob_<target>_rf<rf>.csv}`.

## Run order (cluster, x64, `.venv` py3.11)

```bash
cd /cluster/home/kuchar/code/momemura
git pull
export JAX_ENABLE_X64=1
pip install scipy >/dev/null 2>&1 || true   # min-<O>-over-Gaussian needs scipy

# 0. smoke test (3 states × 4 factors) — check exp_after is now sane (<= ~G),
#    and compare exp_after vs exp_after_raw to see the frame effect
JAX_ENABLE_X64=1 python scripts/run_hanamura_all.py --limit 3

# 1. full sweep, 8-way fan-out
for i in 0 1 2 3 4 5 6 7; do
  JAX_ENABLE_X64=1 python scripts/run_hanamura_all.py --nshards 8 --shard $i \
      > hanamura_all/shard$i.log 2>&1 &
done; wait                                            # ~50–75 min

# 2. rebuild fronts + promoted table + plots
python scripts/build_posthan_fronts.py --sweep-dir hanamura_all

# 3. (optional) pass-2 Wigner + architecture, per factor, for the prob-front only
for rf in 1p0 2p0 3p0 4p0; do rfv=${rf/p/.}; \
  JAX_ENABLE_X64=1 python scripts/run_hanamura_pareto.py \
      --pareto-dir posthan_fronts/front_csvs --groups "prob_*_rf${rf}" \
      --reduction-factor ${rfv} --out hanamura_posthan_front_rf${rf}; done
```

## What to check first (sanity, given the earlier bug)

- In the smoke test, `exp_after` (frame-corrected) should be **= `exp_before`**
  at factor 1.0 (fidG 1.000) and only mildly above for gentle factors — NOT the
  1.1–2.0 blow-ups of the stale-frame `exp_after_raw`. If `exp_after` is still
  ≫ G at factor 1.0, the Gaussian alignment isn't working — stop and check
  `min_exp_over_gaussian` / `--align-cut`.
- **Factor-1.0 P boost**: with `stable_control_probability` wired in, rf1 should
  now show P **> ×1** (the damping boost), not the old ×1.0 fallback, even for
  20-photon states. `prob_after_stable` is recorded next to `prob_after`.
- **Self-check**: the run auto-compares the stable density-matrix probability
  against the loop hafnian on every ≤16-photon state and prints
  `[WARN] stable P != hafnian P ...` on any >1% disagreement. **A clean run
  should print no such warnings** — if it does, the stable path is off; stop.
- `posthan_summary.md`: **PROMOTED count per target**. Expect promotions mostly
  from **factor-1.0 (damping-only)** points (lossless P boost, ⟨O⟩ unchanged);
  factor ≥2 points trade real ⟨O⟩ for larger P/lower photons.

## Notes

- `verdicts.jsonl` is repo-relative; needs `results.pkl`+`config.json` under
  `experiments/`. Unloadable runs log `[skip-state]`.
- Uncapped squeezing (default) ⇒ `max_sq_after` is the *necessary* squeezing.
- Sandbox (aarch64) can't run the jax/thewalrus path; cluster-only. The
  `min_exp_over_gaussian` helper and the front/promotion logic are unit-checked
  (scipy-only) on synthetic data; both scripts pass `py_compile`.
```

---

## 2026-07-12 update: sweep fixed after shard-log review (commit ae6d1f8)

The 2026-07-11/12 shard logs exposed four problems, all fixed; **stop the old
run and relaunch from a fresh output dir** (old JSONLs remain readable by the
new `build_posthan_fronts.py`, but the rerun is ~10x faster and gate-clean):

1. **rf1 "O drift" was a yardstick artifact** — rows compared the archive
   L=200 value against an hcut reconstruction (spurious ±0.01–0.04). rf1 now
   skips reconstruction entirely (damping preserves the state exactly) and
   fronts rank on `exp_after_cal`, the after-quality calibrated to the archive
   scale.
2. **NaN / exp_after=0.0 rows** (e.g. 13453, 7828, 11157) — two causes: the
   `optimize_damping` "identity" fallback actually installed the *vacuum*-
   damped generator (t=coth(1e3)≈1 is full damping, not identity), and
   before-states whose reduced-herald P underflows to 0 produced garbage
   downstream. Both gated now (`before_ok`/`after_ok`; a 0.0 exp_after would
   have ranked as a perfect GKP state in the fronts).
3. **Speed** — `min_exp_over_gaussian` no longer calls `scipy.expm` per
   evaluation (once-per-cutoff eigendecompositions, unit-tested identical:
   `tests/test_gaussian_apply_fast.py`); damping optimizes only fired modes
   with warm starts; duplicate reduced patterns across factors (rf2/rf3
   collide for ~1/3 of states) are computed once. Locally: ~10 s per state x 4
   factors (was ~2 min).
4. **n=0 heralds**: handled *analytically* now — vacuum-herald control modes
   are damped to t=1 inside `optimize_damping`, which reproduces the exact
   vacuum-projection absorption of the mode into U_G (H champion: x4599
   absorption, x8213 total with amplification; verified against
   `scripts/absorb_zero_heralds.py`). The old optimizer *could* find this
   numerically (and did for the H champion), but is no longer trusted to.

Sanity checks for the rerun (replacing §"What to check first"):
- rf1 rows must print `O x.xxxx->x.xxxx` **identical** and `fidG 1.000`.
- the damping gain is printed twice (`xA, damp xB`): hafnian vs stable-ratio
  estimates — they should agree wherever both are finite.
- `[WARN] stable P != hafnian P` should now appear only above the 1e-13
  floor; any such warning is real and worth a look.
- `[invalid-before]` lines are expected for a handful of extreme-squeezing
  states (their reduced-herald P underflows); they are recorded as
  `before_ok=False` and excluded from fronts.

```bash
cd /cluster/home/kuchar/code/momemura && git pull
export JAX_ENABLE_X64=1
mkdir -p hanamura_all2
for i in $(seq 0 23); do OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python scripts/run_hanamura_all.py --nshards 24 --shard $i --out hanamura_all2 \
  > hanamura_all2/shard$i.log 2>&1 & done; wait
python scripts/build_posthan_fronts.py --sweep-dir hanamura_all2
```
