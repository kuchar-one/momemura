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

- In the smoke test, `exp_after` (frame-corrected) should now be **≤ ~G** for
  factor 1.0 (exactly `exp_before`) and only mildly above for gentle factors —
  NOT the 1.1–2.0 blow-ups seen with the stale-frame scoring. If `exp_after` is
  still ≫ G at factor 1.0, the Gaussian alignment isn't doing its job — stop and
  investigate `min_exp_over_gaussian` / `align_cut`.
- `posthan_summary.md`: **PROMOTED count per target**. Given the diagnosis,
  expect promotions to come mostly from the **factor-1.0 (damping-only)** points
  (lossless P boost); factor ≥2 points should mostly sit at higher ⟨O⟩.

## Notes

- `verdicts.jsonl` is repo-relative; needs `results.pkl`+`config.json` under
  `experiments/`. Unloadable runs log `[skip-state]`.
- Uncapped squeezing (default) ⇒ `max_sq_after` is the *necessary* squeezing.
- Sandbox (aarch64) can't run the jax/thewalrus path; cluster-only. The
  `min_exp_over_gaussian` helper and the front/promotion logic are unit-checked
  (scipy-only) on synthetic data; both scripts pass `py_compile`.
```
