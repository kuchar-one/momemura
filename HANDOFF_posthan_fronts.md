# Post-Hanamura fronts over ALL validated states, factor sweep (runbook)

Follow-up to `NG_VALIDATION_REPORT.md`. The 2026-07-11 Hanamura run
(`hanamura_ng/`) optimized only the **84 pre-Hanamura Pareto-front** states at a
single reduction factor. That is a lower bound on the true front, not the front:
the Hanamura step preserves neither the state nor its ranking. It reduces
detected photons and boosts success probability by a **state-dependent** factor
(measured **×13 … ×2.2×10⁵**), so a state that is *dominated* in (⟨O⟩, P) before
the optimization can land on the front *after* it.

This pipeline runs Hanamura over **all 2,395 validated sub-Gaussian states**, at
**three reduction factors (2.0 / 3.0 / 4.0)**, and rebuilds the fronts from
**every optimized point** — each state contributes up to three candidates and
the front picks its best factor. It fixes three things vs.
`run_hanamura_pareto.py`:

1. **Fronts merged per target** (B30F + B30), not per group.
2. **⟨O⟩ recomputed after Hanamura.** Photon reduction changes the state (the
   a141 champion's Wigner negativity moved 2.53→1.76), so ⟨O⟩ is *not*
   preserved. Both fronts live in **post-Hanamura value space and are sorted by
   post values** — unlike the previous plot, which held ⟨O⟩ fixed and only
   shifted P. Before/after P use the same `reduced_herald` estimator, so the
   ratio is self-consistent (fixes `NG_VALIDATION_REPORT.md` §5 caveat ii).
3. **Two fronts**, both from the Hanamura-optimized data:
   * **probability**: minimize ⟨O⟩_after, maximize P_after
   * **squeezing**  : minimize ⟨O⟩_after, minimize max necessary squeezing (dB)
     — the experimental-feasibility cost, read off the optimized architecture.

## Feasibility

Candidate set = 2,395 valid sub-G states (104 artifacts excluded). The heavy
per-state cost is the high-photon before-`reduced_herald`; it is computed **once
per state and shared across the three factors** (only the cheap two-step +
low-photon after-herald repeat), so the 3-factor sweep is ≈ **6–10 s/state**,
not 3×. CPU/`thewalrus`-bound (not VRAM-bound):

| | 1 proc | 8-way fan-out |
|---|---|---|
| 3-factor no-Wigner sweep (all 2,395) | ~4–7 h | ~35–55 min |

Wigner stays off here (not needed for either front); render it only for the
final-front states in pass 2.

## Scripts (committed under `scripts/`, resumable, unit-checked)

- **`run_hanamura_all.py`** — drives off `recompute_ng/verdicts.jsonl` (valid
  rows). Per state: before via `reduced_herald`; per factor: Hanamura via
  `optimize_gbs_architecture` (verify=False, uncapped squeezing ⇒ recorded
  max_sq is the *necessary* squeezing), after via
  `gen_hanamura_data.reduced_full_state` (architecture rule, same signal frame),
  ⟨O⟩ via `moment_operator`, plus before↔after fidelity and max_sq before/after.
  One JSONL row per (state, factor). Flags: `--reduction-factors 2.0,3.0,4.0`,
  `--nshards N --shard i`, `--skip-existing` (default on, keyed by state@factor),
  `--max-seconds`, `--wigner` (off by default), `--max-squeezing-db` (0 =
  uncapped).
- **`build_posthan_fronts.py`** — pools all shards/points, builds the PRE
  probability front (reference) and the POST **probability** and **squeezing**
  fronts per target, flags **PROMOTED** states (on the POST probability front,
  dominated before). Emits `posthan_fronts/{all_points.csv, posthan_summary.md,
  fronts_probability.png, fronts_squeezing.png, front_csvs/prob_<target>_rf<rf>.csv}`.

## Run order (cluster, x64, `.venv` py3.11, both A5000s)

```bash
cd /cluster/home/kuchar/code/momemura
git pull
export JAX_ENABLE_X64=1

# 0. smoke test (3 states × 3 factors, ~1–2 min) — confirm paths resolve
JAX_ENABLE_X64=1 python scripts/run_hanamura_all.py --limit 3

# 1. full 3-factor sweep, 8-way fan-out (CPU-bound; restartable via skip-existing)
for i in 0 1 2 3 4 5 6 7; do
  JAX_ENABLE_X64=1 python scripts/run_hanamura_all.py --nshards 8 --shard $i \
      > hanamura_all/shard$i.log 2>&1 &
done
wait     # ~35–55 min

# 2. rebuild both post-Hanamura fronts + promoted table + plots
python scripts/build_posthan_fronts.py --sweep-dir hanamura_all

# 3. (optional) pass-2 Wigner + full architecture, per reduction factor, for
#    ONLY the probability-front states of that factor:
for rf in 2p0 3p0 4p0; do rfv=${rf/p/.}; \
  JAX_ENABLE_X64=1 python scripts/run_hanamura_pareto.py \
      --pareto-dir posthan_fronts/front_csvs --groups "prob_*_rf${rf}" \
      --reduction-factor ${rfv} --out hanamura_posthan_front_rf${rf}; done
```

Read `posthan_fronts/posthan_summary.md` first: the headline is the **PROMOTED
count per target** (dominated before, on the post-Hanamura probability front).
Zero ⇒ the front-only run already had the front (a clean negative). Nonzero ⇒
those provenances are new front states the campaign missed. The per-factor
breakdown shows which reduction factor each front point prefers.

## Notes / gotchas

- `verdicts.jsonl` `run` is repo-relative; the sweep needs `results.pkl` +
  `config.json` under `experiments/` (same as the rescore). Unloadable runs are
  logged `[skip]` and simply don't enter the fronts.
- Uncapped squeezing is intentional so the squeezing axis is the *necessary*
  squeezing. To probe experimental feasibility, rerun with `--max-squeezing-db
  12` into a separate `--out`.
- `build_posthan_fronts.py --min-fidelity F` gates out after-states that moved
  too far from their original (reported, not gated, by default — the reduction
  is *meant* to move the state).
- Aarch64 sandbox can't run this (no jax/thewalrus wheels); cluster-only. Both
  scripts pass `py_compile`; the Pareto/promotion logic (both fronts) is
  unit-checked against hand-computed synthetic fronts.
```
