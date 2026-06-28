# Re-score archaeology — handoff

Implements `PROMPT_rescore_archaeology.md`: re-decode and exactly re-score every
recoverable past run, purge artifacts, surface genuine per-target optima.

## What's here
- `scripts/rescore_all_experiments.py` — the sweep (CLI, x64, GPU-aware, batched).
- `tests/test_rescore_archaeology.py` — `pytest tests/ -k rescore` → **12 passed**.
- `recompute/` — **limited smoke output** (`--groups '*_b1p41' --max-per-group 2
  --per-run-cap 8`, 42 runs / 320 cells). The full run below **overwrites** it.

## Environment
A `.venv` was created at the repo root (the prompt's expected venv was missing):
```
.venv/bin/python   # numpy, scipy, jax[cpu] 0.10, thewalrus, qutip, pandas, pyarrow, matplotlib
```
`qdax` is **not** needed: MOME repertoires are recovered with a generic stand-in
unpickler (`load_run_arrays`). Always run with `JAX_ENABLE_X64=1` (the moment
recurrence underflows in 32-bit). The first call per target builds the N=1000 GKP
operator via qutip (~12 s) and caches it to `src/cache/operators/`.

## Smoke results (sanity)
- accounting balances: 40 re-scored + 0 undecodable + 2 unloadable = 42 discovered.
- 268/320 cells valid; 52 artifacts (l_truncation=50, over_budget=2); 0 `fake_subgaussian`.
- artifact purging confirmed, e.g. `00B_c30_a1p00_b1p41`: stored "record" ⟨O⟩=0.417
  (a Fock-truncation ghost) → genuine best ⟨O⟩=1.072 (no advantage). Harder targets
  (α=1.41, 2.73) keep genuine Wigner-negative sub-Gaussian states (59 across the slice).
- β-less moment runs resolved via the empirical map `b1p41 → (1+1j)` (cross-validated
  against every config with an explicit β: no contradictions).

## Run the FULL rescoring (every selected cell of every run, `--per-run-cap 0`)
From the repo root:
```bash
PYTHONUNBUFFERED=1 JAX_ENABLE_X64=1 nohup .venv/bin/python \
  scripts/rescore_all_experiments.py \
  --per-run-cap 0 \
  --fidelity-subsample 4 \
  --progress-every 25 \
  --out recompute/ \
  > recompute_full.log 2>&1 &
```
This scans all four documented roots (`experiments/`, `output_old/experiments/`,
`output_oldold/experiments/`, `output/experiments/`) = ~3,493 runs. `--per-run-cap 0`
re-scores each run's full Pareto-front ∪ MAP-Elites-bin-elite set (no cap).

Notes / knobs:
- **Time**: CPU-bound and long (smoke was ~4.5 s/run at cap 8; full is the entire
  Pareto+elite set per run, so budget hours). On a GPU box JAX uses it automatically.
  To throttle without losing coverage breadth, add `--max-per-group N`.
- **Harder targets need a higher cutoff.** α=2.73 states don't fully converge by
  L=120 (many get a correct `l_truncation` flag). For those groups, optionally
  re-run with `--groups '*_a2p73_*' --l-high 200 --bf-high 16384`.
- Other roots exist (`output_old/_legacy`, `*_beforefix`, `*_fock_old`); add them
  with `--roots experiments output_old/experiments output_oldold/experiments \
  output/experiments output_old/experiments_beforefix output_oldold/experiments_fock_old`
  if you want them mined too (they are not in the documented set by default).

## Deliverables (under `--out`, default `recompute/`)
`all_solutions.parquet` (+ `all_solutions_sample.csv`), `pareto_fronts/<group>.csv`,
`best_states/<group>/` (genotype.npy, params.json, psi_hi.npy, wigner.png, meta.json),
`REPORT.md`, `undecodable.csv`, `run_ledger.csv`, `plots/`.
