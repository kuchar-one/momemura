# Re-score archaeology report

_Generated 2026-06-28 12:16:46 | L_search=50 L_high=120 bf_high=8192 maxf=8 tol=0.02 neg_tol=0.001 per_run_cap=8_

## Totals

- roots scanned: 4  (experiments, output_old/experiments, output_oldold/experiments, output/experiments)
- groups discovered: 7; runs discovered: 42
- runs **re-scored**: 40 | **undecodable**: 0 | **unloadable**: 2  (accounting: 40+0+2=42 == 42)
- cells re-scored: 320 | **valid**: 268 | **artifact**: 52 (16.2%)
- undecodable/unloadable reasons: UnpicklingError=2
- artifact reasons: l_truncation=50, over_budget=2

## Empirical target maps

| folder b-code | resolved beta |   | folder a-code | resolved alpha |
|---|---|---|---|---|
| 1p41 | (1+1j) |   | 1p00 | (1+0j) |
|  |  |   | 1p41 | (1.4142+0j) |
|  |  |   | 2p73 | (2.7320508+0j) |

## Per target group: genuine best vs. originally-stored best

| group | target (a,b) | best genuine ⟨O⟩ | vs_gaussian | vs_gs | prob | photons | provenance | stored best ⟨O⟩ | artifact gap |
|---|---|---|---|---|---|---|---|---|---|
| 00B_c30_a1p00_b1p41 | ((1+0j), (1+1j)) | **1.0722** | +0.0722 | +0.9670 | 1.00e+00 | 0 | experiments/20260514-100523_p64_i2500#250 | 0.4168 | +0.6554 |
| 00B_c30_a1p41_b1p41 | ((1.4142+0j), (1+1j)) | **0.8871** | -0.0725 | +0.7814 | 2.91e-05 | 8 | experiments/20260614-134123_p64_i2500#1301 | 0.3699 | +0.5172 |
| 00B_c30_a2p73_b1p41 | ((2.7320508+0j), (1+1j)) | **0.5726** | -0.5167 | +0.4697 | 2.46e-05 | 11 | experiments/20260616-165402_p64_i2500#1050 | 0.3594 | +0.2132 |
| 0_c30_a1p00_b1p41 | ((1+0j), (1+1j)) | **1.0375** | +0.0375 | +0.9323 | 1.66e-03 | 4 | experiments/20260226-125913_p64_i2500#1054 | 0.7856 | +0.2518 |
| B30B_c30_a1p41_b1p41 | ((1.4142+0j), (1+1j)) | **1.0047** | +0.0451 | +0.8990 | 7.95e-04 | 6 | experiments/20260226-015926_p64_i2500#800 | 0.8415 | +0.1632 |
| B30_c30_a1p00_b1p41 | ((1+0j), (1+1j)) | **0.7451** | -0.2549 | +0.6400 | 1.12e-06 | 13 | experiments/20260221-175753_p64_i2500#64 | 0.5058 | +0.2393 |
| B30_c30_a1p41_b1p41 | ((1.4142+0j), (1+1j)) | **0.6936** | -0.2660 | +0.5879 | 1.31e-06 | 13 | experiments/20260224-095158_p64_i2500#2051 | 0.5152 | +0.1784 |

_`vs_gaussian` < 0 means a genuine non-Gaussian advantage (and must show Wigner negativity to survive the Hudson gate). `artifact gap` = genuine best − stored best; large positive means much of the old 'record' was a truncation/placeholder artifact._

## Narrative

- 40/42 runs (95%) decoded & re-scored with the exact pipeline; the rest are logged in `undecodable.csv`.
- 268/320 re-scored cells survived all physical-validity filters.
- genuine sub-Gaussian (⟨O⟩ < Gaussian limit, Wigner-negative) cells: 59.
- `fake_subgaussian` rejections (claimed sub-Gaussian but Wigner-positive → impossible by Hudson's theorem): 0.

## Files

- `all_solutions.parquet` — one row per re-scored cell (full schema).
- `pareto_fronts/<group>.csv` — artifact-free ⟨O⟩-vs-logP front per group.
- `best_states/<group>/` — genotype.npy, params.json, psi_hi.npy, wigner.png, meta.json.
- `undecodable.csv` — every run not re-scored, with reason.
- `run_ledger.csv` — status of every discovered run.
- `plots/` — per-group Pareto fronts, artifact-reason histogram.
