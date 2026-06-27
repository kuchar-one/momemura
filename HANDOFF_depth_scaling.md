# Handoff: depth-generalized moment scorer + 24GB-VRAM depth-4/5 runs

**Date:** 2026-06-27 · Follows `HANDOFF_coverage_and_minimum.md` (item B: push past the
depth-3 ⟨O⟩≈0.5725 floor by adding breeding depth).

## TL;DR

The moment scorer was **hardcoded to depth 3** (8 leaves, N=24, 7-node tree, fixed
17-mode reduced state, `range(8)` everywhere). The decoder was already depth-parametric,
so a `--depth 4` run would have silently scored only the first 8 of 16 leaves with the
wrong tree routing. This change makes the **whole moment path depth-generic** and adds the
VRAM controls needed to keep deep trees inside the 24GB A5000s.

**Key physics-of-memory point:** depth does **not** blow up VRAM. The dominant cost is the
per-genotype Hermite box, which is `~maxf·(L+15·maxf)·BF` complex128 and **depth-independent**
(it's a `1+maxf`-mode object no matter how many control slots the tree carries). Depth only
grows the *cheap* Gaussian-side matrices (`2N×2N`, `N=3·2^depth`) and the vac-conditioning
loop length. So depth-4 fits comfortably, and **depth-5 is VRAM-feasible too** — the real
limits at depth 5 are compile time, genotype dimension, and search difficulty, not memory.

## What changed

`src/simulation/jax/moment_scorer.py`
- `_static_tree(depth)` (cached) builds the leaf-mode layout, balanced-binary mixing tree
  (mix-node order = layer 1 → root, matching the decoder), and keep-projection for any depth.
  **depth=3 reproduces the original `_TREE`/`_KEEP`/`_LEAF_X` tables bit-for-bit** (verified).
- `jax_equivalent_gaussian_static(params, depth=3)`, `jax_reduced_herald_static(..., depth, maxf)`,
  `_leaf_prob_product_static(..., depth)`, `_effective_photons_static(..., depth)` — all
  depth-parametric. Reduced state is `M = 1 + 2·2^depth` modes (depth3→17, depth4→33, depth5→65).
- Dynamic path (`extract_structure`, `jax_equivalent_gaussian`, `moment_score_one`,
  `_struct_fired_product`, `_numpy_leaf_prob_product`) generalized off `len(leaf_active)`.
- **VRAM controls:**
  - `moment_chunk` — shards the population vmap into fixed-size chunks (one compiled graph;
    last chunk padded then truncated). Depth-derived default: full pop ≤depth3, 16 @depth4,
    4 @depth5, 2 @depth≥6. **The primary VRAM knob.**
  - `moment_remat` — `jax.checkpoint` on the per-genotype loss (recompute forward in backward).
    ~2× compute for a big peak-VRAM cut. Default on for depth≥4.
  - `moment_maxf` — in-graph fired-mode cap (Hermite box = `1+maxf` modes), depth-independent.

`run_mome.py`
- New flags `--moment-maxf` (default 8), `--moment-chunk` (0=auto), `--moment-remat {auto,on,off}`.
- Startup `[moment VRAM]` advisory line (rough peak-GB estimate vs the 24GB budget).
- **D3 photon-descriptor rescale (handoff item A):** the photon axis now uses a
  budget-realistic max (`_budget_max_photons` from BF/pnr_max) instead of the unreachable
  theoretical `active·n_control·pnr_max`. For BF=1024 this is ~32 (matches observed ~31),
  not 240 — so the coverage metric stops being dragged down by ~80% dead grid. Only applied
  for `--scorer moment`.

`scripts/validate_moment_archive.py` — threads `depth`/`maxf` from the run config.

## Validation done

- Topology: depth-3 `_static_tree` == original hardcoded tables (exact); depth 4/5 satisfy
  all structural invariants (node/layer counts, A/B=3·min rule, root merges all leaves).
- Forward pass (x64) at depth 3/4/5: cov modes 34/66/130, eff/dens/psi shapes correct,
  herald norm = 1.0000, ⟨O⟩ and prob finite.
- Budget helper: BF 1024→32, 4096→33, 8192→46 photons.
- Both edited files byte-compile.
- (Gradient/vmap/chunk path compiles correctly; full grad timing was only checked on the
  cluster GPUs — the local CPU sandbox is too slow to compile the depth-4 AD graph quickly.)

## Run commands (sized for 24GB A5000)

Your depth-3 baseline (unchanged behavior — depth≤3 takes the validated full-pop, no-remat path):

```bash
python run_pipeline.py --backend jax --pop 64 --iters 2500 --cutoff 30 \
  --genotype 00B --dynamic-limits --modes 3 --pnr-max 15 --depth 3 \
  --target-alpha 2.7320508 --target-beta "1+1j" --chunk-size 50 --emitter hybrid \
  --seed-scan --scorer moment --moment-cutoff 50 --moment-bf 1024 --moment-fast --single-run
```

### Depth 4 (the item-B test: does ⟨O⟩ drop below 0.5725?)

```bash
python run_pipeline.py --backend jax --pop 64 --iters 2500 --cutoff 30 \
  --genotype 00B --dynamic-limits --modes 3 --pnr-max 15 --depth 4 \
  --target-alpha 2.7320508 --target-beta "1+1j" --chunk-size 50 --emitter hybrid \
  --seed-scan --scorer moment --moment-cutoff 50 --moment-bf 1024 --moment-fast \
  --moment-maxf 10 --moment-chunk 16 --moment-remat auto \
  --moment-l-high 120 --moment-validate-every 250 --moment-bf-high 8192 \
  --moment-validate-chunk 8 --single-run
```

- `--moment-maxf 10`: depth-4's richer trees can fire more controls at once; 10 captures the
  multi-fired tail that BF=1024 still allows (prod(n+1)≤1024 ⇒ ≤~10 single-photon modes)
  without dropping it as over-budget. Costs ~1.5× box memory — still tiny.
- `--moment-chunk 16`: depth-4 estimate is only ~1–3 GB even at full pop, so 16 is very safe;
  you can raise to 32/64 for speed if `nvidia-smi` shows ample headroom (watch the
  `[moment VRAM]` line at startup and actual usage on the first generation).
- If depth-4 best states carry more photons/squeezing and you see over-budget drops, bump
  `--moment-bf 4096` (and check fp/norm) — VRAM still fine; re-validate at high L.

**Higher-L artifact detection is automatic — you don't need extra flags for it.**
`--moment-cutoff 50` is only the *search* L. In the QDax/MOME phase the optimizer
runs a periodic **dual-L validation sweep** by default: every `--moment-validate-every`
(250) generations it re-scores the whole archive at `--moment-l-high` (auto =
max(2·cutoff,120) = 120) with `--moment-bf-high` (8192), drops any cell whose
search-L ⟨O⟩ disagrees by >`--moment-validate-tol` (0.02) or whose herald wasn't
normalised, and refreshes survivors to the exact high-L ⟨O⟩. The flags above just
make those defaults explicit. Note this sweep is the **heaviest VRAM moment** of a
deep run (L=120 + BF=8192 is a much bigger Hermite box than the search), which is
why `--moment-validate-chunk` defaults smaller than `--moment-chunk` and shrinks
with depth — lower it (or `--moment-bf-high`) if validation spikes memory. The
single-objective seeding phase has no in-loop sweep; the post-hoc script below
covers it.

Post-hoc exact re-validation (drop L-truncation artifacts, refresh to exact high-L ⟨O⟩):

```bash
JAX_ENABLE_X64=1 python scripts/validate_moment_archive.py \
  --group 00B_c30_a2p73_b1p41 --l-search 50 --l-high 120 \
  --target-alpha 2.7320508 --target-beta "1+1j" --write
```

(The QDax/MOME `config.json` stores `target_alpha` as a bare number and **omits
`target_beta`**, so pass the target explicitly — the script now requires it and
errors clearly instead of failing with a cryptic `complex()` message.)

### Depth 5 (best-effort — VRAM is fine, but compile/search-bound)

```bash
python run_pipeline.py --backend jax --pop 32 --iters 2500 --cutoff 30 \
  --genotype 00B --dynamic-limits --modes 3 --pnr-max 15 --depth 5 \
  --target-alpha 2.7320508 --target-beta "1+1j" --chunk-size 50 --emitter hybrid \
  --seed-scan --scorer moment --moment-cutoff 50 --moment-bf 1024 --moment-fast \
  --moment-maxf 10 --moment-chunk 4 --moment-remat on \
  --moment-l-high 120 --moment-validate-every 250 --moment-bf-high 4096 \
  --moment-validate-chunk 2 --single-run
```

- Memory is **not** the wall at depth 5 (est. <1 GB at chunk=4). Expect the costs to be:
  long first-compile (31-node unrolled tree under AD), doubled genotype dimension (32 leaves),
  and a much larger/harder search space. Consider fewer iters for a first probe, lower `--pop`,
  and seeding from the depth-4 archive (via the upgrade path) once depth-4 has good cells.

## Gotchas
- Moment scorer still **requires `JAX_ENABLE_X64=1`** (run_mome sets it when `moment` is in argv).
- `--moment-chunk` changes *only* peak VRAM, not results (chunks share one graph; verified
  full-vs-chunked parity at depth 4).
- Don't seed a fresh deep run from an **old Fock** archive (injects truncation artifacts);
  the `upgrade_genotype` seed path across depths is fine.
- The depth-3 path is byte-for-byte the prior behavior (full pop, remat off, original tree).
