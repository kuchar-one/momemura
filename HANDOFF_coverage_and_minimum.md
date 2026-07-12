# Handoff: stuck ⟨O⟩≈0.5725 minimum + stuck ~30% MAP-Elites coverage

**Date:** 2026-06-27 · **Target:** `00B_c30_a2p73_b1p41` (α=2.7320508, β=1+1j), depth 3, moment scorer.
**Run analyzed:** `experiments/00B_c30_a2p73_b1p41/20260616-153549_p64_i2500` (single-objective phase, loads locally). The four later `*_p64_i2500` runs in that folder are QDax MOME repertoires (flax structs) and **do not unpickle without qdax installed** — analyze them on the cluster.

**Reference values:** GS Eig (ground state) = 0.30935 · Gaussian Limit = 1.08932 · best found = **0.5725**.

---

## TL;DR

Two *separate* things, both now explained with data — neither is a bug in the scorer:

1. **⟨O⟩ ≈ 0.5725 is an honest minimum, not an artifact.** It survives L=120 re-validation, best genotype has herald norm = 1.0000 at L=60, fp=180 (well under the BF=1024 budget). It is *not* truncation- or budget-limited. It sits ~45% of the way from the Gaussian limit to the ground state. To go lower you almost certainly need **more breeding depth**, not more cutoff/BF.

2. **The ~30% coverage is mostly a mis-scaled descriptor grid, not weak exploration.** The photon axis (D3) is scaled to a theoretical max of **240 photons**, but nothing reachable exceeds **~31 photons**. Only **2 of the 10 D3 bins** can ever be occupied. 80% of that axis is dead grid, so coverage is structurally capped near 20–30% even with perfect search.

---

## Evidence

### 1. The minimum is honest and not budget-limited
From the single-objective archive (62 decoded valid cells), best cells by ⟨O⟩:

| idx | ⟨O⟩ | ΣPNR | squeeze dB | norm@L=60 | fp=∏(nⱼ+1) |
|----:|-----:|----:|-----:|------:|----:|
| 64 | 0.5725 | 11 | 10.8 | 1.0000 | 180 |
| 52 | 0.5732 | 11 | 10.8 | 1.0000 | 180 |
|  2 | 0.5746 | 11 | 11.0 | 1.0000 | 180 |
| 19 | 0.5817 | 13 | 10.1 | 1.0000 | 300 |
| 27 | 0.6554 | 13 | 12.7 | 1.0000 | 300 |

- **0/62** valid cells would be rejected under the search settings (L=60, norm>0.99, BF=1024). The L/norm gate is *not* excluding anything here.
- Best solutions sit at fp=180–300, far below the BF=1024 ceiling → **raising BF will not improve ⟨O⟩** at depth 3.
- The population explores ΣPNR 0→32 and squeezing 4.7→17.5 dB, yet the optimum is at the *moderate* ΣPNR≈11, ~11 dB. More photons/squeezing past that point does not help → consistent with a genuine basin, not an unexplored cliff.

### 2. Coverage is grid-limited
Grid for depth 3 / genotype 00B (`run_mome.py` ~L825–860): D1 active modes 0–8 (9 bins), D2 max-PNR 0–15 (5 bins), **D3 total photons 0–240 (10 bins)** → 450 centroids. The code's "achievable 53.8%" check (`max_total = active·n_control·pnr_max = 8·2·15 = 240`) assumes every one of 8 leaves fires 15 photons — which is astronomically over the BF budget *and* has vanishing heralding probability.

Actual D3 occupancy of the archive:

```
D3 bin edges (photons): [0, 27, 53, 80, 107, 133, 160, 187, 213, 240]
 bin0 (~0):   54 cells
 bin1 (~27):  11 cells
 bin2..bin9 (53–240):  0 cells     <-- 80% of the axis is empty
max photons_eff observed: 30.9   (median 4.0)
```

**Why bins 2–9 are unreachable:** validity requires fp = ∏(nⱼ+1) ≤ BF = 1024. Spread over up to 8 active leaves, that caps total photons at ≈ 8·(1024^(1/8) − 1) ≈ 11; even concentrated, realistic heralded states top out near ~31 photons (observed max 30.9). The grid's 240 ceiling corresponds to a state that can never be produced under the budget (or under any finite heralding probability). So ~8/10 of the photon axis is dead, dragging the coverage fraction down. **30% is close to the reachable ceiling — the optimizer isn't failing, the denominator is wrong.**

---

## Solutions

### A. Fix coverage by rescaling the photon descriptor (high confidence, cheap)
In `run_mome.py` (~L840–852) the D3 axis is `linspace(0, max_active*pnr_max*n_control, 10)`. Replace the theoretical max with a **budget-realistic max**. Two options:

- **Quick:** cap `max_photons` at a value derived from BF, e.g. `max_photons = int(8 * (moment_bf ** (1/ max_active) - 1)) + headroom` (≈ 30–40 here), or simply hard-cap at ~40 for this family.
- **Cleaner:** derive it empirically — take the 99th-percentile observed `photons_eff` from a short pilot run and set the axis to that. This gives meaningful resolution in the region states actually occupy and makes the coverage metric honest. Also update the `n_achievable_bins` constraint to use the same realistic `max_total`.

Expected effect: "coverage" jumps from ~30% to a healthy number *and* the bins now resolve real structural diversity instead of empty photon-count space.

### B. Push ⟨O⟩ below 0.5725 — increase breeding depth (most promising physics lever)
0.5725 is the genuine floor for **depth 3** (8 leaves). The non-Gaussian resource that closes the gap to the GS comes from more breeding rounds, not higher cutoff/BF (best states are low-photon, well under budget). Run **depth 4** (16 leaves):

```bash
python run_mome.py --genotype 00B --depth 4 --scorer moment \
    --moment-cutoff 60 --moment-bf 1024 --moment-l-high 120 --moment-validate-every 250 \
    --target-alpha 2.7320508 --target-beta "1+1j"  ... (usual flags)
```
Watch whether ⟨O⟩ drops below 0.5725. If depth-4 states carry more photons/squeezing, bump `--moment-bf` (e.g. 4096) and possibly `--moment-cutoff` so the richer states stay valid — but confirm against fp/norm, don't raise blindly.

### C. Confirm 0.5725 is the depth-3 floor (cheap sanity check)
Multi-seed single-objective grind at depth 3 from independent random inits; if they all converge to ≈0.5725 from different basins, the floor is confirmed. (The current data already strongly suggests this.)

### D. Secondary: exploration within the reachable region
After the grid is rescaled (A), if reachable bins are still under-filled, tune the hybrid emitter — raise the exploration emitter's mutation / iso-line sigma or its share of the batch (`run_mome.py` ~L1058–1090). Do this *after* A, since most of the apparent under-coverage is the dead grid, not the emitter. Note `moment_fast=False` in this run, so the earlier "flat probability objective" concern does **not** apply here.

---

## Gotchas for the new session
- QDax MOME `results.pkl` (the `*_p64_i2500` runs after 15:35) **won't unpickle without qdax** — inspect those on the cluster, or analyze the single-objective `SimpleRepertoire` locally.
- Moment scorer **requires `JAX_ENABLE_X64=1`** (complex64 underflows the high-squeezing recurrence). `run_mome.py` sets this when `"moment"` is in argv.
- Re-validate any "new best" at high L before believing it: `python scripts/validate_moment_archive.py --group 00B_c30_a2p73_b1p41 --l-search 60 --l-high 120`.
- Don't seed a fresh run from an old Fock archive — it injects truncation artifacts.

## Suggested first action in the new convo
Implement **A** (rescale D3) — it directly resolves the "coverage looks broken" feeling and takes ~10 lines. Then launch a **depth-4** run (**B**) to test whether the 0.5725 floor moves. Those two answer both of your questions.
