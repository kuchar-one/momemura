#!/usr/bin/env python3
"""run_hanamura_all.py -- run the Hanamura two-step control-parameter
optimization on EVERY validated sub-Gaussian state (not just the pre-Hanamura
Pareto front), at SEVERAL reduction factors, so the *true* post-Hanamura Pareto
fronts can be reconstructed from all optimized states.

Why this exists
---------------
`run_hanamura_pareto.py` only optimizes the states that already sit on the
pre-Hanamura (<O>, P) front of their group.  But the Hanamura step preserves
neither the state nor its ranking: it reduces the detected photon number and
boosts the success probability by a *state-dependent* factor (measured x13 ...
x2.2e5 on the 84 front states).  A state that is dominated *before* the
optimization can therefore land on the front *after* it.  To find those states
we run the optimization over the whole validated set -- at several reduction
factors, since the factor trades <O>/fidelity against probability and squeezing
-- and rebuild the fronts from every optimized point.  See
`build_posthan_fronts.py`.

What it computes per (state, reduction-factor)  -- all self-consistent
--------------------------------------------------------------------------
The before-state (heavy: high-photon `reduced_herald`) is computed ONCE per
state and shared across reduction factors; only the cheap Hanamura two-step +
low-photon after-herald repeat per factor.

  * BEFORE : `reduced_herald` on the equivalent-GBS generator, detecting n0
             -> psi_before + P_before ; max necessary squeezing from
             `decompose_architecture`.  States whose reconstruction underflows
             (P=0 / empty psi / recomputed <O> far from the archive value) are
             written as explicit before_ok=False records, never as NaN rows.
  * rf=1 (damping-only) fast path: the heralded state is exactly psi_before,
             so no reconstruction or Gaussian alignment runs; exp_after :=
             exp_before_recomp (identical yardstick), fidelity := 1.  Zero-
             herald control modes are damped to vacuum analytically inside
             `optimize_damping` (t=1 is provably optimal for them), which
             reproduces the exact vacuum-projection absorption of those modes.
  * exp_after_cal = exp_hi + (exp_after - exp_before_recomp): the after-quality
             calibrated to the archive (L=200) scale; both recomputed values
             share the hcut-reconstruction estimator, so the common truncation
             bias cancels.  `build_posthan_fronts.py` ranks on this.
  * identical reduced patterns across factors (rf2/rf3 often coincide) are
             computed once and copied (`dedup_of`).
  * Hanamura two-step : `gbs_optimizer.optimize_gbs_architecture` (verify=False,
             uncapped squeezing so the recorded max_sq is the *necessary* one).
  * AFTER  : `gen_hanamura_data.reduced_full_state` (architecture rule; SAME
             signal frame + SAME estimator as BEFORE) -> psi_after + P_after ;
             max necessary squeezing from the optimized architecture.
  * <O>    : `moment_operator` on the normalized before/after states.  exp_after
             is the KEY new quantity: photon reduction changes the state, so <O>
             is NOT preserved and must be recomputed for a rigorous front.
  * fidelity : |<psi_before|psi_after>|^2 (both in-frame).

Wigner is OFF by default (dominates the per-state cost and is not needed for the
fronts, which live in (exp_after, P_after) and (exp_after, max_sq_after)).
Render Wigner for the final-front states with `run_hanamura_pareto.py` on the
per-factor CSVs that `build_posthan_fronts.py` emits.

Scale / performance
-------------------
2,395 valid states x 3 factors.  The heavy before-herald is amortized over the
factors, so the 3-factor sweep costs only modestly more than a 1-factor sweep
(~6-10 s/state).  Resumable (`--skip-existing`, keyed by state@factor),
time-boxed (`--max-seconds`), shardable (`--shard i --nshards N`, one JSONL per
shard).

  # smoke test (3 states x all factors)
  JAX_ENABLE_X64=1 python scripts/run_hanamura_all.py --limit 3
  # full run, 8-way fan-out (launch i=0..7)
  JAX_ENABLE_X64=1 python scripts/run_hanamura_all.py --nshards 8 --shard 0 &
  ...
  JAX_ENABLE_X64=1 python scripts/run_hanamura_all.py --nshards 8 --shard 7 &
  # then merge + build the fronts
  python scripts/build_posthan_fronts.py
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault("JAX_ENABLE_X64", "1")
# This script is CPU-bound: JAX is only used for the tiny genotype decode and the
# GKP operator build; all the heavy work (reduced_herald, optimize_gbs_architecture,
# the scipy Gaussian alignment / damping, thewalrus density matrices) is numpy /
# scipy / thewalrus on the CPU.  Left on the default GPU backend, JAX preallocates
# ~75% of each card's VRAM and then sits at 0% util -- wasting VRAM and preventing
# an 8-way fan-out from coexisting on the GPUs.  Pin to CPU (override by exporting
# JAX_PLATFORMS=cuda if you ever want the GPU).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]

from frontend.gbs_optimizer import stable_control_probability  # noqa: E402 (re-export)

# (alpha, beta) per target label, exactly as numpy_crosscheck_ng.py uses them.
TARGETS = {
    "a1p00_b1p00": (1.0, 1 + 0j),
    "a1p41_b1p41": (1.4142135623730951, 1 + 1j),
    "a2p73_b1p41": (2.7320508, 1 + 1j),
    "a0p00_b1p00": (0.0, 1 + 0j),
}


def to_numpy(d):
    """Recursively convert a decoded-params dict of jax arrays to numpy."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = to_numpy(v)
        else:
            try:
                out[k] = np.asarray(v)
            except Exception:
                out[k] = v
    return out


def _expval(psi, O):
    """Re <psi|O|psi> on the NORMALIZED state, O truncated to len(psi)."""
    psi = np.asarray(psi, complex).ravel()
    nrm = np.linalg.norm(psi)
    if nrm <= 0:
        return float("nan")
    psi = psi / nrm
    L = len(psi)
    return float(np.real(np.vdot(psi, O[:L, :L] @ psi)))


def _fidelity(psi_a, psi_b):
    a = np.asarray(psi_a, complex).ravel()
    b = np.asarray(psi_b, complex).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    L = min(len(a), len(b))
    return float(np.abs(np.vdot(a[:L] / na, b[:L] / nb)) ** 2)


# --------------------------------------------------------------------------- #
# Score the reduced state UP TO A GAUSSIAN UNITARY.
#
# The Hanamura reduction preserves the heralded output only up to a Gaussian
# unitary G (paper Eqs. 1/38; thesis 4chapter.tex l.315/335 defines the check as
# F(psi_n, G psi_n')).  <O> is not Gaussian-invariant, so the reduced state must
# be re-aligned by the optimal single-mode Gaussian before scoring -- exactly the
# freedom the MOME final-Gaussian gene already used to obtain <O>_before.  Here we
# minimize <O> over G = displace . rotate . squeeze (same parametrization as
# gbs_optimizer.align_states), WARM-STARTED from the nearest already-solved
# neighbour in objective space (neighbours share a similar optimal G, so a seed
# start + one identity guard replaces a cold multi-start -- ~2.5x fewer evals).
# --------------------------------------------------------------------------- #
# Per-cutoff eigendecompositions of the squeeze and displacement generators.
# K_sq = (a a - ad ad)/2 and K_d = (ad - a) are real antisymmetric, so
# i K = H is Hermitian and expm(s K) = V exp(-i s w) V^dagger with (w, V) from
# one eigh.  General phases enter by conjugation with the (diagonal) number
# rotation: S(r, phi) = R(phi) S(r, 0) R(-phi), D(|d| e^{i th}) =
# R(th) D(|d|) R(-th).  Each objective evaluation is then a handful of
# matvecs instead of two dense scipy expm calls (~100x faster).
_GAUSS_FACTORS: dict = {}


def _gauss_factors(c):
    if c not in _GAUSS_FACTORS:
        a = np.diag(np.sqrt(np.arange(1, c)), k=1); ad = a.T
        K_sq = 0.5 * (a @ a - ad @ ad)
        K_d = ad - a
        w_s, V_s = np.linalg.eigh(1j * K_sq)
        w_d, V_d = np.linalg.eigh(1j * K_d)
        _GAUSS_FACTORS[c] = (w_s, V_s, w_d, V_d, np.arange(c))
    return _GAUSS_FACTORS[c]


def _apply_gaussian_fast(params, psi):
    """G(dr, di, r, phi, varphi) |psi> via precomputed eigenfactors; identical
    to the expm implementation (unit-checked) but matvec-only."""
    psi = np.asarray(psi, complex).ravel()
    c = len(psi)
    w_s, V_s, w_d, V_d, nvec = _gauss_factors(c)
    dr, di, r, phi, varphi = params
    # squeeze: S(r, phi) = R(-phi) S(r,0) R(phi) with R(t)=diag(e^{-i n t})
    v = np.exp(-1j * nvec * phi) * psi
    v = V_s @ ((np.exp(-1j * r * w_s)) * (V_s.conj().T @ v))
    v = np.exp(1j * nvec * phi) * v
    # number rotation
    v = np.exp(1j * nvec * varphi) * v
    # displacement: D(d) = R(th) D(|d|) R(-th), d = dr + i di, th = arg d
    d = dr + 1j * di
    if abs(d) > 0:
        th = np.angle(d)
        v = np.exp(-1j * nvec * th) * v
        v = V_d @ ((np.exp(-1j * abs(d) * w_d)) * (V_d.conj().T @ v))
        v = np.exp(1j * nvec * th) * v
    return v / (np.linalg.norm(v) + 1e-300)


def min_exp_over_gaussian(psi, O, cut=48, seed=None):
    """Return (min <O> over single-mode Gaussian G, best params, aligned psi at
    full cutoff).  ``seed`` is a warm-start param vector from a neighbour."""
    from scipy.optimize import minimize
    psi = np.asarray(psi, complex).ravel()
    c = int(min(cut, len(psi)))
    p = psi[:c] / (np.linalg.norm(psi[:c]) + 1e-300)
    Oc = np.asarray(O)[:c, :c]

    def f(params):
        v = _apply_gaussian_fast(params, p)
        return float(np.real(np.vdot(v, Oc @ v)))

    if seed is not None:                       # warm: neighbour seed + 1 guard
        starts = [seed, (0.0, 0.0, 0.0, 0.0, 0.0)]
    else:                                      # cold: robust multi-start
        starts = [(0, 0, 0, 0, 0), (0, 0, 0.4, 0, 0), (0, 0, -0.4, 0, 0)]
    best = None
    for s in starts:
        res = minimize(f, np.array(s, float), method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-8, maxiter=1500))
        if best is None or res.fun < best.fun:
            best = res
    return float(best.fun), best.x, _apply_gaussian_fast(best.x, psi)


# --------------------------------------------------------------------------- #
# Stable success probability of a (damped) control Gaussian state at high photon
# number.  optimize_gbs_architecture computes P via a loop hafnian, guarded off
# above 16 detected photons -- so the DAMPING probability boost is unknown for
# our 20-photon champions (esp. reduction factor 1.0 = damping only).  This is
# the reduced_herald recipe applied to the control marginal: (1) condition the
# n=0 control modes analytically (Schur complement, vacuum covariance = I,
# hbar=2 -- identical to reduced_herald), (2) build the fired-mode density matrix
# with thewalrus' stable Hermite recurrence and read off the diagonal
# <n|rho|n> = P(detect n).  Cost is set by the fired-mode cutoff, not the loop
# hafnian, so it stays tractable where the hafnian does not.  HBAR=2.
# --------------------------------------------------------------------------- #
def load_valid_rows(verdicts_path):
    """Valid sub-Gaussian rows from verdicts.jsonl, deduped by 'key', sorted by
    run so the per-run repertoire cache hits."""
    import pandas as pd
    df = pd.read_json(verdicts_path, lines=True).drop_duplicates(subset="key")
    df = df[df.verdict == "valid"].copy()
    df = df.sort_values(["run", "cell"]).reset_index(drop=True)
    return df.to_dict("records")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts",
                    default=os.path.join(REPO, "recompute_ng", "verdicts.jsonl"))
    ap.add_argument("--out", default=os.path.join(REPO, "hanamura_all"),
                    help="output dir (one JSONL per shard lives here)")
    ap.add_argument("--reduction-factors", default="1.0,2.0,3.0,4.0",
                    help="comma list of reduction factors to sweep per state "
                         "(1.0 = damping-only: exactly output-preserving, the "
                         "lossless probability-boost front)")
    ap.add_argument("--align-cut", type=int, default=48,
                    help="Fock cutoff for the min-<O>-over-Gaussian alignment")
    ap.add_argument("--no-align", action="store_true",
                    help="skip the Gaussian re-alignment (record raw stale-frame "
                         "<O> only; for A/B comparison with the buggy scoring)")
    ap.add_argument("--herald-cap", type=int, default=60)
    ap.add_argument("--max-squeezing-db", type=float, default=0.0,
                    help="cap optimized squeezing (feasibility); 0 = uncapped so "
                         "the recorded max_sq is the NECESSARY squeezing")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap states (smoke runs)")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="skip state@factor ids already in this shard's JSONL")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--wigner", action="store_true",
                    help="also compute before/after Wigner negativity (slow)")
    ap.add_argument("--wigner-grid", type=int, default=51)
    ap.add_argument("--wigner-span", type=float, default=5.0)
    ap.add_argument("--wigner-ncut", type=int, default=60)
    args = ap.parse_args(argv)

    factors = [float(x) for x in str(args.reduction_factors).split(",") if x.strip()]

    import jax
    jax.config.update("jax_enable_x64", True)
    try:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jaxcache")
    except Exception:
        pass
    import jax.numpy as jnp

    from rescore_all_experiments import load_run_arrays
    from src.genotypes.genotypes import get_genotype_decoder
    from src.simulation.jax.moment_scorer import moment_operator
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend import gbs_optimizer as go
    from frontend.gbs_optimizer import (reduced_herald, decompose_architecture,
                                        stable_control_probability)
    from gen_hanamura_data import reduced_full_state

    os.makedirs(args.out, exist_ok=True)
    shard_jsonl = os.path.join(args.out, f"hanamura_all.shard{args.shard}.jsonl")

    done = set()
    if args.skip_existing and os.path.exists(shard_jsonl):
        with open(shard_jsonl) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["rec_id"])
                except Exception:
                    pass

    rows = load_valid_rows(args.verdicts)
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    print(f"[shard {args.shard}/{args.nshards}] {len(rows)} valid states x "
          f"{len(factors)} factors {factors} ({len(done)} recs already done)",
          flush=True)

    def _negvol(psi):
        from scipy.sparse.linalg import expm_multiply
        from scipy.sparse import diags
        psi = np.asarray(psi, complex).ravel()[: args.wigner_ncut]
        psi = psi / (np.linalg.norm(psi) + 1e-30)
        N = len(psi)
        a = diags(np.sqrt(np.arange(1, N)), 1)
        ad = diags(np.sqrt(np.arange(1, N)), -1)
        sign = (-1.0) ** np.arange(N)
        xs = np.linspace(-args.wigner_span, args.wigner_span, args.wigner_grid)
        dx = xs[1] - xs[0]
        neg = 0.0
        for x in xs:
            for pp in xs:
                z = (x + 1j * pp) / np.sqrt(2.0)
                phi = expm_multiply(-(z * ad - np.conj(z) * a), psi)
                w = (2.0 / np.pi) * np.real(np.sum(sign * np.abs(phi) ** 2))
                if w < 0:
                    neg += -w * dx * dx
        return float(neg)

    rep_cache = {}
    op_cache = {}
    # warm-start seeds for the Gaussian re-alignment, keyed by (target, factor):
    # each entry is a list of (exp_before, best_gaussian_params).  A new state
    # seeds from the entry with the nearest exp_before -- neighbours in objective
    # space share a similar optimal final Gaussian (user's observation).
    g_seed_cache: dict = {}

    def _nearest_seed(target, rf, exp_key):
        entries = g_seed_cache.get((target, rf))
        if not entries:
            return None
        return min(entries, key=lambda e: abs(e[0] - exp_key))[1]

    def _store_seed(target, rf, exp_key, params):
        lst = g_seed_cache.setdefault((target, rf), [])
        lst.append((float(exp_key), np.asarray(params, float)))
        if len(lst) > 64:                      # bound memory; keep most recent
            del lst[0]

    def load_run(run_rel):
        if run_rel not in rep_cache:
            base = os.path.join(REPO, run_rel)
            try:
                gens, _f, _d = load_run_arrays(os.path.join(base, "results.pkl"))
                cfg = json.load(open(os.path.join(base, "config.json")))
                gens = np.asarray(gens).reshape(-1, np.asarray(gens).shape[-1])
                rep_cache[run_rel] = (gens, cfg, None)
            except Exception as e:
                rep_cache[run_rel] = (None, None, e)
        gens, cfg, err = rep_cache[run_rel]
        if err is not None:
            raise err
        return gens, cfg

    def get_operator(target, L):
        key = (target, L)
        if key not in op_cache:
            a, b = TARGETS[target]
            op_cache[key] = np.asarray(moment_operator(int(L), a, b))
        return op_cache[key]

    fh = open(shard_jsonl, "a")
    t0 = time.time()
    n_states = n_recs = n_fail = 0
    for r in rows:
        if args.limit and n_states >= args.limit:
            break
        # which factors still need doing for this state?
        pending = [f for f in factors if f"{r['key']}@rf{f:g}" not in done]
        if not pending:
            continue
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"[time-box] stopping after {n_states} states / {n_recs} recs;"
                  f" rerun to continue", flush=True)
            break
        run_rel = r["run"]
        group = os.path.basename(os.path.dirname(run_rel))
        root = os.path.dirname(os.path.dirname(run_rel))
        run_tag = os.path.basename(run_rel).split("_")[0]  # e.g. 20260710-143857
        try:
            gens, cfg = load_run(run_rel)
            if int(cfg.get("modes") or 3) != 3:
                raise ValueError("modes!=3 (static/Hanamura path is 3-modes/leaf)")
            depth = int(cfg.get("depth") or r.get("depth") or 3)
            cutoff = int(cfg.get("cutoff") or 30)
            target = r["target"]
            g = gens[int(r["cell"])].astype(np.float32)

            dec = get_genotype_decoder(cfg.get("genotype"), depth=depth, config=cfg)
            params = to_numpy(dec.decode(jnp.asarray(g), cutoff))
            eq = compute_equivalent_gaussian(params)
            n0 = [int(x) for x in eq["pnr_outcomes"]]
            total0 = int(sum(n0))
            hcut = int(min(args.herald_cap, max(cutoff, 2 * total0 + 8)))
            O = get_operator(target, hcut)

            # ---- BEFORE (shared across all factors) ----
            psi_b, prob_b = reduced_herald(np.asarray(eq["cov"], float),
                                           np.asarray(eq["mu"], float),
                                           int(eq["signal_idx"]),
                                           [int(c) for c in eq["control_idx"]],
                                           n0, cutoff=hcut)
            psi_b = np.asarray(psi_b).ravel()
            exp_before_recomp = _expval(psi_b, O)
            try:
                arch_b = decompose_architecture(np.asarray(eq["cov"], float),
                                                np.asarray(eq["mu"], float))
                max_sq_before = float(arch_b.get("max_squeezing_db", float("nan")))
            except Exception:
                max_sq_before = float("nan")
            negvol_b = _negvol(psi_b) if args.wigner else None

            # ---- validity gate on the before-state --------------------------
            # A vanishing / non-finite reduced-herald probability or an empty
            # reconstructed state means the generator is numerically outside
            # what this pipeline can score (extreme squeezing).  Emit explicit
            # invalid records instead of NaN/0.0 rows (which would poison the
            # post-Hanamura fronts -- exp_after=0 would rank as a perfect GKP).
            before_ok = (np.isfinite(prob_b) and prob_b > 0.0
                         and np.isfinite(exp_before_recomp)
                         and np.linalg.norm(psi_b) > 1e-6
                         and abs(exp_before_recomp - float(r["exp_hi"])) < 0.25)
            if not before_ok:
                for rf in pending:
                    fh.write(json.dumps({
                        "rec_id": f"{r['key']}@rf{rf:g}", "key": r["key"],
                        "reduction_factor": float(rf), "target": target,
                        "group": group, "root": root, "run": run_rel,
                        "cell": int(r["cell"]), "after_ok": False,
                        "before_ok": False,
                        "fail_reason": "before_underflow_or_mismatch",
                        "exp_before": float(r["exp_hi"]),
                        "exp_before_recomp": (None if not np.isfinite(exp_before_recomp)
                                              else exp_before_recomp),
                        "prob_before": (None if not np.isfinite(prob_b)
                                        else float(prob_b)),
                        "prob_before_archive": float(10.0 ** float(r["logP"])),
                    }) + "\n")
                fh.flush()
                n_recs += len(pending)
                n_states += 1
                print(f"  [invalid-before] {group}/{run_tag}/{r['cell']}: "
                      f"P_b={prob_b:.1e} <O>_recomp={exp_before_recomp:.4f} "
                      f"(archive {float(r['exp_hi']):.4f})", flush=True)
                continue

            # consistent-yardstick before probability (same estimator family as
            # the damped/after values): stable control-marginal probability
            try:
                _C0, _b0 = go.extract_control(np.asarray(eq["cov"], float),
                                              np.asarray(eq["mu"], float),
                                              [int(c) for c in eq["control_idx"]])
                prob_b_stable = stable_control_probability(_C0, _b0, n0)
            except Exception:
                prob_b_stable = None

            maxsq = None if args.max_squeezing_db <= 0 else args.max_squeezing_db
            state_done = False
            n1_cache = {}       # tuple(n1) -> record dict (dedupe across factors)
            lam_seed = None     # warm-start damping from the previous factor
            for rf in pending:
                try:
                    # identical reduced patterns across factors produce identical
                    # generators -- copy the previous record instead of redoing
                    # the optimization (rf2/rf3 collide for roughly a third of
                    # the states).
                    n1_probe = [int(min(t, nn)) for t, nn in
                                zip(go.default_targets(n0, factor=rf), n0)]
                    if tuple(n1_probe) in n1_cache:
                        rec = dict(n1_cache[tuple(n1_probe)])
                        rec["rec_id"] = f"{r['key']}@rf{rf:g}"
                        rec["reduction_factor"] = float(rf)
                        rec["dedup_of"] = float(rec.get("dedup_of",
                                                        rec["reduction_factor"]))
                        fh.write(json.dumps({k: (None if isinstance(v, float)
                                                 and not np.isfinite(v) else v)
                                             for k, v in rec.items()}) + "\n")
                        fh.flush()
                        n_recs += 1
                        state_done = True
                        print(f"  {group}/{run_tag}/{r['cell']} rf{rf:g}: "
                              f"= rf{rec['dedup_of']:g} (same reduced pattern "
                              f"{n1_probe})", flush=True)
                        continue

                    han = go.optimize_gbs_architecture(
                        eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"],
                        n0, reduction_factor=rf, original_probability=float(prob_b),
                        max_squeezing_db=maxsq, verify=False, herald_cutoff=hcut,
                        damping_seed_lam=lam_seed)
                    n1 = [int(x) for x in han["n_after"]]
                    lam_seed = np.asarray(han.get("damping", {}).get("lam_fired",
                                                                     None))
                    arch_a = han.get("architecture") or {}
                    max_sq_after = float(arch_a.get("max_squeezing_db", float("nan")))

                    # stable damping-boosted P (the loop-hafnian path in
                    # optimize_gbs_architecture is guarded off above 16 detected
                    # photons -> None for our 20-photon champions, esp. rf=1).
                    p_opt = han.get("prob_after")
                    p_stable = float("nan")
                    try:
                        cm = han.get("control_moments") or {}
                        if cm.get("C2") is not None:
                            ps = stable_control_probability(cm["C2"], cm["beta2"], n1)
                            if ps is not None and np.isfinite(ps):
                                p_stable = float(ps)
                    except Exception as _e:
                        if os.environ.get("HAN_DEBUG"):
                            print(f"    stable-prob fail rf{rf:g}: {_e!r}", flush=True)
                    # auto cross-check where both exist AND are above the
                    # float-noise floor (below ~1e-13 both estimators run out
                    # of double precision and disagreements are meaningless)
                    if (p_opt is not None and np.isfinite(p_opt)
                            and p_opt > 1e-13 and np.isfinite(p_stable)
                            and p_stable > 1e-13
                            and abs(p_stable - p_opt) / p_opt > 1e-2):
                        print(f"    [WARN] stable P {p_stable:.3e} != hafnian P "
                              f"{p_opt:.3e} (rf{rf:g}, n1={n1})", flush=True)

                    exp_after_raw = exp_after_G = prob_a = float("nan")
                    fid_raw = fid_G = float("nan")
                    after_ok = False
                    negvol_a = None
                    try:
                        if rf == 1.0 or n1 == n0:
                            # damping-only: the heralded state is EXACTLY the
                            # before-state (damping preserves the output; only
                            # P changes).  Skip the heavy reconstruction and
                            # alignment; use the recomputed before-value so
                            # the before/after yardstick is identical.
                            psi_a, prob_a = psi_b, prob_b
                            exp_after_raw = exp_after_G = exp_before_recomp
                            fid_raw = fid_G = 1.0
                            after_ok = True
                        else:
                            psi_a, prob_a = reduced_full_state(eq, n0, n1, hcut)
                            psi_a = np.asarray(psi_a).ravel()
                            if (not np.isfinite(psi_a).all()
                                    or np.linalg.norm(psi_a) < 1e-6):
                                raise FloatingPointError(
                                    "after-state reconstruction underflowed")
                            # raw (stale-frame) numbers -- the buggy scoring,
                            # kept for A/B contrast
                            exp_after_raw = _expval(psi_a, O)
                            fid_raw = _fidelity(psi_a, psi_b)
                            if args.no_align:
                                exp_after_G = exp_after_raw
                                fid_G = fid_raw
                            else:
                                # frame-corrected: <O> minimized over a
                                # single-mode Gaussian (the reduction is only
                                # defined up to one), warm-started from the
                                # nearest neighbour
                                seed = _nearest_seed(target, rf, float(r["exp_hi"]))
                                exp_after_G, gpar, psi_a_al = min_exp_over_gaussian(
                                    psi_a, O, cut=args.align_cut, seed=seed)
                                _store_seed(target, rf, float(r["exp_hi"]), gpar)
                                fid_G = _fidelity(psi_a_al, psi_b)
                            after_ok = (np.isfinite(exp_after_G)
                                        and np.isfinite(fid_G)
                                        and exp_after_G > 0.05)
                        if args.wigner and after_ok:
                            negvol_a = _negvol(psi_a)
                    except Exception as ea:
                        if os.environ.get("HAN_DEBUG"):
                            print(f"    after-fail rf{rf:g}: {ea!r}", flush=True)

                    # calibrated after-quality on the ARCHIVE (L=200) scale:
                    # the recomputed before/after values share the estimator
                    # (hcut reconstruction), so their difference applied to the
                    # validated archive value cancels the common reconstruction
                    # bias.  At rf=1 this is exactly exp_hi by construction.
                    exp_after_cal = (float(r["exp_hi"])
                                     + (exp_after_G - exp_before_recomp)
                                     if (np.isfinite(exp_after_G)
                                         and np.isfinite(exp_before_recomp))
                                     else float("nan"))
                    # damping gain as a same-estimator ratio (stable path for
                    # both numerator and denominator; robust where absolute
                    # values underflow)
                    damp_gain = (p_stable / prob_b_stable
                                 if (np.isfinite(p_stable) and prob_b_stable
                                     and np.isfinite(prob_b_stable)
                                     and prob_b_stable > 0)
                                 else float("nan"))
                    rec = {
                        "rec_id": f"{r['key']}@rf{rf:g}", "key": r["key"],
                        "reduction_factor": float(rf),
                        "target": target, "group": group, "root": root,
                        "run": run_rel, "cell": int(r["cell"]),
                        "design": cfg.get("genotype"), "depth": depth,
                        "before_ok": True,
                        "exp_before": float(r["exp_hi"]),
                        "exp_before_recomp": exp_before_recomp,
                        # PRIMARY: <O>_after minimized over the final Gaussian
                        # (the reduction is only defined up to one).
                        "exp_after": exp_after_G,
                        # calibrated to the archive scale (USE THIS for fronts)
                        "exp_after_cal": exp_after_cal,
                        # diagnostic: the stale-frame value (old buggy scoring)
                        "exp_after_raw": exp_after_raw,
                        "prob_before": float(prob_b),
                        "prob_before_stable": (float(prob_b_stable)
                                               if prob_b_stable else None),
                        "damp_gain_stable": (float(damp_gain)
                                             if np.isfinite(damp_gain) else None),
                        # PRIMARY P after = damping-optimized success prob (Step 2
                        # is where the boost lives): exact loop-hafnian value when
                        # tractable, else the stable density-matrix value, else
                        # the undamped reduced-herald prob.
                        "prob_after": (float(p_opt)
                                       if p_opt is not None and np.isfinite(p_opt)
                                       else (p_stable if np.isfinite(p_stable)
                                             else (float(prob_a)
                                                   if np.isfinite(prob_a) else None))),
                        "prob_after_stable": (p_stable if np.isfinite(p_stable)
                                              else None),
                        "prob_after_herald": (float(prob_a)
                                              if np.isfinite(prob_a) else None),
                        "prob_before_archive": float(10.0 ** float(r["logP"])),
                        "max_sq_before": max_sq_before,
                        "max_sq_after": max_sq_after,
                        "Nc_before": total0, "Nc_after": int(sum(n1)),
                        "n0": n0, "n1": n1,
                        # fidelity up to a Gaussian unitary (thesis F metric);
                        # fid_raw is the stale-frame overlap for contrast
                        "fidelity_after_before": fid_G,
                        "fidelity_raw": fid_raw,
                        "k_eff_before": int(sum(1 for x in n0 if x >= 1)),
                        "k_eff_after": int(sum(1 for x in n1 if x >= 1)),
                        "herald_cutoff": hcut, "after_ok": after_ok,
                        "negvol_before": negvol_b, "negvol_after": negvol_a,
                    }
                    n1_cache[tuple(n1)] = dict(rec, dedup_of=float(rf))
                    fh.write(json.dumps({k: (None if isinstance(v, float)
                                             and not np.isfinite(v) else v)
                                         for k, v in rec.items()}) + "\n")
                    fh.flush()
                    n_recs += 1
                    state_done = True
                    p_after = rec["prob_after"]
                    gain = (p_after / prob_b) if (p_after and prob_b > 0) else float("nan")
                    print(f"  {group}/{run_tag}/{r['cell']} rf{rf:g}: "
                          f"O {r['exp_hi']:.4f}->{exp_after_cal:.4f}"
                          f"(raw {exp_after_raw:.4f})  Nc {total0}->{sum(n1)}  "
                          f"P {prob_b:.1e}->{(p_after or float('nan')):.1e} "
                          f"(x{gain:.1f}, damp x{damp_gain:.1f})  "
                          f"sqdB {max_sq_before:.1f}->{max_sq_after:.1f}"
                          f"  fidG {fid_G:.3f}"
                          f"{'' if after_ok else '  [AFTER-INVALID]'}", flush=True)
                except Exception as ef:
                    n_fail += 1
                    print(f"  [skip] {group}/{run_tag}/{r['cell']} rf{rf:g}: {ef!r}",
                          flush=True)
                    if os.environ.get("HAN_DEBUG"):
                        import traceback
                        traceback.print_exc()
            if state_done:
                n_states += 1
        except Exception as e:
            n_fail += 1
            print(f"  [skip-state] {group}/{r.get('cell')}: {e!r}", flush=True)
            if os.environ.get("HAN_DEBUG"):
                import traceback
                traceback.print_exc()
    fh.close()
    print(f"\n[shard {args.shard}] DONE: {n_states} states / {n_recs} recs new, "
          f"{n_fail} skipped -> {shard_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
