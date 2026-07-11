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
             `decompose_architecture`.
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
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]

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
    ap.add_argument("--reduction-factors", default="2.0,3.0,4.0",
                    help="comma list of reduction factors to sweep per state")
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
    from frontend.gbs_optimizer import reduced_herald, decompose_architecture
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

            maxsq = None if args.max_squeezing_db <= 0 else args.max_squeezing_db
            state_done = False
            for rf in pending:
                try:
                    han = go.optimize_gbs_architecture(
                        eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"],
                        n0, reduction_factor=rf, original_probability=float(prob_b),
                        max_squeezing_db=maxsq, verify=False, herald_cutoff=hcut)
                    n1 = [int(x) for x in han["n_after"]]
                    arch_a = han.get("architecture") or {}
                    max_sq_after = float(arch_a.get("max_squeezing_db", float("nan")))

                    exp_after = prob_a = fid = float("nan")
                    after_ok = False
                    negvol_a = None
                    try:
                        psi_a, prob_a = reduced_full_state(eq, n0, n1, hcut)
                        psi_a = np.asarray(psi_a).ravel()
                        exp_after = _expval(psi_a, O)
                        fid = _fidelity(psi_a, psi_b)
                        after_ok = True
                        if args.wigner:
                            negvol_a = _negvol(psi_a)
                    except Exception as ea:
                        if os.environ.get("HAN_DEBUG"):
                            print(f"    after-fail rf{rf:g}: {ea!r}", flush=True)

                    rec = {
                        "rec_id": f"{r['key']}@rf{rf:g}", "key": r["key"],
                        "reduction_factor": float(rf),
                        "target": target, "group": group, "root": root,
                        "run": run_rel, "cell": int(r["cell"]),
                        "design": cfg.get("genotype"), "depth": depth,
                        "exp_before": float(r["exp_hi"]),
                        "exp_before_recomp": exp_before_recomp,
                        "exp_after": exp_after,
                        "prob_before": float(prob_b),
                        "prob_after": (float(prob_a) if np.isfinite(prob_a) else None),
                        "prob_before_archive": float(10.0 ** float(r["logP"])),
                        "prob_after_opt": (float(han["prob_after"])
                                           if han.get("prob_after") is not None
                                           and np.isfinite(han["prob_after"])
                                           else None),
                        "max_sq_before": max_sq_before,
                        "max_sq_after": max_sq_after,
                        "Nc_before": total0, "Nc_after": int(sum(n1)),
                        "n0": n0, "n1": n1,
                        "fidelity_after_before": fid,
                        "k_eff_before": int(sum(1 for x in n0 if x >= 1)),
                        "k_eff_after": int(sum(1 for x in n1 if x >= 1)),
                        "herald_cutoff": hcut, "after_ok": after_ok,
                        "negvol_before": negvol_b, "negvol_after": negvol_a,
                    }
                    fh.write(json.dumps({k: (None if isinstance(v, float)
                                             and not np.isfinite(v) else v)
                                         for k, v in rec.items()}) + "\n")
                    fh.flush()
                    n_recs += 1
                    state_done = True
                    gain = (prob_a / prob_b) if (np.isfinite(prob_a)
                                                 and prob_b > 0) else float("nan")
                    print(f"  {group}/{r['cell']} rf{rf:g}: "
                          f"O {r['exp_hi']:.4f}->{exp_after:.4f}  "
                          f"Nc {total0}->{sum(n1)}  P {prob_b:.1e}->{prob_a:.1e} "
                          f"(x{gain:.1f})  sqdB {max_sq_before:.1f}->{max_sq_after:.1f}"
                          f"  fid {fid:.3f}", flush=True)
                except Exception as ef:
                    n_fail += 1
                    print(f"  [skip] {group}/{r['cell']} rf{rf:g}: {ef!r}", flush=True)
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
