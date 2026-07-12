#!/usr/bin/env python3
"""validate_ng_winners.py -- HANDOFF_ng_results_validation §3 independent
high-L revalidation of the ng-campaign sub-Gaussian winners (CPU-friendly,
resumable in short time-boxed invocations).

What it does
------------
1. Reads the per-run scan state produced by ``ng_archive_stats.py --state``
   (JSONL; one record per campaign run with its sub-Gaussian rows).
2. Dedupes sub-Gaussian rows per target by (exp, logP) signature -- the
   seeding machinery copies identical solutions between runs -- keeping one
   provenance (run dir + flat cell index) per physical state.  Rows from
   phase C (``--mode single`` Adam polish) are ALWAYS kept for rescoring even
   when a same-signature A/B row exists, because C archives were never swept
   (stored <O> is L=50) -- the audit found the a1b1 champion drifts
   0.3754 -> 0.3917 under the exact rescore.
3. Re-scores every unique state with the exact moment pipeline at
   L=``--l-high`` (default 200) and BF=``--bf-high`` (default 8192), computes
   the exact leaf herald probability, top-decile tail mass, effective
   (coupled) photons n_eff at coupling_eps=0.05, and classifies:

     artifact      |<O>_hi - <O>_stored| > --tol  (0.02)  OR tail > --tail-tol
     decoupled     n_eff < 0.5 (descriptor photon axis inflated by detectors
                   not coupled to the signal -- the 2026-06-10 exploit)
     valid         otherwise; vs_gaussian = <O>_hi - G < 0 confirms subG

4. Emits (per target): a JSONL of verdicts (the resumable state), and -- once
   everything is scored -- a validated-front CSV in the exact schema of
   ``recompute/pareto_fronts/<group>.csv`` so that
   ``scripts/run_hanamura_pareto.py --pareto-dir <out>/pareto_fronts`` runs
   unchanged on the validated states.

Usage (repeat until it prints ALL DONE; safe to kill any time):
  JAX_ENABLE_X64=1 python scripts/validate_ng_winners.py \
      --scan-state /tmp/ng_scan_state.jsonl --out recompute_ng \
      --max-seconds 35 [--targets a1p00_b1p00 a1p41_b1p41 a2p73_b1p41]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]

PHASE_PREF = {"B": 0, "A": 1, "C": 2}          # provenance preference (stored-value trust)
BATCH = 8                                       # fixed vmap batch (one compile per bucket)

TARGETS = {
    "a1p00_b1p00": (1.0, complex(1, 0)),
    "a1p41_b1p41": (1.4142135623730951, complex(1, 1)),
    "a2p73_b1p41": (2.7320508, complex(1, 1)),
    "a0p00_b1p00": (0.0, complex(1, 0)),
}
G_REF = {"a1p00_b1p00": (0.666667, 0.710782), "a1p41_b1p41": (0.959560, 0.997924),
         "a2p73_b1p41": (1.089316, 1.124977), "a0p00_b1p00": (0.666667, 0.710779)}


def build_queue(scan_state, targets):
    """unique states: key -> dict(run, cell, exp_stored, logp_stored, des,
    target, genotype, depth, phase)."""
    per_t = defaultdict(dict)
    with open(scan_state) as fh:
        for line in fh:
            r = json.loads(line)
            tgt = r.get("target")
            if r.get("error") or tgt not in targets:
                continue
            phase = r["phase"].split("_")[0]
            for idx, e, lp, des in r.get("_sub_sorted", []):
                sig = (round(e, 7), round(lp, 5))
                cand = dict(run=r["run"], cell=int(idx), exp_stored=float(e),
                            logp_stored=float(lp), des=des, target=tgt,
                            genotype=r["genotype"], depth=int(r["depth"][1:]),
                            phase=phase, cycle=r["cycle"])
                old = per_t[tgt].get(sig)
                if old is None or PHASE_PREF[phase] < PHASE_PREF[old["phase"]]:
                    per_t[tgt][sig] = cand
    # flatten with stable ordering: bucket-major (genotype, depth), then exp asc
    queue = []
    for tgt in sorted(per_t):
        cands = list(per_t[tgt].values())
        # bucket-major, then run-major (pickle-cache friendly), then exp
        cands.sort(key=lambda c: (c["genotype"], c["depth"], c["run"],
                                  c["exp_stored"]))
        queue += cands
    return queue


def cand_key(c):
    return f"{c['target']}|{c['run']}|{c['cell']}|{c['exp_stored']:.7f}"


_GEN_CACHE = {}


def genotype_row(run, cell):
    from rescore_all_experiments import load_run_arrays
    if run not in _GEN_CACHE:
        if len(_GEN_CACHE) > 4:
            _GEN_CACHE.clear()
        gen, fit, des = load_run_arrays(os.path.join(REPO, run, "results.pkl"))
        _GEN_CACHE[run] = gen
    return _GEN_CACHE[run][cell]


def n_eff_coupled(eng, g, eps=0.05):
    """Effective coupled photons of the fired detectors (numpy; mirrors
    ``_effective_photons_static``: control slot s <-> mode 1+s, eff_pnr[s]=0
    means unused; coupling of a fired control = Frobenius norm of its cov
    rows against (signal + all other controls), both quadratures."""
    cov, mu, eff_pnr = eng.equivalent_gaussian(np.asarray(g, np.float64))
    cov = np.asarray(cov, float)
    N = cov.shape[0] // 2
    eff_pnr = np.asarray(eff_pnr, float).ravel()
    n_eff = 0.0
    max_eff = 0.0
    for c in range(1, N):
        n0 = eff_pnr[c - 1] if c - 1 < len(eff_pnr) else 0.0
        if n0 < 1:
            continue
        others = [0] + [c2 for c2 in range(1, N) if c2 != c]
        oi = [i for o in others for i in (o, o + N)]
        cpl = float(np.linalg.norm(cov[np.ix_([c, c + N], oi)]))
        w = cpl ** 2 / (cpl ** 2 + eps ** 2)
        ne = n0 * w
        n_eff += ne
        max_eff = max(max_eff, ne)
    return n_eff, max_eff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-state", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "recompute_ng"))
    ap.add_argument("--targets", nargs="+",
                    default=["a1p00_b1p00", "a1p41_b1p41", "a2p73_b1p41"])
    ap.add_argument("--l-high", type=int, default=200)
    ap.add_argument("--bf-high", type=int, default=8192)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--tail-tol", type=float, default=0.02)
    ap.add_argument("--neff-min", type=float, default=0.5)
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    verd_path = os.path.join(args.out, "verdicts.jsonl")
    done = set()
    if os.path.exists(verd_path):
        with open(verd_path) as fh:
            for line in fh:
                done.add(json.loads(line)["key"])

    queue = build_queue(args.scan_state, set(args.targets))
    pending = [c for c in queue if cand_key(c) not in done]
    print(f"queue {len(queue)} unique states; {len(done)} done; "
          f"{len(pending)} pending", flush=True)

    if pending:
        import jax
        jax.config.update("jax_compilation_cache_dir", "/tmp/jaxcache")
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        from rescore_all_experiments import get_engine
        from src.simulation.jax.moment_scorer import moment_operator

        O_cache = {}

        def O_for(tgt, L):
            if (tgt, L) not in O_cache:
                a, b = TARGETS[tgt]
                O_cache[(tgt, L)] = np.asarray(moment_operator(L, a, b))
            return O_cache[(tgt, L)]

        cfg_cache = {}

        def cfg_for(run):
            if run not in cfg_cache:
                cfg_cache[run] = json.load(
                    open(os.path.join(REPO, run, "config.json")))
            return cfg_cache[run]

        t0 = time.time()
        fh = open(verd_path, "a")
        n_new = 0
        i = 0
        while i < len(pending):
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                print(f"[time-box] scored {n_new}; "
                      f"{len(pending)-i} still pending", flush=True)
                break
            if args.limit and n_new >= args.limit:
                break
            # batch: consecutive candidates in the same engine bucket
            c0 = pending[i]
            cfg = cfg_for(c0["run"])
            batch = [c0]
            j = i + 1
            while (j < len(pending) and len(batch) < BATCH and
                   pending[j]["genotype"] == c0["genotype"] and
                   pending[j]["depth"] == c0["depth"] and
                   pending[j]["target"] == c0["target"]):
                batch.append(pending[j])
                j += 1
            gs = [genotype_row(c["run"], c["cell"]) for c in batch]
            G = np.zeros((BATCH, len(gs[0])), dtype=np.float64)
            for k, g in enumerate(gs):
                G[k] = g
            for k in range(len(gs), BATCH):
                G[k] = gs[0]                    # pad with copies
            eng = get_engine(c0["genotype"], c0["depth"],
                             int(cfg.get("moment_maxf", 10)), cfg)
            psi, prob, eff, na = eng.score(G, args.l_high, args.bf_high)
            O = O_for(c0["target"], args.l_high)
            ntail = max(1, args.l_high // 10)
            G_an, GN = G_REF[c0["target"]]
            for k, c in enumerate(batch):
                p = psi[k]
                e_hi = float(np.real(np.vdot(p, O[: len(p), : len(p)] @ p)))
                tail = float(np.sum(np.abs(p[args.l_high - ntail:]) ** 2))
                logp = float(np.log10(max(eng.leaf_prob(
                    G[k].astype(np.float32), args.l_high), 1e-45)))
                neff, maxeff = n_eff_coupled(eng, G[k])
                drift = e_hi - c["exp_stored"]
                verdict = "valid"
                if abs(drift) > args.tol or tail > args.tail_tol:
                    verdict = "artifact"
                elif neff < args.neff_min:
                    verdict = "decoupled"
                elif e_hi >= min(G_an, GN):
                    verdict = "not_subG"        # honest but not sub-Gaussian
                rec = dict(key=cand_key(c), **{k2: v for k2, v in c.items()
                                               if k2 != "des"},
                           des=c["des"], exp_hi=e_hi, drift=drift, tail=tail,
                           logP=logp, n_eff=neff, max_eff=maxeff,
                           n_active=float(na[k]),
                           vs_gaussian=e_hi - G_an, vs_gn=e_hi - GN,
                           verdict=verdict)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_new += 1
            i = j
        fh.close()
        if i < len(pending):
            return 1

    # ---------------- aggregate: validated fronts + summary ----------------
    import pandas as pd
    recs = [json.loads(l) for l in open(verd_path)]
    df = pd.DataFrame(recs).drop_duplicates(subset="key")
    print(f"\n=== verdicts ({len(df)} states) ===")
    print(df.groupby(["target", "verdict"]).size().to_string())
    pf_dir = os.path.join(args.out, "pareto_fronts")
    os.makedirs(pf_dir, exist_ok=True)
    for old in os.listdir(pf_dir):             # idempotent re-aggregation
        if old.endswith(".csv"):
            os.remove(os.path.join(pf_dir, old))
    for tgt, sub in df.groupby("target"):
        v = sub[sub["verdict"] == "valid"].copy()
        if v.empty:
            continue
        a, b = TARGETS[tgt]
        des = np.vstack(v["des"].values)
        v["total_photons"] = des[:, 2]
        v["max_pnr"] = des[:, 1]
        v["fired_modes"] = np.nan
        v["prob"] = 10.0 ** v["logP"]
        v["wigner_negvol"] = np.nan
        v["vs_gs"] = np.nan
        # non-dominated front over (exp_hi min, logP max)
        objs = np.column_stack([v["exp_hi"].values, -v["logP"].values])
        keep = np.ones(len(v), bool)
        for ii in range(len(v)):
            dom = np.all(objs <= objs[ii], axis=1) & np.any(objs < objs[ii], axis=1)
            keep[ii] = not np.any(dom)
        fr = v[keep].sort_values("exp_hi").copy()
        # schema of recompute/pareto_fronts/<group>.csv (run_hanamura_pareto input)
        fr["root"] = "output/experiments"
        fr["group"] = fr["run"].str.split("/").str[-2]
        fr["run"] = fr["run"].str.split("/").str[-1]
        fr["cell_idx"] = fr["cell"]
        fr["target_alpha"] = str(a)
        fr["target_beta"] = str(b)
        cols = ["root", "group", "run", "cell_idx", "exp_hi", "logP", "prob",
                "total_photons", "fired_modes", "max_pnr", "vs_gaussian",
                "vs_gs", "wigner_negvol", "target_alpha", "target_beta"]
        # one CSV per (target, experiment-group) so hanamura groups stay clean
        for grp, gsub in fr.groupby("group"):
            gsub[cols].to_csv(os.path.join(pf_dir, f"{grp}.csv"), index=False,
                              mode="a" if os.path.exists(
                                  os.path.join(pf_dir, f"{grp}.csv")) else "w",
                              header=not os.path.exists(
                                  os.path.join(pf_dir, f"{grp}.csv")))
        print(f"{tgt}: validated front {keep.sum()} states "
              f"(of {len(v)} valid, {len(sub)} scored)")
    df.to_csv(os.path.join(args.out, "verdicts.csv"), index=False)
    print("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
