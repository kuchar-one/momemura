#!/usr/bin/env python3
"""numpy_crosscheck_ng.py -- HANDOFF §3.2: independent numpy reconstruction of
the top validated ng-campaign winners.

For the top-N valid states per target (by exp_hi from
``recompute_ng/verdicts.jsonl``): decode the genotype, build the equivalent
Gaussian via the NUMPY reference path (``frontend.gaussian_decomposition.
compute_equivalent_gaussian``), herald via ``frontend.gbs_optimizer.
reduced_herald`` (thewalrus state_vector machinery -- fully independent of the
JAX moment scorer), and compare <O> and the state vector against the JAX
values recorded by validate_ng_winners.py.  Expected agreement ~1e-8
(tests/test_moment_scorer.py pattern).

Resumable: verdict rows are appended to ``recompute_ng/numpy_crosscheck.jsonl``.

  JAX_ENABLE_X64=1 python scripts/numpy_crosscheck_ng.py [--top 10] [--l 200]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdicts", default=os.path.join(REPO, "recompute_ng",
                                                       "verdicts.jsonl"))
    ap.add_argument("--out", default=os.path.join(REPO, "recompute_ng",
                                                  "numpy_crosscheck.jsonl"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--l", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=0)
    args = ap.parse_args(argv)

    import pandas as pd
    df = pd.read_json(args.verdicts, lines=True).drop_duplicates(subset="key")
    v = df[df.verdict == "valid"]
    picks = (v.sort_values("exp_hi").groupby("target", as_index=False)
             .head(args.top))

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {json.loads(l)["key"] for l in fh}

    import jax
    jax.config.update("jax_compilation_cache_dir", "/tmp/jaxcache")
    import jax.numpy as jnp
    from rescore_all_experiments import load_run_arrays
    from src.genotypes.genotypes import get_genotype_decoder
    from src.simulation.jax.moment_scorer import (
        moment_operator, jax_equivalent_gaussian_static,
        jax_reduced_herald_static)
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend.gbs_optimizer import reduced_herald

    TARGETS = {"a1p00_b1p00": (1.0, 1 + 0j), "a1p41_b1p41":
               (1.4142135623730951, 1 + 1j), "a2p73_b1p41": (2.7320508, 1 + 1j),
               "a0p00_b1p00": (0.0, 1 + 0j)}

    t0 = time.time()
    fh = open(args.out, "a")
    n = 0
    for _, r in picks.iterrows():
        if r["key"] in done:
            continue
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"[time-box] {n} new; rerun to continue", flush=True)
            fh.close()
            return 1
        cfg = json.load(open(os.path.join(REPO, r["run"], "config.json")))
        gen, _, _ = load_run_arrays(os.path.join(REPO, r["run"], "results.pkl"))
        g = gen[int(r["cell"])].astype(np.float64)
        dec = get_genotype_decoder(r["genotype"], depth=int(r["depth"]),
                                   config=cfg)
        params = dec.decode(jnp.asarray(g), int(cfg.get("cutoff") or 30))
        pnp = {k: (np.asarray(vv) if hasattr(vv, "shape") else
                   {k2: np.asarray(v2) for k2, v2 in vv.items()}
                   if isinstance(vv, dict) else vv)
               for k, vv in params.items()}
        eq = compute_equivalent_gaussian(pnp)
        n0 = [int(x) for x in eq["pnr_outcomes"]]
        psi_np, prob_np = reduced_herald(np.asarray(eq["cov"], float),
                                         np.asarray(eq["mu"], float),
                                         int(eq["signal_idx"]),
                                         [int(x) for x in eq["control_idx"]],
                                         n0, cutoff=args.l)
        psi_np = np.asarray(psi_np).ravel()
        psi_np = psi_np / np.linalg.norm(psi_np)
        a, b = TARGETS[r["target"]]
        O = np.asarray(moment_operator(args.l, a, b))
        e_np = float(np.real(np.vdot(psi_np, O[: len(psi_np), : len(psi_np)]
                                     @ psi_np)))
        # JAX state for vector-level comparison
        cs, ms, ep, _ = jax_equivalent_gaussian_static(params, int(r["depth"]))
        psi_j, _ = jax_reduced_herald_static(cs, ms, ep, args.l, 8192,
                                             int(r["depth"]),
                                             int(cfg.get("moment_maxf", 10)))
        psi_j = np.asarray(psi_j).ravel()
        psi_j = psi_j / np.linalg.norm(psi_j)
        L = min(len(psi_np), len(psi_j))
        fid = float(np.abs(np.vdot(psi_np[:L], psi_j[:L])) ** 2)
        rec = dict(key=r["key"], target=r["target"], run=r["run"],
                   cell=int(r["cell"]), exp_hi_jax=float(r["exp_hi"]),
                   exp_numpy=e_np, diff=e_np - float(r["exp_hi"]),
                   fidelity_np_vs_jax=fid, prob_np=float(prob_np),
                   fired=[x for x in n0 if x >= 1])
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        n += 1
        print(f"{r['target']} cell {r['cell']}: jax {r['exp_hi']:.8f} "
              f"numpy {e_np:.8f} diff {e_np - r['exp_hi']:+.2e} fid {fid:.10f}",
              flush=True)
    fh.close()
    print("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
