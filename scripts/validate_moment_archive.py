#!/usr/bin/env python3
"""validate_moment_archive.py -- exact re-validation of a moment-scored archive.

The optimizer can SEARCH fast with a low moment cutoff L and small fired-box
buffer BF (--moment-cutoff / --moment-bf).  A low L truncates the final signal
state, so a few archive points can be low-L artifacts.  This script re-scores
every archive genotype with the SAME exact moment scorer at a high L/BF and
flags points whose <O> shifts (or whose low-L herald wasn't normalised) -- the
moment analogue of the old dual-cutoff sweep, but both ends are exact.

Run on the cluster (GPU, x64):
    JAX_ENABLE_X64=1 python scripts/validate_moment_archive.py \
        --group 00B_c30_a2p73_b1p41 --l-search 50 --l-high 120 --write
"""
import os, sys, glob, json, argparse, pickle
os.environ["JAX_ENABLE_X64"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--group", default="00B_c30_a2p73_b1p41")
    ap.add_argument("--path", default=None, help="specific results.pkl (overrides group: newest run)")
    ap.add_argument("--l-search", type=int, default=50)
    ap.add_argument("--bf-search", type=int, default=1024)
    ap.add_argument("--l-high", type=int, default=120)
    ap.add_argument("--bf-high", type=int, default=8192)
    ap.add_argument("--tol", type=float, default=0.02, help="max |Δ<O>| before flagging an artifact")
    ap.add_argument("--write", action="store_true", help="write an exact (L-high) rescored repertoire next to the input")
    args = ap.parse_args()

    import jax, jax.numpy as jnp
    import pareto_report as pr
    from src.genotypes.genotypes import get_genotype_decoder
    from src.utils.result_manager import SimpleRepertoire
    from src.simulation.jax.moment_scorer import (
        moment_operator, jax_equivalent_gaussian_static, jax_reduced_herald_static)

    if args.path:
        pkl = args.path
    else:
        roots = ([args.root] if args.root else
                 [os.path.join(REPO, "experiments"), os.path.join(REPO, "output", "experiments")])
        runs = []
        for r in roots:
            runs = sorted(glob.glob(os.path.join(r, args.group, "*", "results.pkl")))
            if runs:
                break
        pkl = runs[-1]
    cfg = json.load(open(os.path.join(os.path.dirname(pkl), "config.json")))
    print(f"validating {pkl}")
    rep = pr.load_repertoire(pkl)
    fit = np.asarray(rep.fitnesses, float).reshape(-1, np.asarray(rep.fitnesses).shape[-1])
    des = np.asarray(rep.descriptors, float).reshape(-1, np.asarray(rep.descriptors).shape[-1])
    gen = np.asarray(rep.genotypes, float).reshape(-1, np.asarray(rep.genotypes).shape[-1])
    valid = np.where(np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9))[0]

    base = int(cfg.get("cutoff") or 30)
    a, b = cfg.get("target_alpha"), cfg.get("target_beta")
    O_lo = np.asarray(moment_operator(args.l_search, a, b))
    O_hi = np.asarray(moment_operator(args.l_high, a, b))
    depth = int(cfg.get("depth") or 3)
    maxf = int(cfg.get("moment_maxf") or 8)
    dec = get_genotype_decoder(cfg.get("genotype"), depth=depth, config=cfg)

    @jax.jit
    def chain(g, L, BF):
        p = dec.decode(g, base)
        cs, ms, ep, _ = jax_equivalent_gaussian_static(p, depth)
        return jax_reduced_herald_static(cs, ms, ep, L, BF, depth, maxf)

    new_fit = fit.copy()
    n_art = 0
    worst = []
    for k in valid:
        g = jnp.asarray(gen[k].astype(np.float32))
        psi_lo = np.asarray(chain(g, args.l_search, args.bf_search)[0])
        psi_hi = np.asarray(chain(g, args.l_high, args.bf_high)[0])
        norm_lo = float(np.sum(np.abs(psi_lo) ** 2))
        exp_lo = float(np.real(np.vdot(psi_lo, O_lo[:len(psi_lo), :len(psi_lo)] @ psi_lo)))
        exp_hi = float(np.real(np.vdot(psi_hi, O_hi[:len(psi_hi), :len(psi_hi)] @ psi_hi)))
        d = abs(exp_hi - exp_lo)
        new_fit[k, 0] = -exp_hi                      # archive now holds the exact value
        if d > args.tol or norm_lo < 0.99:
            n_art += 1
            new_fit[k, 0] = -np.inf                   # drop the artifact
            worst.append((k, exp_lo, exp_hi, d, norm_lo))

    worst.sort(key=lambda x: -x[3])
    print(f"\n{'idx':>6} {'exp_search':>10} {'exp_high':>9} {'dexp':>8} {'norm_lo':>8}")
    for k, el, eh, d, nl in worst[:20]:
        print(f"{k:>6} {el:10.4f} {eh:9.4f} {d:8.2e} {nl:8.4f}")
    best_hi = float(np.min(-new_fit[np.isfinite(new_fit[:, 0]), 0])) if np.any(np.isfinite(new_fit[:, 0])) else float('nan')
    print(f"\nchecked {len(valid)} | low-L artifacts (|Δ<O>|>{args.tol} or norm<0.99): {n_art} "
          f"({100*n_art/max(len(valid),1):.1f}%)")
    print(f"best exact <O> after dropping artifacts: {best_hi:.4f}")

    if args.write:
        out = os.path.join(os.path.dirname(pkl), "results_validated.pkl")
        new_rep = SimpleRepertoire(gen.astype(np.float32), new_fit, des)
        with open(out, "wb") as f:
            pickle.dump({"repertoire": new_rep, "config": cfg}, f)
        print(f"wrote exact-rescored repertoire -> {out}")


if __name__ == "__main__":
    main()
