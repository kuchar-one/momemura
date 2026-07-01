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


def _as_complex(v, name):
    """Robustly parse a config/CLI target into a Python complex, or None if
    absent. Handles bare numbers (the QDax configs store target_alpha as a
    number), parenthesised reprs like '(1+1j)', 'i'->'j', and stray spaces."""
    if v is None:
        return None
    if isinstance(v, (int, float, complex)):
        return complex(v)
    s = str(v).strip().replace(" ", "").replace("i", "j")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    try:
        return complex(s)
    except ValueError:
        raise SystemExit(f"could not parse {name}={v!r} as a complex number")


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
    ap.add_argument("--target-alpha", default=None,
                    help="override target alpha (QDax/MOME configs may lack it)")
    ap.add_argument("--target-beta", default=None,
                    help="override target beta, e.g. '1+1j' (QDax/MOME configs omit target_beta)")
    args = ap.parse_args()

    import jax, jax.numpy as jnp
    import pareto_report as pr
    from src.genotypes.genotypes import get_genotype_decoder
    from src.utils.result_manager import SimpleRepertoire
    from src.simulation.jax.moment_scorer import (
        moment_operator, jax_equivalent_gaussian_static, jax_reduced_herald_static,
        _leaf_prob_product_static)

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
    # Resolve the target (alpha,beta): CLI override > config. QDax/MOME configs
    # store target_alpha as a bare number and OMIT target_beta entirely, so fall
    # back to an explicit --target-beta and fail loudly (not with a cryptic
    # complex() error) if neither source has it.
    a = _as_complex(args.target_alpha if args.target_alpha is not None
                    else cfg.get("target_alpha"), "target_alpha")
    b = _as_complex(args.target_beta if args.target_beta is not None
                    else cfg.get("target_beta"), "target_beta")
    if a is None:
        raise SystemExit("no target_alpha in config or --target-alpha; pass --target-alpha.")
    if b is None:
        raise SystemExit(
            f"config.json for this run has no 'target_beta' (QDax/MOME runs omit it). "
            f"Re-run with an explicit target, e.g.  --target-alpha {a.real:g} --target-beta '1+1j'  "
            f"(group {args.group!r} encodes alpha=a..., |beta|=b...; supply the true complex beta).")
    print(f"target: alpha={a}  beta={b}")
    O_lo = np.asarray(moment_operator(args.l_search, a, b))
    O_hi = np.asarray(moment_operator(args.l_high, a, b))
    depth = int(cfg.get("depth") or 3)
    maxf = int(cfg.get("moment_maxf") or 8)
    dec = get_genotype_decoder(cfg.get("genotype"), depth=depth, config=cfg)

    import functools
    # L and BF are buffer/scan SHAPES inside the herald (jnp.arange(BF), psi[L]),
    # so they must be static -- otherwise int(BF)/int(L) hit a traced value
    # (ConcretizationTypeError). g stays traced.
    @functools.partial(jax.jit, static_argnums=(1, 2))
    def chain(g, L, BF):
        p = dec.decode(g, base)
        cs, ms, ep, _ = jax_equivalent_gaussian_static(p, depth)
        return jax_reduced_herald_static(cs, ms, ep, L, BF, depth, maxf)

    @functools.partial(jax.jit, static_argnums=(1,))
    def leaf_prob(g, L):
        # exact 'leaf' herald probability (recovers the real value when the run
        # used --moment-fast, which stores a prob=1 placeholder in obj1).
        return _leaf_prob_product_static(dec.decode(g, base), L, depth)

    new_fit = fit.copy()
    has_prob = new_fit.shape[1] > 1
    n_probfix = 0
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
        if has_prob:
            P = float(np.real(leaf_prob(g, args.l_high)))
            log10P = float(np.log10(np.clip(P, 1e-45, 1.0)))
            if abs(log10P - new_fit[k, 1]) > 1e-6:
                n_probfix += 1
            new_fit[k, 1] = log10P                   # refresh exact probability (obj1=log10 P)
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
    if has_prob:
        print(f"probability objective refreshed on {n_probfix}/{len(valid)} cells "
              f"(recovers real prob where --moment-fast stored a prob=1 placeholder)")
    print(f"best exact <O> after dropping artifacts: {best_hi:.4f}")

    if args.write:
        out = os.path.join(os.path.dirname(pkl), "results_validated.pkl")
        new_rep = SimpleRepertoire(gen.astype(np.float32), new_fit, des)
        with open(out, "wb") as f:
            pickle.dump({"repertoire": new_rep, "config": cfg}, f)
        print(f"wrote exact-rescored repertoire -> {out}")


if __name__ == "__main__":
    main()
