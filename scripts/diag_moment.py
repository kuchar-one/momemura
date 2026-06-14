#!/usr/bin/env python3
"""diag_moment.py -- diagnose the moment scorer on a population.

Confirms x64 is actually on, scores a batch of genotypes with the single-compile
moment scorer, and flags any pathological points (raw <O> ~ 0, zero-norm herald,
invalid/-inf fitness).  Use to verify the exp=0.000 artifact is gone.

Run on the cluster:
    JAX_ENABLE_X64=1 python scripts/diag_moment.py --group 00B_c30_a2p73_b1p41 --n 64
"""
import os, sys, glob, json, argparse
os.environ["JAX_ENABLE_X64"] = "1"           # the moment scorer REQUIRES x64
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--group", default="00B_c30_a2p73_b1p41")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--moment-cutoff", type=int, default=100)
    args = ap.parse_args()

    import jax, jax.numpy as jnp
    print(f"jax x64 enabled: {jax.config.jax_enable_x64}  (MUST be True)")
    print(f"jax devices: {jax.devices()}")

    import pareto_report as pr
    from src.genotypes.genotypes import get_genotype_decoder
    from src.simulation.jax.moment_scorer import (
        moment_operator, jax_equivalent_gaussian_static,
        jax_reduced_herald_static, fired_product, extract_structure,
        MOMENT_MAXF, MOMENT_BF)

    roots = ([args.root] if args.root else
             [os.path.join(REPO, "experiments"), os.path.join(REPO, "output", "experiments")])
    cfgs = []
    for r in roots:
        cfgs = sorted(glob.glob(os.path.join(r, args.group, "*", "config.json")))
        if cfgs:
            break
    cfg = json.load(open(cfgs[0]))
    rep = pr.load_repertoire(cfgs[0].replace("config.json", "results.pkl"))
    fit = np.asarray(rep.fitnesses, float).reshape(-1, np.asarray(rep.fitnesses).shape[-1])
    gen = np.asarray(rep.genotypes, float).reshape(-1, np.asarray(rep.genotypes).shape[-1])
    valid = np.where(np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9))[0][:args.n]
    base = int(cfg.get("cutoff") or 30); L = args.moment_cutoff
    a, b = cfg.get("target_alpha"), cfg.get("target_beta")
    OL = moment_operator(L, a, b)
    dec = get_genotype_decoder(cfg.get("genotype"), depth=int(cfg.get("depth") or 3), config=cfg)

    print(f"\nscoring {len(valid)} genotypes from {args.group} (L={L})")
    print(f"{'idx':>5} {'raw_exp':>9} {'herald_norm':>11} {'P_leaf':>10} {'kf':>3} {'prod':>7} {'flag':>6}")
    n_zero = 0
    for k in valid:
        g = jnp.asarray(gen[k].astype(np.float32))
        params = dec.decode(g, base)
        st = extract_structure(params)
        n0 = tuple(p for leaf in st[2] for p in leaf)
        kf = sum(1 for p in n0 if p >= 1)
        prod = fired_product(n0)
        cs, ms, ep, dn = jax_equivalent_gaussian_static(params)
        psi, prob = jax_reduced_herald_static(cs, ms, ep, L)
        psi = np.asarray(psi)
        norm = float(np.sum(np.abs(psi) ** 2))
        exp = float(np.real(np.vdot(psi, np.asarray(OL)[:len(psi), :len(psi)] @ psi)))
        flag = ""
        if abs(exp) < 1e-6 or norm < 0.5:
            flag = "ZERO"; n_zero += 1
        if kf > MOMENT_MAXF or prod > MOMENT_BF:
            flag = (flag + " OVER").strip()
        print(f"{int(k):>5} {exp:9.4f} {norm:11.4f} {float(prob):10.2e} {kf:>3} {prod:>7} {flag:>6}")
    print(f"\nzero/invalid heralds: {n_zero}/{len(valid)}  "
          f"(should be 0 with x64 on; over-budget rows route to CPU fallback in the live scorer)")


if __name__ == "__main__":
    main()
