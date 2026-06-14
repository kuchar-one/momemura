#!/usr/bin/env python3
"""consistency_moment_vs_fock.py -- the moment-scorer acceptance gate.

Scores the SAME genotypes with both backends through the real wired path
(`jax_scoring_fn_batch`, `scorer="fock"` vs `scorer="moment"`) and checks that
on WELL-BEHAVED (low-squeezing) solutions the exact moment scorer reproduces the
truncated Fock scorer's <O> and log-probability. On high-squeezing solutions
they are EXPECTED to diverge (the Fock score is the truncation artifact) -- the
script reports both so you can see the regime split.

PASS criterion: median |Δ<O>| over the low-dB set < --tol (default 0.02), and
all gradients from the moment scorer are finite.

Run on the cluster (GPU, x64):
    JAX_ENABLE_X64=1 python scripts/consistency_moment_vs_fock.py \
        --group 00B_c30_a1p00_b1p00 --n 60
"""
import os, sys, glob, json, argparse
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "experiments"))
    ap.add_argument("--group", default="00B_c30_a1p00_b1p00")
    ap.add_argument("--n", type=int, default=60, help="max genotypes to score")
    ap.add_argument("--moment-cutoff", type=int, default=100)
    ap.add_argument("--sq-threshold", type=float, default=8.0,
                    help="gbs squeezing (dB) below which a solution is 'well-behaved'")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="max median |Δ<O>| on the well-behaved set to PASS")
    args = ap.parse_args()

    import jax.numpy as jnp
    import pareto_report as pr
    from src.simulation.jax.runner import jax_scoring_fn_batch
    from src.genotypes.genotypes import get_genotype_decoder
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from src.utils.gkp_operator import construct_gkp_operator

    cfgf = sorted(glob.glob(os.path.join(args.root, args.group, "*", "config.json")))[0]
    cfg = json.load(open(cfgf))
    rep = pr.load_repertoire(cfgf.replace("config.json", "results.pkl"))
    fit = np.asarray(rep.fitnesses, float).reshape(-1, np.asarray(rep.fitnesses).shape[-1])
    gen = np.asarray(rep.genotypes, float).reshape(-1, np.asarray(rep.genotypes).shape[-1])
    valid = np.where(np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9))[0][:args.n]

    base = int(cfg.get("cutoff") or 30)
    a, b = cfg.get("target_alpha"), cfg.get("target_beta")
    O30 = construct_gkp_operator(base, complex(str(a).replace("i", "j")),
                                 complex(str(b).replace("i", "j")), backend="jax")
    name = cfg.get("genotype")
    dec = get_genotype_decoder(name, depth=int(cfg.get("depth") or 3), config=cfg)

    # squeezing per genotype (regime split)
    sq = []
    for k in valid:
        params = dec.decode(jnp.asarray(gen[k].astype(np.float32)), base)
        pnp = {kk: (np.asarray(v) if hasattr(v, "shape") else v)
               for kk, v in params.items()}
        sq.append(float(compute_equivalent_gaussian(pnp, light=True)["max_squeezing_db"]))
    sq = np.asarray(sq)
    G = jnp.asarray(gen[valid].astype(np.float32))

    cfg_fock = dict(cfg); cfg_fock["scorer"] = "fock"
    cfg_mom = dict(cfg); cfg_mom["scorer"] = "moment"
    cfg_mom["moment_cutoff"] = args.moment_cutoff
    cfg_mom["target_alpha"] = a; cfg_mom["target_beta"] = b
    pnr_max = int(cfg.get("pnr_max", 3))

    _, _, eF = jax_scoring_fn_batch(G, base, O30, genotype_name=name,
                                    genotype_config=cfg_fock, pnr_max=pnr_max)
    _, _, eM = jax_scoring_fn_batch(G, base, O30, genotype_name=name,
                                    genotype_config=cfg_mom, pnr_max=pnr_max)
    reF = np.asarray(eF["raw_expectation"]); reM = np.asarray(eM["raw_expectation"])
    jpF = np.asarray(eF["joint_probability"]); jpM = np.asarray(eM["joint_probability"])
    gM = np.asarray(eM["gradients"])

    dexp = np.abs(reF - reM)
    well = sq < args.sq_threshold
    grads_ok = bool(np.all(np.isfinite(gM)))

    print(f"\n{'idx':>5} {'sq_dB':>6} {'exp_fock':>9} {'exp_mom':>9} {'dexp':>9} "
          f"{'logPf':>8} {'logPm':>8} {'regime':>6}")
    for i, k in enumerate(valid):
        print(f"{k:>5} {sq[i]:6.1f} {reF[i]:9.4f} {reM[i]:9.4f} {dexp[i]:9.2e} "
              f"{np.log10(max(jpF[i],1e-30)):8.3f} {np.log10(max(jpM[i],1e-30)):8.3f} "
              f"{'WELL' if well[i] else 'high':>6}")

    med_well = float(np.median(dexp[well])) if well.any() else float('nan')
    print(f"\nwell-behaved (<{args.sq_threshold}dB): {int(well.sum())}/{len(valid)} | "
          f"median |Δ<O>|={med_well:.2e} | high-dB median |Δ<O>|="
          f"{float(np.median(dexp[~well])) if (~well).any() else float('nan'):.2e}")
    print(f"moment gradients all finite: {grads_ok}")
    ok = well.any() and med_well < args.tol and grads_ok
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} "
          f"(median |Δ<O>| on well-behaved {med_well:.2e} {'<' if ok else '>='} tol {args.tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
