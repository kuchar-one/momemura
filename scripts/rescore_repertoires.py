#!/usr/bin/env python3
"""rescore_repertoires.py -- exact, cutoff-free re-scoring of MOME repertoires.

Why
---
The Fock breeding sim that scored the repertoires is badly truncated for
high-squeezing solutions (HANAMURA_VALIDATION_FINDINGS.md), and the "photons"
descriptor can be padded with detections on control modes that are DECOUPLED
from the signal (physically inert "dud" photons).  Additionally, the scored
probability is the product of LEAF herald probabilities only -- the homodyne
acceptance (density x window per node) was never included in the fitness.

This script re-scores every valid genotype exactly in moment space:

  exp_exact   <O> on the reduced_herald state (analytic vacuum conditioning +
              Hermite recurrence -- exact at any squeezing, cutoff only limits
              the FINAL single-mode state, default 100)
  P_leaf      product of per-leaf PNR herald probabilities, computed exactly
              (the OLD fitness convention, now truncation-free)
  P_physical  P_leaf-equivalent joint INCLUDING homodyne acceptance:
              prod(density_i * window) * P(PNR pattern | homodyne outcomes)
  n_eff       effective photons: only detections on control modes whose
              cross-covariance to (signal + other controls) exceeds --coupling-eps

Outputs (mirror tree under --out, seedable by scan_results_for_seeds and
readable by pareto_report):
  <out>/<group>/<run>/results.pkl   {"repertoire": SimpleRepertoire, "config": ...}
        fitnesses = [-exp_exact, log10(P), -active, -n_eff]   (P per --prob)
        descriptors = [active, max_pnr_eff, n_eff]
        genotypes unchanged
  <out>/<group>/<run>/config.json   copied
  <out>/rescore_summary.csv         old-vs-new per point

Run on the cluster:
    JAX_ENABLE_X64=1 python scripts/rescore_repertoires.py \
        --root experiments --out experiments_rescored
"""
from __future__ import annotations
import os, sys, json, glob, time, shutil, argparse, pickle
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pareto_report as pr


# --------------------------------------------------------------------------- #
# GKP operator: load the cached high-dim (N=1000) operator if present (no
# qutip needed), else construct it with numpy (eigh-based cosm).
# --------------------------------------------------------------------------- #
def _numpy_cosm(H):
    w, V = np.linalg.eigh(H)
    return (V * np.cos(w)) @ V.conj().T


def gkp_operator(L, alpha, beta):
    from src.utils.gkp_operator import (CACHE_DIR, get_u_vec_from_alpha_beta,
                                        SQRT_PI)
    ux, uy, uz = get_u_vec_from_alpha_beta(complex(alpha), complex(beta))
    path = os.path.join(CACHE_DIR, f"high_dim_O_GKP_{ux:.6g}_{uy:.6g}_{uz:.6g}.npy")
    if os.path.isfile(path):
        return np.load(path)[:L, :L], (ux, uy, uz)
    # numpy fallback (qutip-free); N_big >> L so edge truncation is negligible
    N = max(4 * L, 400)
    a = np.diag(np.sqrt(np.arange(1, N)), k=1)
    x = (a + a.T) / np.sqrt(2.0)         # qutip position convention
    p = 1j * (a.T - a) / np.sqrt(2.0)
    Ox = _numpy_cosm(p * SQRT_PI)
    Oz = _numpy_cosm(x * SQRT_PI)
    Oy = _numpy_cosm((x - p) * SQRT_PI)
    O1 = np.eye(N) - (_numpy_cosm(2 * p * SQRT_PI) + _numpy_cosm(2 * (x - p) * SQRT_PI)
                      + _numpy_cosm(2 * x * SQRT_PI)) / 3.0
    O = O1 + np.eye(N) - (ux * Ox + uy * Oy + uz * Oz)
    return np.asarray(O)[:L, :L], (ux, uy, uz)


# --------------------------------------------------------------------------- #
def leaf_pnr_probs(params, cutoff=64):
    """Exact per-leaf PNR herald probabilities (the old fitness convention,
    truncation-free): reduced_herald on each active leaf's own Gaussian."""
    from frontend.independent_verifier import _build_gaussian_moments
    from frontend.gbs_optimizer import reduced_herald
    lp = params["leaf_params"]
    out = []
    for i in range(8):
        if not bool(np.asarray(params["leaf_active"])[i]):
            continue
        n_ctrl = int(np.asarray(lp["n_ctrl"])[i])
        N = n_ctrl + 1
        r = np.asarray(lp["r"][i], float)[:N]
        ph = np.asarray(lp["phases"][i], float)[:N * N]
        dv = np.asarray(lp["disp"][i])[:N]
        pnr = [int(x) for x in np.asarray(lp["pnr"][i])[:n_ctrl]]
        mu, cov = _build_gaussian_moments(r, ph, dv, N)
        _, p = reduced_herald(cov, mu, 0, list(range(1, N)), pnr, cutoff=cutoff)
        out.append(float(p))
    return out


def rescore_genotype(g, decoder, cfg, O, L, coupling_eps):
    """Exact moment-space score of one genotype.  Returns a dict or None."""
    import jax.numpy as jnp
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend.gbs_optimizer import reduced_herald

    params = {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in
              decoder.decode(jnp.asarray(np.asarray(g, np.float32)),
                             int(cfg.get("cutoff") or 30)).items()}
    eq = compute_equivalent_gaussian(params, light=True)
    cov = np.asarray(eq["cov"], float); mu = np.asarray(eq["mu"], float)
    N = cov.shape[0] // 2
    ctrl = [int(x) for x in eq["control_idx"]]
    sig = int(eq["signal_idx"])
    n0 = [int(x) for x in eq["pnr_outcomes"]]

    # effective photons: coupling of each fired mode to (signal + other ctrls)
    n_eff, max_pnr_eff, couplings = 0, 0, []
    for j, ci in enumerate(ctrl):
        if n0[j] == 0:
            continue
        others = [sig] + [c2 for k2, c2 in enumerate(ctrl) if k2 != j]
        oi = [i for o in others for i in (o, o + N)]
        cpl = float(np.linalg.norm(cov[np.ix_([ci, ci + N], oi)], 2))
        couplings.append(round(cpl, 4))
        if cpl > coupling_eps:
            n_eff += n0[j]
            max_pnr_eff = max(max_pnr_eff, n0[j])

    psi, p_pnr = reduced_herald(cov, mu, sig, ctrl, n0, cutoff=L)
    if not np.isfinite(psi).all() or np.linalg.norm(psi) < 0.5:
        return None
    exp_exact = float(np.real(np.vdot(psi, O @ psi)))

    window = float(params.get("homodyne_window") or 0.0)
    dens = [float(d) for d in eq.get("homodyne_densities", [])]
    P_leaf = float(np.prod(leaf_pnr_probs(params))) if True else None
    P_phys = float(np.prod([d * (window if window > 0 else 1.0) for d in dens])
                   * p_pnr) if dens else float(p_pnr)

    return dict(exp=exp_exact, P_leaf=P_leaf, P_phys=P_phys, p_pnr=float(p_pnr),
                n_det=int(sum(n0)), n_eff=int(n_eff), max_pnr_eff=int(max_pnr_eff),
                couplings=couplings,
                gbs_sq_db=round(float(eq["max_squeezing_db"]), 2))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO, "experiments"))
    ap.add_argument("--out", default=os.path.join(REPO, "experiments_rescored"))
    ap.add_argument("--cutoff", type=int, default=100,
                    help="Fock cutoff of the FINAL single-mode state (cheap)")
    ap.add_argument("--coupling-eps", type=float, default=0.05,
                    help="fired-mode coupling below this counts as a dud photon")
    ap.add_argument("--prob", choices=["leaf", "physical"], default="leaf",
                    help="probability convention for fitness[1]: 'leaf' = old "
                         "convention (product of leaf herald probs, exact); "
                         "'physical' = includes homodyne acceptance")
    ap.add_argument("--groups", default=None,
                    help="comma-separated group-dir name filter (substring)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max genotypes per run (0 = all; use for smoke tests)")
    args = ap.parse_args()

    from src.utils.result_manager import SimpleRepertoire
    from src.genotypes.genotypes import get_genotype_decoder

    run_dirs = sorted(glob.glob(os.path.join(args.root, "*", "*", "config.json")))
    if args.groups:
        keys = [k.strip() for k in args.groups.split(",")]
        run_dirs = [c for c in run_dirs
                    if any(k in os.path.basename(os.path.dirname(os.path.dirname(c)))
                           for k in keys)]

    os.makedirs(args.out, exist_ok=True)
    summary = open(os.path.join(args.out, "rescore_summary.csv"), "w")
    summary.write("group,run,idx,exp_old,exp_new,logP_old,logP_leaf_new,logP_phys,"
                  "photons_old,n_eff,max_pnr_eff,gbs_sq_db,couplings\n")

    op_cache = {}
    for cfgf in run_dirs:
        rundir = os.path.dirname(cfgf)
        run = os.path.basename(rundir)
        group = os.path.basename(os.path.dirname(rundir))
        pkl = os.path.join(rundir, "results.pkl")
        if not os.path.exists(pkl):
            continue
        cfg = json.load(open(cfgf))
        try:
            rep = pr.load_repertoire(pkl)
        except Exception as e:
            print(f"[!] {group}/{run}: repertoire load failed ({e!r}) -- skipped")
            continue
        if rep is None:
            print(f"[!] {group}/{run}: unreadable repertoire -- skipped"); continue
        fit = np.asarray(rep.fitnesses, np.float64).reshape(-1, np.asarray(rep.fitnesses).shape[-1])
        des = np.asarray(rep.descriptors, np.float64).reshape(-1, np.asarray(rep.descriptors).shape[-1])
        gen = np.asarray(rep.genotypes, np.float64).reshape(-1, np.asarray(rep.genotypes).shape[-1])
        valid = np.where(np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9))[0]
        if args.limit:
            valid = valid[:args.limit]

        ab = (cfg.get("target_alpha"), cfg.get("target_beta"))
        key = str(ab)
        if key not in op_cache:
            op_cache[key] = gkp_operator(args.cutoff, complex(str(ab[0]).replace("i", "j")),
                                         pr.parse_complex(str(ab[1])))
        O, _u = op_cache[key]
        decoder = get_genotype_decoder(cfg.get("genotype"),
                                       depth=int(cfg.get("depth") or 3), config=cfg)

        new_fit = np.full_like(fit, -np.inf)
        new_des = des.copy()
        t0 = time.time(); n_ok = 0; n_dud = 0
        for k in valid:
            try:
                r = rescore_genotype(gen[k], decoder, cfg, O, args.cutoff,
                                     args.coupling_eps)
            except Exception:
                r = None
            if r is None:
                continue
            P = r["P_leaf"] if args.prob == "leaf" else r["P_phys"]
            logP = float(np.log10(max(P, 1e-300)))
            new_fit[k] = [-r["exp"], logP, fit[k, 2], -float(r["n_eff"])]
            new_des[k] = [des[k, 0], float(r["max_pnr_eff"]), float(r["n_eff"])] \
                if des.shape[1] >= 3 else des[k]
            n_ok += 1
            if r["n_eff"] < r["n_det"]:
                n_dud += 1
            summary.write(f"{group},{run},{k},{-fit[k,0]:.6g},{r['exp']:.6g},"
                          f"{fit[k,1]:.4f},{np.log10(max(r['P_leaf'],1e-300)):.4f},"
                          f"{np.log10(max(r['P_phys'],1e-300)):.4f},"
                          f"{-fit[k,3]:.0f},{r['n_eff']},{r['max_pnr_eff']},"
                          f"{r['gbs_sq_db']},\"{r['couplings']}\"\n")
        summary.flush()

        outdir = os.path.join(args.out, group, run)
        os.makedirs(outdir, exist_ok=True)
        new_rep = SimpleRepertoire(gen.astype(np.float32), new_fit, new_des)
        with open(os.path.join(outdir, "results.pkl"), "wb") as f:
            pickle.dump({"repertoire": new_rep, "config": cfg}, f)
        shutil.copy(cfgf, os.path.join(outdir, "config.json"))
        print(f"[+] {group}/{run}: rescored {n_ok}/{len(valid)} "
              f"({n_dud} with dud photons) in {time.time()-t0:.1f}s")

    summary.close()
    print(f"\n[+] wrote {args.out} (mirror tree + rescore_summary.csv)")


if __name__ == "__main__":
    main()
