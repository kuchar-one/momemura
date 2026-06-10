#!/usr/bin/env python3
"""check_path_consistency.py -- the DECISIVE A-vs-B check from handoff.md §7.2.

For one chosen solution (default plus_0, the N_c=1 smoking gun) it computes:

  (A) path-1: utils.compute_state_with_jax   (called DIRECTLY so errors surface;
      compute_heralded_state swallows exceptions and silently returns zeros)
  (B) path-2: independent_verifier.verify_circuit (thewalrus+scipy, no JAX)

and reports:
  * raw fidelity |<A|B>|^2  (A and B live in the same absolute frame -> raw
    fidelity is the right metric here, no Gaussian alignment needed)
  * rank-n core-state fidelity of A and of B (up to a Gaussian unitary),
    n = the detected photon number of the firing control mode
  * the correlation between the firing control mode and the n_m=0 spectator
    modes in the equivalent-GBS covariance (hypothesis 3 of the handoff)

Run with x64 (default; pass --f32 to reproduce the cluster's single-precision
behaviour for comparison):

    python scripts/check_path_consistency.py --key plus_0 --cutoff 24
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="plus_0", help="chosen_genotypes key, e.g. plus_0")
    ap.add_argument("--cutoff", type=int, default=24)
    ap.add_argument("--f32", action="store_true", help="run JAX in float32 (cluster default)")
    ap.add_argument("--outputs", default=os.path.join(REPO, "outputs"))
    ap.add_argument("--restarts", type=int, default=8,
                    help="extra random restarts for the core-state alignment")
    args = ap.parse_args()

    if not args.f32:
        os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", not args.f32)
    import jax.numpy as jnp

    from src.genotypes.genotypes import get_genotype_decoder
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend import gbs_optimizer as go
    from frontend.utils import compute_state_with_jax
    from frontend.independent_verifier import verify_circuit
    import importlib.util as _u
    _spec = _u.spec_from_file_location("ghd", os.path.join(REPO, "scripts", "gen_hanamura_data.py"))
    ghd = _u.module_from_spec(_spec); _spec.loader.exec_module(ghd)

    # ---- load the pinned solution ---------------------------------------- #
    geno = np.load(os.path.join(args.outputs, "chosen_genotypes.npz"))
    meta = json.load(open(os.path.join(args.outputs, "chosen_genotypes_meta.json")))
    tgt, row = args.key.rsplit("_", 1)
    m = next(c for c in meta if c["target"] == tgt and int(c["row"]) == int(row))
    g = geno[f"{args.key}_g"]
    c = m["config"]
    depth = int(c.get("depth") or 3)
    cfg_cutoff = int(c.get("cutoff") or 30)
    L = int(args.cutoff)
    print(f"[{args.key}] gname={m['gname']} run={m['run']} Nc={m['Nc']} "
          f"P={m['prob']:.3e} | sim cutoff(cfg)={cfg_cutoff} recon cutoff={L} "
          f"| x64={'OFF (f32)' if args.f32 else 'ON'}")

    dec = get_genotype_decoder(m["gname"], depth=depth, config=c)
    params = {k: np.asarray(v) if hasattr(v, "shape") else v
              for k, v in dec.decode(jnp.asarray(np.asarray(g, np.float32)), cfg_cutoff).items()}

    # ---- equivalent-GBS moments (trusted) --------------------------------- #
    eq = compute_equivalent_gaussian(params)
    n0 = [int(x) for x in eq["pnr_outcomes"]]
    fire = [j for j, x in enumerate(n0) if x >= 1]
    print(f"  GBS: k_control={len(eq['control_idx'])} n0={n0} k_eff={len(fire)} "
          f"sq={float(eq['max_squeezing_db']):.2f} dB")

    # ---- (A) path-1 (direct, errors surface) ------------------------------ #
    psiA, probA = compute_state_with_jax(params, cutoff=L)
    psiA = np.asarray(psiA).ravel()
    print(f"  A: norm={np.linalg.norm(psiA):.6f} P={probA:.3e} "
          f"nbar={np.sum(np.arange(len(psiA)) * np.abs(psiA)**2) / max(np.linalg.norm(psiA)**2, 1e-300):.3f}")
    psiA = psiA / np.linalg.norm(psiA)

    # ---- (B) path-2 -------------------------------------------------------- #
    pnr_max = max(3, max(n0, default=0) + 1)
    vb = verify_circuit(params, cutoff=L, pnr_max=pnr_max)
    psiB = np.asarray(vb["state"]).ravel()
    print(f"  B: norm={np.linalg.norm(psiB):.6f} P={vb['probability']:.3e} "
          f"warnings={vb['report']['warnings'] or 'none'}")
    psiB = psiB / np.linalg.norm(psiB)

    # ---- A vs B (raw -- same frame) ---------------------------------------- #
    f_raw = abs(np.vdot(psiA, psiB)) ** 2
    print(f"\n  ==> raw |<A|B>|^2 = {f_raw:.6f}   "
          f"({'A==B: reconstructions AGREE' if f_raw > 0.999 else 'A!=B: RECONSTRUCTION BUG'})")

    # ---- rank-n core check (up to Gaussian U) ------------------------------ #
    if len(fire) == 1:
        jf = fire[0]
        ci = int(eq["control_idx"][jf]); N = np.asarray(eq["cov"]).shape[0] // 2
        idx = [ci, ci + N]
        cov = np.asarray(eq["cov"], float); mu = np.asarray(eq["mu"], float)
        p = go.control_parameters(cov[np.ix_(idx, idx)], mu[idx])
        n_m = n0[jf]
        core = ghd.core_state(p["s0"], p["delta0"], n_m, L)
        rng = np.random.default_rng(0)
        for name, psi in (("A", psiA), ("B", psiB)):
            best = 0.0
            f, _ = go.align_states(psi, core, L, align_cut=min(L, 30))
            best = max(best, f)
            for _ in range(args.restarts):
                gss = (rng.normal(0, 0.8), rng.normal(0, 0.8),
                       rng.normal(0, 0.6), rng.uniform(-np.pi, np.pi),
                       rng.uniform(-np.pi, np.pi))
                f, _ = go.align_states(psi, core, L, align_cut=min(L, 30), guess=gss)
                best = max(best, f)
            print(f"  core(n={n_m}, s0={p['s0']:.4g}, |d0|={abs(p['delta0']):.4g}) "
                  f"vs {name}: F_gauss = {best:.4f}")

        # ---- hypothesis 3: firing-mode <-> spectator correlations ---------- #
        ctrl = [int(x) for x in eq["control_idx"]]
        spect = [cidx for j, cidx in enumerate(ctrl) if j != jf]
        if spect:
            sp_idx = [i for s in spect for i in (s, s + N)]
            cross = cov[np.ix_(idx, sp_idx)]
            # normalize by the geometric mean of the diagonal blocks
            sA = np.sqrt(np.linalg.norm(cov[np.ix_(idx, idx)], 2))
            sB = np.sqrt(np.linalg.norm(cov[np.ix_(sp_idx, sp_idx)], 2))
            print(f"  firing-mode/spectator cross-cov: |C_xy|_2 = {np.linalg.norm(cross, 2):.4f} "
                  f"(normalized {np.linalg.norm(cross, 2) / (sA * sB):.4f}) "
                  f"-- 0 means the single-mode core form is exact")
        # signal-spectator too
    else:
        print(f"  k_eff={len(fire)} != 1 -> no single-mode core check")


if __name__ == "__main__":
    main()
