#!/usr/bin/env python3
"""validate_convergence.py -- prove that the breeding sim (path-1) converges to
the exact moment-space herald as the Fock cutoff grows.

Background (see HANAMURA_VALIDATION_FINDINGS.md): the canonical-trio solutions
carry 11-12 dB of internal squeezing, so the breeding sim at the config cutoff
(30) is BADLY truncated (tail amplitudes ~0.1).  All previous Fock-space
validation failures (the plus_0 "rank-1 failure", coreFid 0.5-0.9, A-vs-C
mismatches) trace back to this.  The exact physical output is computed cheaply
and stably by `gbs_optimizer.reduced_herald` on the equivalent-GBS moments
(analytic vacuum conditioning + Hermite recurrence; verified exact against
post-selected heralding in tests/test_reduced_herald.py).

This script ramps the path-1 cutoff and reports F(A_L, reduced_herald) per
solution.  Expected: F -> 1.  Sandbox CPU results (plus_0): 0.58@24, 0.73@40,
0.88@56; (T_1): 0.24@40, 0.74@56.

Run on the cluster (GPU jax, x64 is forced):

    python scripts/validate_convergence.py --out outputs/convergence
    # quick subset:
    python scripts/validate_convergence.py --keys plus_0,T_1 --cutoffs 40,72,104,136

Also prints each row's fired-mode coupling |C_cross|_2 (a fired control mode
that is DECOUPLED from signal+rest contributes nothing physically -- plus_0's
single photon is such a dud, its physical output is exactly Gaussian).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULT_KEYS = ["plus_0", "plus_2", "plus_3", "H_0", "H_1", "T_0", "T_1", "T_2"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default=os.path.join(REPO, "outputs"),
                    help="dir with chosen_genotypes.npz/meta (default <repo>/outputs)")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "convergence"))
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    ap.add_argument("--cutoffs", default="40,56,72,96,128,160")
    ap.add_argument("--check-b", action="store_true",
                    help="also run independent_verifier at cutoff 24 (slow, CPU)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import jax.numpy as jnp
    from src.genotypes.genotypes import get_genotype_decoder
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend.utils import compute_state_with_jax
    from frontend import gbs_optimizer as go

    cutoffs = [int(x) for x in args.cutoffs.split(",")]
    Lmax = max(cutoffs)
    geno = np.load(os.path.join(args.outputs, "chosen_genotypes.npz"))
    meta = json.load(open(os.path.join(args.outputs, "chosen_genotypes_meta.json")))

    report = {}
    for key in args.keys.split(","):
        tgt, row = key.rsplit("_", 1)
        m = next((c for c in meta if c["target"] == tgt and int(c["row"]) == int(row)), None)
        if m is None:
            print(f"[{key}] not in chosen_genotypes -- skipped"); continue
        c = m["config"]
        dec = get_genotype_decoder(m["gname"], depth=int(c.get("depth") or 3), config=c)
        params = {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in
                  dec.decode(jnp.asarray(np.asarray(geno[f"{key}_g"], np.float32)),
                             int(c.get("cutoff") or 30)).items()}
        eq = compute_equivalent_gaussian(params)
        cov = np.asarray(eq["cov"], float); mu = np.asarray(eq["mu"], float)
        N = cov.shape[0] // 2
        ctrl = [int(x) for x in eq["control_idx"]]; sig = int(eq["signal_idx"])
        n0 = [int(x) for x in eq["pnr_outcomes"]]

        # fired-mode coupling audit
        couplings = []
        for j, ci in enumerate(ctrl):
            if n0[j] == 0:
                continue
            others = [sig] + [c2 for k2, c2 in enumerate(ctrl) if k2 != j]
            oi = [i for o in others for i in (o, o + N)]
            X = cov[np.ix_([ci, ci + N], oi)]
            couplings.append((n0[j], round(float(np.linalg.norm(X, 2)), 4)))

        ref, p_ref = go.reduced_herald(cov, mu, sig, ctrl, n0, cutoff=Lmax)
        rec = dict(n0=n0, fired_couplings=couplings,
                   gbs_sq_db=round(float(eq["max_squeezing_db"]), 2),
                   P_exact=p_ref, scored_P=m["prob"], curve=[])
        print(f"[{key}] n0={n0} sq={rec['gbs_sq_db']}dB fired(n,|C|)={couplings} "
              f"P_exact={p_ref:.3e} P_scored={m['prob']:.3e}")

        for L in cutoffs:
            psi, prob = compute_state_with_jax(params, cutoff=L)
            psi = np.asarray(psi).ravel()
            nrm = np.linalg.norm(psi)
            if nrm < 1e-12:
                print(f"    L={L}: path-1 returned zeros"); continue
            psi /= nrm
            r = ref[:L] / np.linalg.norm(ref[:L])
            F = float(abs(np.vdot(psi, r)) ** 2)
            nb = float(np.sum(np.arange(L) * np.abs(psi) ** 2))
            tail = float(np.abs(psi[-4:]).max())
            rec["curve"].append(dict(L=L, P=prob, nbar=round(nb, 3),
                                     tail=tail, F=round(F, 5)))
            print(f"    L={L:4d}: P={prob:.4e} nbar={nb:7.3f} tail={tail:.2e} "
                  f"F(A,exact)={F:.5f}")

        if args.check_b:
            from frontend.independent_verifier import verify_circuit
            try:
                vb = verify_circuit(params, cutoff=24, pnr_max=max(3, max(n0) + 1))
                psiB = np.asarray(vb["state"]).ravel()
                psiA24, _ = compute_state_with_jax(params, cutoff=24)
                psiA24 = np.asarray(psiA24).ravel()
                fab = float(abs(np.vdot(psiA24 / np.linalg.norm(psiA24),
                                        psiB / np.linalg.norm(psiB))) ** 2)
                rec["F_AB_24"] = round(fab, 5)
                print(f"    A==B @24: {fab:.5f}")
            except Exception as e:
                rec["F_AB_error"] = repr(e)[:120]

        report[key] = rec

    out_json = os.path.join(args.out, "convergence_report.json")
    json.dump(report, open(out_json, "w"), indent=2, default=float)
    print(f"\n[+] wrote {out_json}")

    # verdict summary
    print("\n=== VERDICT (F at largest cutoff; want > 0.99) ===")
    for key, rec in report.items():
        if rec["curve"]:
            last = rec["curve"][-1]
            dud = any(cp < 0.05 for _, cp in rec["fired_couplings"]) if rec["fired_couplings"] else False
            print(f"  {key}: F={last['F']:.4f} @L={last['L']} tail={last['tail']:.1e}"
                  + ("   [contains DECOUPLED fired mode -> physically Gaussian-ish photon dud]" if dud else ""))


if __name__ == "__main__":
    main()
