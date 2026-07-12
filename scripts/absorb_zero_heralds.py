#!/usr/bin/env python3
"""Absorb zero-photon heralds of the picks' equivalent-GBS generators into the
Gaussian unitary (exact vacuum projection), verify state preservation and the
probability factor, and emit the absorbed architectures.

For each pick with 0-entries in its PNR pattern:
  1. decode genotype -> compute_equivalent_gaussian -> (cov, mu, signal, controls, n0)
  2. sanity: reduced_herald reproduces the stored psi_before (fidelity) and prob
  3. project the 0-herald control modes onto vacuum (formulas unit-tested in
     test_vacuum_projection.py; hbar=2, xp-ordered)
  4. re-herald the projected generator on the remaining pattern:
     - fidelity(psi', psi_before) ~ 1
     - prob' ~ prob / P0
  5. report new squeezing spectrum (pure_state_squeezings on cov')
"""
import os, sys, json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src.genotypes.genotypes import get_genotype_decoder
from frontend.gaussian_decomposition import compute_equivalent_gaussian
from frontend.gbs_optimizer import reduced_herald, pure_state_squeezings
from rescore_all_experiments import load_run_arrays

def to_numpy(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out[k] = to_numpy(v)
        elif hasattr(v, "shape"): out[k] = np.asarray(v)
        else: out[k] = v
    return out

DB = 10*np.log10(np.exp(2.0))

def project_vacuum(cov, mu, modes_B, N):
    A = [i for i in range(N) if i not in modes_B]
    iA = A + [a + N for a in A]; iB = list(modes_B) + [b + N for b in modes_B]
    S_AA = cov[np.ix_(iA, iA)]; S_AB = cov[np.ix_(iA, iB)]
    S_BB = cov[np.ix_(iB, iB)]; muA = mu[iA]; muB = mu[iB]
    M = S_BB + np.eye(len(iB)); Minv = np.linalg.inv(M)
    P0 = 2.0**len(modes_B)/np.sqrt(np.linalg.det(M))*np.exp(-muB @ Minv @ muB/2)
    covp = S_AA - S_AB @ Minv @ S_AB.T
    mup = muA - S_AB @ Minv @ muB
    return float(P0), covp, mup, A

D = json.load(open(os.environ.get("NG_DATA",
    os.path.expanduser("~/Nextcloud/vojtech/writing/mgr/scripts/ng_results_data.json"))))
Z = np.load(os.environ.get("NG_STATES",
    os.path.expanduser("~/Nextcloud/vojtech/writing/mgr/scripts/ng_results_states.npz")))

out = {}
for t in ("plus", "H", "T"):
    for i, rec in enumerate(D["picks"][t]):
        if 0 not in rec["arch"]["pnr"]:
            continue
        run = rec["run"].split("/")[-1]
        base = os.path.join(REPO, "output/experiments", rec["group"], run)
        gens, _f, _d = load_run_arrays(os.path.join(base, "results.pkl"))
        cfg = json.load(open(os.path.join(base, "config.json")))
        gens = np.asarray(gens).reshape(-1, np.asarray(gens).shape[-1])
        g = gens[rec["cell"]].astype(np.float32)
        depth = int(cfg.get("depth") or 3); cutoff = int(cfg.get("cutoff") or 30)
        dec = get_genotype_decoder(cfg.get("genotype"), depth=depth, config=cfg)
        params = to_numpy(dec.decode(jnp.asarray(g), cutoff))
        eq = compute_equivalent_gaussian(params)
        cov = np.asarray(eq["cov"], float); mu = np.asarray(eq["mu"], float)
        sig = int(eq["signal_idx"]); ctr = [int(c) for c in eq["control_idx"]]
        n0 = [int(x) for x in eq["pnr_outcomes"]]
        N = cov.shape[0]//2
        hcut = 32
        psi_b, prob_b = reduced_herald(cov, mu, sig, ctr, n0, cutoff=hcut)
        psi_b = np.asarray(psi_b).ravel()
        psi_ref = Z[f"{t}_{i}_psi"]
        L = min(len(psi_b), len(psi_ref))
        fid_ref = abs(np.vdot(psi_ref[:L], psi_b[:L]))**2
        print(f"\n== {t}_{i} ({rec['group']} {run} cell{rec['cell']}) ==")
        print(f" pattern {n0}  prob_reduced={prob_b:.6e} "
              f"(stored {rec['arch']['prob_reduced']:.6e})  fid(stored psi)={fid_ref:.10f}")

        zero_modes = [c for c, n in zip(ctr, n0) if n == 0]
        keep_pattern = [n for n in n0 if n > 0]
        P0, covp, mup, A = project_vacuum(cov, mu, zero_modes, N)
        # index bookkeeping in the reduced register
        old2new = {m: k for k, m in enumerate(A)}
        sig2 = old2new[sig]
        ctr2 = [old2new[c] for c in ctr if c not in zero_modes]
        psi_p, prob_p = reduced_herald(covp, mup, sig2, ctr2, keep_pattern, cutoff=hcut)
        psi_p = np.asarray(psi_p).ravel()
        fid = abs(np.vdot(psi_b, psi_p[:len(psi_b)]))**2
        r_new = pure_state_squeezings(covp)
        db_new = np.sort(r_new)[::-1]*DB
        db_old = np.array(rec["arch"]["squeezings_db"])
        print(f" P0(vacuum herald) = {P0:.6f}  -> boost x{1/P0:.3f}")
        print(f" prob' = {prob_p:.6e}  vs prob/P0 = {prob_b/P0:.6e}  "
              f"ratio {prob_p/(prob_b/P0):.8f}")
        print(f" fidelity(psi_absorbed, psi_before) = {fid:.10f}")
        print(f" squeezings before [dB]: {np.round(db_old,2)}")
        print(f" squeezings after  [dB]: {np.round(db_new,2)}")
        out[f"{t}_{i}"] = dict(
            P0=P0, boost=1/P0, pnr=keep_pattern,
            squeezings_db=[float(x) for x in db_new],
            max_squeezing_db=float(db_new[0]),
            n_modes=len(A), fid=fid,
        )

json.dump(out, open("absorbed_archs.json", "w"), indent=1)
print("\nwrote absorbed_archs.json")
