#!/usr/bin/env python3
"""
gen_hanamura_data.py -- regenerate the Hanamura before/after data for the
canonical trio {|+>, |H>, |T>} with the BUG-FIXED heralding code.

Why this exists
---------------
The cached `wigner_pareto_pairs.npz` (and the `tab:hanamura` numbers derived
from it) were produced *before* the heralding-convention fix in
`frontend/gaussian_decomposition.py` (transposed beam-splitter / parity), and
the "before" states in it were reconstructed with `heralded_output` on the
moment-reduced Gaussian (the "path-3" route) which is numerically ill-conditioned
for high-energy heralds and collapses displaced-squeezed states onto their
even-parity core.  This script regenerates everything from the raw repertoires
with the fixed code, using the two TRUSTED reconstruction routes:

  * BEFORE state  -> `utils.compute_heralded_state` (the JAX breeding simulation,
                     "path-1": heralds per leaf in small, well-conditioned Fock
                     systems then mixes -- the route the optimizer/runner trust).

  * AFTER  state  -> the ARCHITECTURE RULE: apply the Step-1 photon-reduction
                     single-mode symplectic to the FULL equivalent-GBS covariance
                     (which already contains the single signal mode) and then
                     herald, instead of `purify_control`-ing the reduced control
                     marginal (which over-purifies into a spurious multi-mode
                     state for high-Nc reductions).

The Hanamura optimisation itself lives purely in moment space and is unaffected
by the Fock-reconstruction bug; we still rerun it on the fixed moments so the
reported photon reductions, probability gains and squeezings are bug-free.

The Pareto *selection* (which genotypes, 5 per target) is reproduced exactly via
`pareto_report`'s functions, so the functional Pareto data is not altered.

Run on the cluster node (needs jax + thewalrus; the repertoires under
`experiments/`).  Read-only w.r.t. the experiment data.

    cd <repo-root>
    python scripts/gen_hanamura_data.py --out ../mgr/scripts -n 5

Outputs (into --out, default <repo>/outputs):
    chosen_genotypes.npz          # selected genotypes + configs + per-row metadata
    wigner_pareto_data.json       # per-target rows (refreshed Hanamura + gbs_sq cols)
    wigner_pareto_pairs.npz       # {tgt}_{i}_psi_before / _psi_after  state vectors
    wigner_pareto_pairs_meta.json # {tgt}_{i} -> Nc_before/after, prob_gain, ...
    hanamura_table.csv            # one row per (target, pareto-row): the tab:hanamura data

Then render with the thesis-style plotter (numpy/scipy/matplotlib only):
    python ../mgr/scripts/gen_wigner_pareto.py
and update parts/4chapter.tex tab:hanamura from hanamura_table.csv.
"""
from __future__ import annotations
import argparse, os, sys, glob, json, math
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Pareto machinery (selection identical to the existing report) -------------- #
import pareto_report as pr

# --------------------------------------------------------------------------- #
# Canonical trio.  (alpha, beta) -> logical target; groups are matched by Bloch
# vector exactly as `pareto_report --target` does, so the selection is identical
# to what produced the current Pareto figure/table.  `00B` is the canonical
# point-homodyne / depth-3 / balanced-BS genotype.
# --------------------------------------------------------------------------- #
TRIO = {
    "plus": dict(alpha="1",         beta="1"),     # u = ( 1, 0, 0)
    "H":    dict(alpha="1.4142",    beta="1+1j"),  # u = (0.707, 0.707, 0)
    "T":    dict(alpha="2.7320508", beta="1+1j"),  # u = (0.577, 0.577, 0.577)
}
TARGET_ORDER = ["plus", "H", "T"]


# --------------------------------------------------------------------------- #
# AFTER state via the architecture rule: apply the Step-1 reduction symplectic
# (and its displacement) to the FULL equivalent-GBS covariance, then herald.
# This mirrors gbs_optimizer.reduce_control_mode, but embeds the single-mode
# symplectic at the control mode's index in the FULL (signal+control) space so
# the single signal mode is carried through untouched.
# --------------------------------------------------------------------------- #
def reduced_full_state(eq, n0, n1, cutoff):
    """Return (psi_after, prob_after) heralded from the reduced FULL generator."""
    from frontend.gbs_optimizer import (
        control_parameters, _reduced_params, block_from_params,
        _embed_single_mode_symplectic, heralded_output,
    )
    cov = np.asarray(eq["cov"], float).copy()
    mu = np.asarray(eq["mu"], float).copy()
    N = cov.shape[0] // 2
    signal_idx = int(eq["signal_idx"])
    control_idx = [int(c) for c in eq["control_idx"]]

    for j, ci in enumerate(control_idx):
        n_m, np_m = int(n0[j]), int(n1[j])
        if np_m >= n_m:
            continue  # no reduction on this control mode
        idx = [ci, ci + N]
        Cm = cov[np.ix_(idx, idx)]
        beta_m = mu[idx]
        p = control_parameters(Cm, beta_m)
        s0, delta0, O, nu = p["s0"], p["delta0"], p["O"], p["nu"]
        s0p, delta0p, _kk, _dd = _reduced_params(s0, delta0, n_m, np_m)
        _, _, cprime, dprime = block_from_params(s0p, delta0p, nu, O)
        # single-mode squeeze preserving the symplectic eigenvalue nu
        Dsq = np.diag([np.sqrt(cprime / p["c"]), np.sqrt(dprime / p["d"])])
        S2 = O.T @ Dsq @ O
        Semb = _embed_single_mode_symplectic(S2, ci, N)   # embed in FULL space
        cov = Semb @ cov @ Semb.T
        cov = 0.5 * (cov + cov.T)
        mu = Semb @ mu
        # displacement to hit the reduced target control-block mean
        _, beta_target, _, _ = block_from_params(s0p, delta0p, nu, O)
        mu[ci] += beta_target[0] - mu[ci]
        mu[ci + N] += beta_target[1] - mu[ci + N]

    psi, prob = heralded_output(cov, mu, signal_idx, control_idx, n1, cutoff=cutoff)
    return np.asarray(psi).ravel(), float(prob)


def core_state(s0, delta0, n, cutoff):
    """Hanamura core state  |psi> ~ (a^dag + s0 a + delta0)^n |0>  in the Fock
    basis (PRX 16, 021034, Eq. for the particle picture; thesis eq:hanamura-particle).
    This is the heralded output of a single-control-mode generator UP TO a Gaussian
    unitary, built directly from the control parameters -- no heralding, so it is
    well-conditioned even when the physical generator is highly squeezed."""
    c = int(cutoff)
    a = np.diag(np.sqrt(np.arange(1, c)), k=1)          # annihilation
    ad = a.conj().T                                     # creation
    O = ad + complex(s0) * a + complex(delta0) * np.eye(c)
    psi = np.zeros(c, dtype=complex); psi[0] = 1.0
    for _ in range(int(n)):
        psi = O @ psi
    nrm = np.linalg.norm(psi)
    return (psi / nrm) if nrm > 0 else psi


def _norm_ok(v, max_len=80):
    v = np.asarray(v).ravel()
    return v is not None and 0 < len(v) <= max_len and np.isfinite(v).all() \
        and np.linalg.norm(v) > 0.5


def _pad(v, length):
    v = np.asarray(v, dtype=complex).ravel()
    if len(v) >= length:
        return v[:length]
    out = np.zeros(length, dtype=complex)
    out[:len(v)] = v
    return out


# --------------------------------------------------------------------------- #
def match_groups(root, alpha, beta):
    """Reproduce pareto_report's --target group matching (Bloch-vector match)."""
    u = pr.alpha_beta_to_u(complex(alpha), pr.parse_complex(beta))
    groups = []
    for gdir in sorted(glob.glob(os.path.join(root, "*"))):
        cfgf = next(iter(glob.glob(os.path.join(gdir, "*", "config.json"))), None)
        if not cfgf:
            continue
        try:
            cfg = json.load(open(cfgf))
            gu = pr.alpha_beta_to_u(complex(cfg["target_alpha"]),
                                    pr.parse_complex(str(cfg["target_beta"])))
        except Exception:
            continue
        if max(abs(gu[i] - u[i]) for i in range(3)) < 1e-2:
            groups.append(gdir)
    return sorted(set(groups)), u


def _match_point(pts, row):
    """Pin an existing Pareto-figure row back to its exact genotype using the
    full-precision success probability (+ <O> + Nc as tie-breakers).  This reuses
    the identical solutions behind the validated Pareto figure/table, so the
    functional Pareto data is preserved -- we only recompute Hanamura + states."""
    rp = float(row["prob"]); ro = row.get("O"); rNc = int(row["Nc"])
    cand = [p for p in pts if rp > 0 and abs(p["prob"] - rp) <= 1e-9 * max(rp, 1e-30)]
    if not cand:
        cand = [p for p in pts if rp > 0 and abs(p["prob"] - rp) <= 1e-6 * max(rp, 1e-30)]
    if ro is not None:
        c2 = [p for p in cand if abs(p["exp"] - float(ro)) <= 5e-4]
        if c2:
            cand = c2
    c3 = [p for p in cand if round(p["photons"]) == rNc]
    if c3:
        cand = c3
    cand.sort(key=lambda p: (p["run"], p["exp"]))   # deterministic tie-break
    return cand[0] if cand else None


def selected_points(tgt, groups, u, B, n_rep, select_from):
    """Return the list of Pareto points to analyse for this target.

    Default (``select_from`` given): pin the rows of the existing
    wigner_pareto_data.json to their genotypes (preserves the Pareto selection).
    Otherwise: re-derive the selection from scratch (front may have grown)."""
    pts, _ = pr.collect_points(groups)
    herald = [p for p in pts if round(p["photons"]) >= 1]
    nd = pr.nondominated(herald)
    for p in nd:
        p["nls_db"] = -10.0 * math.log10(max(p["exp"], 1e-12) / B)
    for p in pts:
        p["nls_db"] = -10.0 * math.log10(max(p["exp"], 1e-12) / B)

    if select_from and tgt in select_from:
        reps = []
        for row in select_from[tgt]["rows"]:
            mp = _match_point(pts, row)
            if mp is None:
                print(f"    [!] {tgt}: could not pin row Nc={row['Nc']} "
                      f"P={row['prob']:.3e} -- skipping")
            reps.append(mp)
        print(f"    {len(pts)} pts | reused {sum(r is not None for r in reps)}/"
              f"{len(select_from[tgt]['rows'])} existing Pareto rows")
        return [r for r in reps if r is not None]

    reps = pr.select_representatives(nd, n_rep)
    print(f"    {len(pts)} pts, {len(herald)} heralded, {len(nd)} non-dominated, "
          f"{len(reps)} reselected")
    return reps


def process_target(tgt, cfg_t, root, n_rep, reduction_factor, herald_cap, select_from):
    """Build all per-row records for one target."""
    import jax.numpy as jnp
    from src.genotypes.genotypes import get_genotype_decoder
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend import gbs_optimizer as go
    from frontend import utils as futils

    groups, u = match_groups(root, cfg_t["alpha"], cfg_t["beta"])
    if not groups:
        print(f"[!] {tgt}: no matching experiment groups under {root}")
        return None
    print(f"[+] {tgt}: u=({u[0]:.3f},{u[1]:.3f},{u[2]:.3f})  groups: "
          f"{', '.join(os.path.basename(g) for g in groups)}")

    B = pr.gaussian_bound(u)
    reps = selected_points(tgt, groups, u, B, n_rep, select_from)

    rows, pairs, meta, chosen = [], {}, {}, []
    for i, p in enumerate(reps):
        c = p["config"]
        depth = int(c.get("depth") or 3)
        cutoff = int(c.get("cutoff") or 30)
        herald_cutoff = int(min(herald_cap, max(cutoff, 2 * int(round(p["photons"])) + 8)))

        dec = get_genotype_decoder(p["gname"], depth=depth, config=c)
        params = pr.to_numpy(dec.decode(jnp.asarray(np.asarray(p["g"], np.float32)), cutoff))
        eq = compute_equivalent_gaussian(params)
        n0 = [int(x) for x in eq["pnr_outcomes"]]
        Nc_before = int(sum(n0))
        gbs_sq_db = round(float(eq["max_squeezing_db"]), 2)

        # ---- Hanamura optimisation (moment space; unaffected by the bug) ----
        han = dict(han_ok=False, han_Nc=Nc_before, han_prob=None, han_gain=None,
                   han_sq_db=None, han_error="")
        n1 = list(n0)
        han_res = None
        k_ctrl = len(eq["control_idx"])
        try:
            han_res = go.optimize_gbs_architecture(
                eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"], n0,
                reduction_factor=reduction_factor, original_probability=p["prob"],
                verify=False, herald_cutoff=herald_cutoff)
            n1 = [int(x) for x in han_res["n_after"]]
            valid = bool(han_res.get("damping", {}).get("max_squeezing_db", np.inf) < 1e3) \
                and han_res.get("prob_after") is not None \
                and np.isfinite(han_res.get("prob_after") or np.nan)
            han.update(
                han_ok=valid,
                han_Nc=int(han_res["total_photons_after"]),
                han_prob=(float(han_res["prob_after"]) if han_res["prob_after"] is not None else None),
                han_gain=(float(han_res["prob_gain"]) if han_res["prob_gain"] is not None else None),
                han_sq_db=round(float(han_res["architecture"].get("max_squeezing_db", float("nan"))), 2),
            )
        except Exception as e:
            han["han_error"] = repr(e)[:160]
            print(f"    [{i}] Hanamura failed: {han['han_error']}")

        # ---- BEFORE state: trusted path-1 breeding simulation ---------------
        psi_before = None
        try:
            pb, _ = futils.compute_heralded_state(params, cutoff=herald_cutoff)
            if _norm_ok(pb):
                psi_before = np.asarray(pb).ravel()
        except Exception as e:
            print(f"    [{i}] before (path-1) failed: {e!r}")

        # ---- core-state reconstruction (paper-faithful, well-conditioned) ----
        # The heralded output of a single-control-mode generator is, up to a
        # Gaussian unitary, the core state (a^dag + s0 a + delta0)^n |0>.  We build
        # it directly from the control parameters -- no thewalrus herald -- so it is
        # immune to the high-squeezing ill-conditioning.  We validate it against the
        # trusted path-1 "before"; the same construction with the reduced/damped
        # parameters gives the "after".
        psi_before_core = psi_after_core = None
        core_fid = None
        s0_b = d0_b = s0_a = d0_a = None
        if han_res is not None and k_ctrl == 1:
            pb0 = han_res["params_before"][0]
            pa0 = han_res["params_after"][0]   # post Step1+Step2; damping preserves (s0,delta0)
            s0_b, d0_b = float(pb0["s0"]), complex(pb0["delta0"])
            s0_a, d0_a = float(pa0["s0"]), complex(pa0["delta0"])
            psi_before_core = core_state(s0_b, d0_b, n0[0], herald_cutoff)
            psi_after_core = core_state(s0_a, d0_a, n1[0], herald_cutoff)
            if psi_before is not None:
                try:
                    core_fid, _ = go.align_states(psi_before, psi_before_core,
                                                  herald_cutoff, align_cut=min(herald_cutoff, 36))
                except Exception:
                    core_fid = None

        # ---- AFTER state for the figure --------------------------------------
        # Prefer the core-state reconstruction (k=1); fall back to the
        # architecture-rule herald otherwise (flagged via core_ok in meta).
        psi_after = None
        core_ok = (psi_after_core is not None
                   and (core_fid is None or core_fid > 0.9))
        if core_ok:
            psi_after = psi_after_core
        else:
            try:
                pa, _ = reduced_full_state(eq, n0, n1, cutoff=herald_cutoff)
                if _norm_ok(pa):
                    psi_after = pa
            except Exception as e:
                print(f"    [{i}] after (architecture-rule fallback) failed: {e!r}")

        # ---- store --------------------------------------------------------- #
        row = dict(nls_db=round(p["nls_db"], 2), O=round(p["exp"], 4),
                   prob=float(p["prob"]), Nc=int(round(p["photons"])),
                   leaves=int(round(p["active"])), maxpnr=int(round(p["max_pnr"])),
                   gbs_sq_db=gbs_sq_db, run=p["run"], gname=p["gname"], **han)
        rows.append(row)

        key = f"{tgt}_{i}"
        L = herald_cutoff
        if psi_before is not None:
            pairs[f"{key}_psi_before"] = _pad(psi_before, L)
        if psi_after is not None:
            pairs[f"{key}_psi_after"] = _pad(psi_after, L)
        if psi_before_core is not None:
            pairs[f"{key}_psi_before_core"] = _pad(psi_before_core, L)
        meta[key] = dict(
            Nc=int(round(p["photons"])), nls_db=round(p["nls_db"], 2),
            prob=float(p["prob"]), gname=p["gname"],
            Nc_before=Nc_before, Nc_after=int(han["han_Nc"]),
            prob_gain=han["han_gain"],
            gbs_sq_db=gbs_sq_db, han_sq_db=han["han_sq_db"], han_ok=han["han_ok"],
            k_control=k_ctrl, after_source=("core" if core_ok else "herald_fallback"),
            core_validation_fid=(round(core_fid, 4) if core_fid is not None else None),
            s0_before=s0_b, delta0_before=(abs(d0_b) if d0_b is not None else None),
            s0_after=s0_a, delta0_after=(abs(d0_a) if d0_a is not None else None),
            psi_before_len=int(len(psi_before)) if psi_before is not None else 0,
            psi_after_len=int(len(psi_after)) if psi_after is not None else 0,
        )
        chosen.append(dict(target=tgt, row=i, genotype=np.asarray(p["g"], np.float32),
                           gname=p["gname"], config=c, run=p["run"],
                           Nc=int(round(p["photons"])), nls_db=round(p["nls_db"], 2),
                           prob=float(p["prob"])))
        cf = f"{core_fid:.3f}" if core_fid is not None else "--"
        print(f"    [{i}] xi={row['nls_db']:.2f}dB Nc={row['Nc']} P={row['prob']:.2e} "
              f"sq={gbs_sq_db:.1f}dB k={k_ctrl} | Hanamura Nc {Nc_before}->{han['han_Nc']} "
              f"gain={('x%.2f'%han['han_gain']) if han['han_gain'] else '--'} "
              f"sq'={han['han_sq_db']} | before={'Y' if psi_before is not None else '-'} "
              f"after={'Y(%s)'%('core' if core_ok else 'fb') if psi_after is not None else '-'} "
              f"core_fid={cf}")

    return dict(u=list(u), B_G=float(B), rows=rows), pairs, meta, chosen


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO, "experiments"),
                    help="experiments root (default <repo>/experiments)")
    ap.add_argument("--out", default=os.path.join(REPO, "outputs"),
                    help="output directory (default <repo>/outputs; pass ../mgr/scripts "
                         "to drop files where gen_wigner_pareto.py reads them)")
    ap.add_argument("-n", "--num", type=int, default=5,
                    help="representative Pareto solutions per target when reselecting")
    ap.add_argument("--reduction-factor", type=float, default=3.0)
    ap.add_argument("--herald-cap", type=int, default=48,
                    help="max Fock cutoff used for state reconstruction")
    ap.add_argument("--select-from", default=None,
                    help="existing wigner_pareto_data.json whose rows pin the "
                         "genotypes to reuse (preserves the Pareto selection). "
                         "Default: the bundled scripts/data/hanamura_selection_spec.json, "
                         "else the thesis copy next to gen_wigner_pareto.py.")
    ap.add_argument("--reselect", action="store_true",
                    help="ignore --select-from and re-derive the Pareto selection "
                         "from scratch (the front may have grown since).")
    args = ap.parse_args()

    # resolve the selection spec: explicit flag, then bundled copy, then mgr sibling
    spec_path = args.select_from
    if spec_path is None:
        for cand in (os.path.join(REPO, "scripts", "data", "hanamura_selection_spec.json"),
                     os.path.join(REPO, "..", "mgr", "scripts", "wigner_pareto_data.json")):
            if os.path.exists(cand):
                spec_path = cand
                break
    select_from = None
    if not args.reselect and spec_path and os.path.exists(spec_path):
        select_from = json.load(open(spec_path))
        print(f"[+] reusing Pareto selection from {spec_path}")
    elif not args.reselect:
        print(f"[!] selection spec not found; reselecting from scratch")

    os.makedirs(args.out, exist_ok=True)
    DATA, PAIRS, META, CHOSEN = {}, {}, {}, []
    for tgt in TARGET_ORDER:
        res = process_target(tgt, TRIO[tgt], args.root, args.num,
                             args.reduction_factor, args.herald_cap, select_from)
        if res is None:
            continue
        data_t, pairs_t, meta_t, chosen_t = res
        DATA[tgt] = data_t
        PAIRS.update(pairs_t)
        META.update(meta_t)
        CHOSEN.extend(chosen_t)

    # ---- write artifacts --------------------------------------------------- #
    json.dump(DATA, open(os.path.join(args.out, "wigner_pareto_data.json"), "w"),
              indent=2, default=float)
    np.savez_compressed(os.path.join(args.out, "wigner_pareto_pairs.npz"), **PAIRS)
    json.dump(META, open(os.path.join(args.out, "wigner_pareto_pairs_meta.json"), "w"),
              indent=2, default=float)

    # chosen_genotypes.npz : flat arrays + a json sidecar of the metadata
    geno_arrs = {f"{c['target']}_{c['row']}_g": c["genotype"] for c in CHOSEN}
    np.savez_compressed(os.path.join(args.out, "chosen_genotypes.npz"), **geno_arrs)
    json.dump([{k: v for k, v in c.items() if k != "genotype"} for c in CHOSEN],
              open(os.path.join(args.out, "chosen_genotypes_meta.json"), "w"),
              indent=2, default=float)

    # hanamura table CSV (one row per (target, pareto-row), aligned to tab:breeding_pareto)
    cols = ["target", "row", "Nc", "han_Nc", "prob", "han_prob", "han_gain",
            "gbs_sq_db", "han_sq_db", "han_ok"]
    with open(os.path.join(args.out, "hanamura_table.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for tgt in TARGET_ORDER:
            if tgt not in DATA:
                continue
            for i, r in enumerate(DATA[tgt]["rows"]):
                f.write(",".join(str(x) for x in [
                    tgt, i, r["Nc"], r["han_Nc"], f"{r['prob']:.3e}",
                    (f"{r['han_prob']:.3e}" if r["han_prob"] else ""),
                    (f"{r['han_gain']:.4g}" if r["han_gain"] else ""),
                    r["gbs_sq_db"], r["han_sq_db"], r["han_ok"]]) + "\n")

    print(f"\n[+] wrote artifacts to {args.out}:")
    for fn in ("wigner_pareto_data.json", "wigner_pareto_pairs.npz",
               "wigner_pareto_pairs_meta.json", "chosen_genotypes.npz",
               "chosen_genotypes_meta.json", "hanamura_table.csv"):
        print("    ", fn)


if __name__ == "__main__":
    main()
