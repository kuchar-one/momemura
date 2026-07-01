#!/usr/bin/env python3
"""run_hanamura_pareto.py -- run the Hanamura GBS optimization on EVERY
artifact-free Pareto-front state from the rescore (``recompute/pareto_fronts/``)
and emit a COMPLETE before/after comparison per state:

  * photon detections        n0 (before)  ->  n1 (after)         + totals
  * squeezing                per-mode r / dB, before and after (max + full list)
  * the full Gaussian        passive interferometer U + per-mode displacements,
                             before and after (the "what Gaussian" the paper asks for)
  * success probability      P_before -> P_after, gain = P_after / P_before
  * pre/post Wigner          the heralded signal state's Wigner function, before
                             vs after, with negative-volume for each (the
                             non-Gaussian resource, by Hudson)

The Pareto CSVs carry provenance (root/group/run/cell_idx) but not genotypes, so
each state is re-derived from its repertoire, decoded, reduced to its equivalent
GBS generator, and fed to ``frontend.gbs_optimizer.optimize_gbs_architecture``
(the same call the thesis Hanamura regenerator uses). Everything here is
qutip-free (uses ``reduced_herald`` for states and a displaced-parity Wigner).

Outputs (under --out, default ``hanamura/``):
  <group>/<run>_cell<idx>.json           full architecture comparison (one state)
  <group>/<run>_cell<idx>_wigner.png     before | after Wigner panels
  <group>/states/<run>_cell<idx>.npz     psi_before / psi_after vectors
  hanamura_summary.csv                   one row per state (key numbers)
  REPORT.md                              per-group + overall summary

Run on a box that has the repertoires (cluster or a checkout next to experiments/):
  JAX_ENABLE_X64=1 python scripts/run_hanamura_pareto.py            # all Pareto states
  ... --groups 'B30_*' --top 20 --wigner-grid 61                    # subset / faster
  ... --max-squeezing-db 12                                         # feasibility-capped
"""
import os, sys, glob, json, argparse, fnmatch, traceback
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# --------------------------------------------------------------------------- #
# Wigner via displaced parity (qutip-free):  W(z) = (2/pi) <psi| D(z) Π D(-z) |psi>
# --------------------------------------------------------------------------- #
def wigner_grid(psi, grid=61, span=5.0, ncut=None):
    from scipy.sparse.linalg import expm_multiply
    from scipy.sparse import diags
    psi = np.asarray(psi, complex).ravel()
    if ncut:
        psi = psi[:ncut]
    nrm = np.linalg.norm(psi)
    psi = psi / (nrm + 1e-30)
    N = len(psi)
    a = diags(np.sqrt(np.arange(1, N)), 1)
    ad = diags(np.sqrt(np.arange(1, N)), -1)
    sign = (-1.0) ** np.arange(N)
    xs = np.linspace(-span, span, grid)
    W = np.zeros((grid, grid))
    for ix, x in enumerate(xs):
        for ip, pp in enumerate(xs):
            z = (x + 1j * pp) / np.sqrt(2.0)
            phi = expm_multiply(-(z * ad - np.conj(z) * a), psi)
            W[ip, ix] = (2.0 / np.pi) * np.real(np.sum(sign * np.abs(phi) ** 2))
    return xs, W


def neg_volume(xs, W):
    dx = xs[1] - xs[0]
    return float(np.sum(np.abs(W[W < 0])) * dx * dx)


def core_state(s0, delta0, n, cutoff):
    """Hanamura core state (a^dag + s0 a + delta0)^n |0> in the Fock basis --
    the single-control-mode heralded output up to a Gaussian unitary (PRX 16,
    021034). Well-conditioned even for highly-squeezed generators."""
    c = int(cutoff)
    a = np.diag(np.sqrt(np.arange(1, c)), k=1)
    ad = a.conj().T
    O = ad + complex(s0) * a + complex(delta0) * np.eye(c)
    psi = np.zeros(c, dtype=complex); psi[0] = 1.0
    for _ in range(int(n)):
        psi = O @ psi
    nrm = np.linalg.norm(psi)
    return (psi / nrm) if nrm > 0 else psi


def to_numpy(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = to_numpy(v)
        else:
            try:
                out[k] = np.asarray(v)
            except Exception:
                out[k] = v
    return out


def _arch_json(arch):
    """JSON-safe view of a decompose_architecture / optimized-architecture dict."""
    if arch is None:
        return None
    U = arch.get("U_passive")
    Uj = None
    if U is not None:
        U = np.asarray(U)
        Uj = [[[float(np.real(z)), float(np.imag(z))] for z in row] for row in U]
    return {
        "squeezings_r": [float(x) for x in arch.get("squeezings_r", [])],
        "squeezings_db": [float(x) for x in arch.get("squeezings_db", [])],
        "max_squeezing_db": float(arch.get("max_squeezing_db", float("nan"))),
        "displacements_xp": [[float(a), float(b)] for (a, b) in arch.get("displacements", [])],
        "U_passive_reim": Uj,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pareto-dir", default=os.path.join(REPO, "recompute", "pareto_fronts"))
    ap.add_argument("--groups", default="*", help="group-name glob filter")
    ap.add_argument("--out", default=os.path.join(REPO, "hanamura"))
    ap.add_argument("--reduction-factor", type=float, default=3.0)
    ap.add_argument("--herald-cap", type=int, default=60)
    ap.add_argument("--max-squeezing-db", type=float, default=0.0,
                    help="cap optimized squeezing (experimental feasibility); 0 = uncapped/paper-faithful")
    ap.add_argument("--top", type=int, default=0, help="limit states per group (0 = all front)")
    ap.add_argument("--wigner-grid", type=int, default=61)
    ap.add_argument("--wigner-span", type=float, default=5.0)
    ap.add_argument("--wigner-ncut", type=int, default=60)
    ap.add_argument("--no-wigner", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap total states (smoke runs)")
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import pandas as pd
    import pareto_report as pr
    from src.genotypes.genotypes import get_genotype_decoder
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend import gbs_optimizer as go
    from frontend.gbs_optimizer import reduced_herald, decompose_architecture

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    # robust loader that recovers QDax MOME repertoires without qdax (the
    # 'NEWOBJ class argument must be a type, not function' UnpicklingError) via a
    # generic-type unpickler -- the same one the rescore used, so every run in the
    # Pareto fronts is loadable here too.
    from rescore_all_experiments import load_run_arrays
    # reduced_full_state needs gbs_optimizer internals; import lazily from the
    # thesis regenerator if available, else fall back to the architecture rule.
    try:
        from gen_hanamura_data import reduced_full_state
    except Exception:
        reduced_full_state = None

    os.makedirs(args.out, exist_ok=True)
    rep_cache = {}

    def load_run(root, group, run):
        key = (root, group, run)
        if key not in rep_cache:
            base = os.path.join(REPO, root, group, run)
            try:
                gens, _fit, _des = load_run_arrays(os.path.join(base, "results.pkl"))
                cfg = json.load(open(os.path.join(base, "config.json")))
                gens = np.asarray(gens).reshape(-1, np.asarray(gens).shape[-1])
                rep_cache[key] = (gens, cfg, None)
            except Exception as e:          # cache the failure so duplicate rows don't retry
                rep_cache[key] = (None, None, e)
        gens, cfg, err = rep_cache[key]
        if err is not None:
            raise err
        return gens, cfg

    summary = []
    n_done = n_fail = 0
    for csvf in sorted(glob.glob(os.path.join(args.pareto_dir, "*.csv"))):
        group = os.path.basename(csvf)[:-4]
        if not fnmatch.fnmatch(group, args.groups):
            continue
        df = pd.read_csv(csvf)
        # the rescore Pareto CSVs contain duplicate rows -- one physical state per
        # (root,run,cell_idx); process each once.
        df = df.drop_duplicates(subset=["root", "run", "cell_idx"]).reset_index(drop=True)
        if args.top:
            df = df.nsmallest(args.top, "exp_hi")
        gdir = os.path.join(args.out, group)
        os.makedirs(os.path.join(gdir, "states"), exist_ok=True)
        print(f"\n=== {group}: {len(df)} Pareto states ===", flush=True)
        for _, r in df.iterrows():
            if args.limit and n_done >= args.limit:
                break
            tag = f"{r['run']}_cell{int(r['cell_idx'])}"
            try:
                gens, cfg = load_run(r["root"], r["group"], r["run"])
                if int(cfg.get("modes") or 3) != 3:
                    raise ValueError("modes!=3 (static/Hanamura path is 3-modes/leaf only)")
                g = gens[int(r["cell_idx"])].astype(np.float32)
                depth = int(cfg.get("depth") or 3)
                cutoff = int(cfg.get("cutoff") or 30)
                a = complex(str(r["target_alpha"]).replace("i", "j"))
                b = complex(str(r["target_beta"]).replace("i", "j"))
                hcut = int(min(args.herald_cap, max(cutoff, 2 * int(round(float(r["total_photons"]))) + 8)))

                dec = get_genotype_decoder(cfg.get("genotype"), depth=depth, config=cfg)
                params = to_numpy(dec.decode(jnp.asarray(g), cutoff))
                eq = compute_equivalent_gaussian(params)
                n0 = [int(x) for x in eq["pnr_outcomes"]]

                # ---- BEFORE: architecture + heralded state ----
                arch_before = decompose_architecture(np.asarray(eq["cov"], float),
                                                     np.asarray(eq["mu"], float))
                psi_b, prob_b = reduced_herald(eq["cov"], eq["mu"], eq["signal_idx"],
                                               eq["control_idx"], n0, cutoff=hcut)
                psi_b = np.asarray(psi_b).ravel()

                # ---- Hanamura two-step (reduce photons + maximize prob) ----
                maxsq = None if args.max_squeezing_db <= 0 else args.max_squeezing_db
                han = go.optimize_gbs_architecture(
                    eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"], n0,
                    reduction_factor=args.reduction_factor,
                    original_probability=float(r["prob"]),
                    max_squeezing_db=maxsq, verify=False, herald_cutoff=hcut)
                n1 = [int(x) for x in han["n_after"]]
                arch_after = han.get("architecture")

                # ---- AFTER: heralded state (exact core if exactly one mode fires) ----
                k_eff_after = sum(1 for x in n1 if x >= 1)
                psi_a = None; after_src = "none"
                if k_eff_after == 1 and han.get("params_after"):
                    ma = next(j for j, x in enumerate(n1) if x >= 1)
                    pam = han["params_after"][ma]
                    psi_a = core_state(pam["s0"], pam["delta0"], int(n1[ma]), hcut)
                    after_src = "core_exact"
                elif reduced_full_state is not None:
                    try:
                        psi_a, _ = reduced_full_state(eq, n0, n1, hcut)
                        after_src = "architecture_rule"
                    except Exception:
                        psi_a = None
                if psi_a is not None:
                    psi_a = np.asarray(psi_a).ravel()

                # ---- Wigner before/after + negativity ----
                negb = nega = None
                if not args.no_wigner:
                    xs, Wb = wigner_grid(psi_b, args.wigner_grid, args.wigner_span, args.wigner_ncut)
                    negb = neg_volume(xs, Wb)
                    Wa = None
                    if psi_a is not None:
                        _, Wa = wigner_grid(psi_a, args.wigner_grid, args.wigner_span, args.wigner_ncut)
                        nega = neg_volume(xs, Wa)
                    _save_wigner_png(os.path.join(gdir, f"{tag}_wigner.png"), xs, Wb, Wa,
                                     f"{group}  {tag}", n0, n1)

                np.savez(os.path.join(gdir, "states", f"{tag}.npz"),
                         psi_before=psi_b, psi_after=(psi_a if psi_a is not None else np.zeros(0)))

                rec = {
                    "group": group, "target_alpha": str(a), "target_beta": str(b),
                    "design": cfg.get("genotype"), "depth": depth,
                    "provenance": f"{r['root']}/{group}/{r['run']}#{int(r['cell_idx'])}",
                    "exp_O": float(r["exp_hi"]), "vs_gaussian": float(r["vs_gaussian"]),
                    "before": {
                        "pnr_detections": n0, "total_photons": int(sum(n0)),
                        "prob": float(prob_b), "architecture": _arch_json(arch_before),
                        "wigner_negvol": negb,
                    },
                    "after": {
                        "pnr_detections": n1, "total_photons": int(sum(n1)),
                        "prob": (float(han["prob_after"]) if han.get("prob_after") is not None else None),
                        "prob_gain": (float(han["prob_gain"]) if han.get("prob_gain") is not None else None),
                        "architecture": _arch_json(arch_after),
                        "state_source": after_src, "wigner_negvol": nega,
                    },
                    "k_eff_before": int(sum(1 for x in n0 if x >= 1)),
                    "k_eff_after": k_eff_after,
                    "herald_cutoff": hcut,
                }
                json.dump(rec, open(os.path.join(gdir, f"{tag}.json"), "w"), indent=2)

                summary.append({
                    "group": group, "design": cfg.get("genotype"), "depth": depth,
                    "exp_O": float(r["exp_hi"]), "vs_gaussian": float(r["vs_gaussian"]),
                    "Nc_before": int(sum(n0)), "Nc_after": int(sum(n1)),
                    "prob_before": float(prob_b),
                    "prob_after": (float(han["prob_after"]) if han.get("prob_after") is not None else np.nan),
                    "prob_gain": (float(han["prob_gain"]) if han.get("prob_gain") is not None else np.nan),
                    "sqdb_before": float(arch_before["max_squeezing_db"]),
                    "sqdb_after": (float(arch_after["max_squeezing_db"]) if arch_after else np.nan),
                    "negvol_before": negb, "negvol_after": nega,
                    # han_valid=False marks the ill-conditioned high-squeezing
                    # generators (Nc->Nc' reduction hits sqrt(c*d-1)<0), where the
                    # optimized prob is nan -- expected per the Hanamura runbook.
                    "han_valid": bool(han.get("prob_after") is not None
                                      and np.isfinite(han.get("prob_after") or np.nan)),
                    "after_source": after_src, "provenance": rec["provenance"],
                })
                n_done += 1
                print(f"  [{n_done}] {tag}: Nc {sum(n0)}->{sum(n1)}  "
                      f"P {prob_b:.2e}->{(han.get('prob_after') or float('nan')):.2e}  "
                      f"gain x{(han.get('prob_gain') or float('nan')):.2f}  "
                      f"sqdB {arch_before['max_squeezing_db']:.1f}->"
                      f"{(arch_after['max_squeezing_db'] if arch_after else float('nan')):.1f}", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"  [skip] {group}/{tag}: {e!r}", flush=True)
                if os.environ.get("HAN_DEBUG"):
                    traceback.print_exc()
        if args.limit and n_done >= args.limit:
            break

    import pandas as pd
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(args.out, "hanamura_summary.csv"), index=False)
    _write_report(args.out, sdf, n_done, n_fail)
    print(f"\nDONE: {n_done} states, {n_fail} skipped. -> {args.out}")


def _save_wigner_png(path, xs, Wb, Wa, title, n0, n1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 2 if Wa is not None else 1
    fig, axs = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.6), squeeze=False)
    vmax = max(abs(Wb).max(), (abs(Wa).max() if Wa is not None else 0)) or 1.0
    ext = [xs[0], xs[-1], xs[0], xs[-1]]
    axs[0][0].imshow(Wb, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axs[0][0].set_title(f"before  (det {sum(n0)} ph)")
    if Wa is not None:
        axs[0][1].imshow(Wa, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axs[0][1].set_title(f"after Hanamura  (det {sum(n1)} ph)")
    fig.suptitle(title, fontsize=9)
    for ax in axs[0]:
        ax.set_xlabel("x"); ax.set_ylabel("p")
    fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def _write_report(out, sdf, n_done, n_fail):
    lines = ["# Hanamura optimization over the artifact-free Pareto fronts\n",
             f"_states processed: {n_done} | skipped: {n_fail}_\n",
             "Each state has a full before/after architecture in `<group>/<run>_cell<idx>.json` "
             "(photon detections, per-mode squeezing dB, passive interferometer U, displacements, "
             "success probability) and a before|after Wigner panel `*_wigner.png`. "
             "`hanamura_summary.csv` is the flat table.\n"]
    if len(sdf):
        lines.append("## Best probability-gain per group\n")
        lines.append("| group | design | ⟨O⟩ | Nc before→after | P before→after | gain | sqdB before→after | negvol b→a |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for grp, sub in sdf.groupby("group"):
            valid = sub[np.isfinite(sub["prob_gain"])]
            b = (valid.loc[valid["prob_gain"].idxmax()] if len(valid)
                 else sub.loc[sub["exp_O"].idxmin()])
            lines.append(
                f"| {grp} | {b['design']} | {b['exp_O']:.3f} | {int(b['Nc_before'])}→{int(b['Nc_after'])} | "
                f"{b['prob_before']:.1e}→{b['prob_after']:.1e} | "
                f"x{b['prob_gain']:.2f} | {b['sqdb_before']:.1f}→{b['sqdb_after']:.1f} | "
                f"{b['negvol_before']}→{b['negvol_after']} |")
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
