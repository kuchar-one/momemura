#!/usr/bin/env python3
"""
pareto_report.py -- extract and analyse Pareto-optimal breeding solutions.

Given one or more experiment groups (the ``experiments/<genotype>_c<cutoff>_a..b..``
directories produced by ``run_mome.py``), this tool:

  1. loads every run, aggregates a global Pareto front over
     (nonlinear squeezing <O>, success probability), and keeps only genuine
     heralded solutions (detected photon number N_c >= 1);
  2. selects a handful of representative non-dominated solutions spanning the
     front (cheapest, best-quality, and an even spread in between);
  3. for each, decodes the genotype, reduces it to its equivalent GBS generator
     (vacuum -> Gaussian unitary -> PNR pattern + 1 heralded mode) and prints the
     GBS-deconstructed resources: max squeezing of the generator, number of PNR
     detectors, detected photon number, success probability, and the nonlinear
     squeezing  xi[dB] = -10 log10(<O> / B_G), with B_G = 5/3 - ||u||_inf;
  4. runs the Hanamura et al. two-step control-parameter optimisation
     (photon-number reduction + success-probability maximisation) on each
     generator and reports the before/after photon count, probability and
     squeezing;
  5. optionally renders the Wigner function of each heralded output (needs QuTiP).

Outputs a Markdown report, a CSV of all selected solutions, and (optionally)
Wigner PNGs into ``--out``.  Read-only with respect to the experiment data.

Examples
--------
    # the logical |+> runs (alpha=1, beta=1)
    python pareto_report.py --target 1,1 --label plus -n 3

    # a magic state by explicit (alpha, beta); beta may be complex, e.g. 1+1j
    python pareto_report.py --target 1.4142,1+1j --label H_magic -n 4

    # or point straight at one or more group directories
    python pareto_report.py --group experiments/B30B_c30_a1p00_b1p00 --label plus

Run it from the repository root (it adds that root to ``sys.path``) inside the
project's virtual environment (needs jax + thewalrus; QuTiP only for Wigners).
"""
from __future__ import annotations
import argparse, os, sys, json, glob, math, pickle
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

DB_PER_R = 10.0 / math.log(10.0)  # 8.6859: squeezing[dB] = DB_PER_R * r


# --------------------------------------------------------------------------- #
# tolerant repertoire loading (works for SimpleRepertoire without qdax;        #
# uses the real qdax classes for MOMERepertoire when they are importable)      #
# --------------------------------------------------------------------------- #
class _SimpleRepertoire:  # plain attribute holder for pickle BUILD
    pass


def _absorb(*_a, **_k):
    return None


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "SimpleRepertoire":
            return _SimpleRepertoire
        try:
            return super().find_class(module, name)
        except Exception:
            # qdax/flax not installed: we only need the array attributes, so
            # absorbing the missing class is fine.
            return _absorb


def load_repertoire(pkl_path):
    with open(pkl_path, "rb") as f:
        data = _SafeUnpickler(f).load()
    rep = data.get("repertoire") if isinstance(data, dict) else None
    return rep


# --------------------------------------------------------------------------- #
# (alpha, beta) -> Bloch vector u and Gaussian bound B_G                        #
# --------------------------------------------------------------------------- #
def alpha_beta_to_u(alpha, beta):
    alpha, beta = complex(alpha), complex(beta)
    n = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha, beta = alpha / n, beta / n
    ph = np.angle(alpha)
    a = (alpha * np.exp(-1j * ph)).real
    b = beta * np.exp(-1j * ph)
    uz = 2 * a * a - 1.0
    uu = 2 * a * b
    return float(uu.real), float(uu.imag), float(uz)


def gaussian_bound(u):
    return 5.0 / 3.0 - max(abs(x) for x in u)


def parse_complex(s):
    return complex(s.replace("i", "j"))


# --------------------------------------------------------------------------- #
# collect points from a set of group directories                               #
# --------------------------------------------------------------------------- #
def collect_points(group_dirs):
    """Return list of point dicts and the (alpha,beta) of the first config."""
    pts, ab = [], None
    for gdir in group_dirs:
        for cfgf in glob.glob(os.path.join(gdir, "*", "config.json")):
            rundir = os.path.dirname(cfgf)
            pkl = os.path.join(rundir, "results.pkl")
            if not os.path.exists(pkl):
                continue
            with open(cfgf) as f:
                config = json.load(f)
            if ab is None:
                ab = (config.get("target_alpha"), config.get("target_beta"))
            try:
                rep = load_repertoire(pkl)
                fit = np.asarray(rep.fitnesses, dtype=np.float64)
                des = np.asarray(rep.descriptors, dtype=np.float64)
                gen = np.asarray(rep.genotypes, dtype=np.float64)
            except Exception:
                continue
            fit = fit.reshape(-1, fit.shape[-1])
            des = des.reshape(-1, des.shape[-1])
            gen = gen.reshape(-1, gen.shape[-1])
            ok = np.isfinite(fit[:, 0]) & (fit[:, 0] != -np.inf)
            for k in np.where(ok)[0]:
                exp, logp = -fit[k, 0], fit[k, 1]
                if not (np.isfinite(exp) and np.isfinite(logp)):
                    continue
                pts.append(dict(exp=float(exp), logp=float(logp),
                                prob=float(10.0 ** logp),
                                active=float(-fit[k, 2]), photons=float(-fit[k, 3]),
                                max_pnr=float(des[k, 1]) if des.shape[1] > 1 else float("nan"),
                                gname=config.get("genotype"), config=config,
                                g=gen[k], run=os.path.basename(rundir)))
    return pts, ab


def nondominated(pts):
    objs = np.array([[p["exp"], -p["prob"]] for p in pts])
    keep = []
    for j in range(len(pts)):
        dom = np.all(objs <= objs[j], axis=1) & np.any(objs < objs[j], axis=1)
        if not np.any(dom):
            keep.append(pts[j])
    return keep


def select_representatives(nd, n):
    """Pick n points spanning the front: cheapest N_c, best NLS, even NLS spread."""
    if len(nd) <= n:
        return sorted(nd, key=lambda p: (round(p["photons"]), p["exp"]))
    nd = sorted(nd, key=lambda p: p["nls_db"])
    cheap = min(nd, key=lambda p: (round(p["photons"]), -p["nls_db"]))
    best = max(nd, key=lambda p: p["nls_db"])
    chosen = [cheap, best]
    chosen_ids = {id(cheap), id(best)}
    lo, hi = cheap["nls_db"], best["nls_db"]
    for t in np.linspace(lo, hi, n)[1:-1]:
        cand = min((p for p in nd if id(p) not in chosen_ids),
                   key=lambda p: abs(p["nls_db"] - t), default=None)
        if cand is not None:
            chosen.append(cand); chosen_ids.add(id(cand))
    return sorted(chosen, key=lambda p: (round(p["photons"]), p["exp"]))[:n]


# --------------------------------------------------------------------------- #
# per-solution analysis: equivalent GBS + Hanamura optimisation                #
# --------------------------------------------------------------------------- #
def to_numpy(obj):
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return np.asarray(obj)
    if isinstance(obj, (list, tuple)):
        return [to_numpy(o) for o in obj]
    return obj


def analyse_solution(p, B, do_hanamura=True):
    import jax.numpy as jnp
    from src.genotypes.genotypes import get_genotype_decoder
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend import gbs_optimizer as go

    cfg = p["config"]
    depth = int(cfg.get("depth") or 3)
    cutoff = int(cfg.get("cutoff") or 30)
    dec = get_genotype_decoder(p["gname"], depth=depth, config=cfg)
    params = to_numpy(dec.decode(jnp.asarray(np.asarray(p["g"], np.float32)), cutoff))
    eq = compute_equivalent_gaussian(params)

    rec = dict(nls_db=round(p["nls_db"], 2), O=round(p["exp"], 4), prob=p["prob"],
               Nc=int(round(p["photons"])), k_det=len(eq["control_idx"]),
               leaves=int(round(p["active"])), max_pnr=int(round(p["max_pnr"])),
               gbs_sq_db=round(float(eq["max_squeezing_db"]), 2), geno=p["gname"],
               run=p["run"], cutoff=cutoff, _eq=eq)
    if do_hanamura:
        try:
            r = go.optimize_gbs_architecture(
                eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"],
                eq["pnr_outcomes"], reduction_factor=3.0,
                original_probability=p["prob"], verify=False, herald_cutoff=cutoff)
            rec.update(han_Nc=int(r["total_photons_after"]),
                       han_prob=float(r["prob_after"]) if r["prob_after"] else None,
                       han_gain=float(r["prob_gain"]) if r["prob_gain"] else None,
                       han_sq_db=round(float(r["architecture"].get("max_squeezing_db", float("nan"))), 2))
        except Exception as e:
            rec["han_error"] = repr(e)[:120]
    return rec


# --------------------------------------------------------------------------- #
# Wigner rendering (optional; needs QuTiP + matplotlib)                         #
# --------------------------------------------------------------------------- #
def render_wigner(rec, out_png, max_photons=14, grid=121, span=5.0):
    Nc = rec["Nc"]
    if Nc > max_photons:
        return None  # heralding too expensive / truncation-limited
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import qutip as qt
        from frontend import gbs_optimizer as go
    except Exception as e:
        return f"wigner skipped ({e!r})"
    eq = rec["_eq"]
    cutoff = max(rec["cutoff"], 2 * Nc + 8)
    try:
        psi, _ = go.heralded_output(eq["cov"], eq["mu"], eq["signal_idx"],
                                    eq["control_idx"], eq["pnr_outcomes"], cutoff=cutoff)
    except Exception as e:
        return f"herald failed ({e!r})"
    psi = np.asarray(psi).ravel()
    if not np.isfinite(psi).all() or np.linalg.norm(psi) == 0:
        return "herald produced an invalid state"
    ket = qt.Qobj(psi.reshape(-1, 1))
    xvec = np.linspace(-span, span, grid)
    W = qt.wigner(ket, xvec, xvec)
    wlim = np.max(np.abs(W))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.contourf(xvec, xvec, W, 80, cmap="RdBu_r", vmin=-wlim, vmax=wlim)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("p")
    ax.set_title(f"{rec['geno']}  Nc={Nc}  xi={rec['nls_db']:.1f} dB")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Extract & analyse Pareto-optimal breeding solutions.")
    ap.add_argument("--root", default=os.path.join(REPO, "experiments"),
                    help="experiments root (default: <repo>/experiments)")
    ap.add_argument("--group", action="append", default=[],
                    help="explicit group directory (repeatable)")
    ap.add_argument("--target", help="alpha,beta of the logical target, e.g. '1,1' or '1.4142,1+1j'")
    ap.add_argument("--label", default="report", help="name for the output files")
    ap.add_argument("-n", "--num", type=int, default=3, help="representative solutions")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--no-hanamura", action="store_true")
    ap.add_argument("--no-wigner", action="store_true")
    ap.add_argument("--wigner-max-photons", type=int, default=14)
    args = ap.parse_args()

    # resolve target groups
    groups = [g if os.path.isabs(g) else os.path.join(REPO, g) for g in args.group]
    if args.target:
        a_str, b_str = args.target.split(",")
        u = alpha_beta_to_u(complex(a_str), parse_complex(b_str))
        target_u = u
        for gdir in sorted(glob.glob(os.path.join(args.root, "*"))):
            cfgf = next(iter(glob.glob(os.path.join(gdir, "*", "config.json"))), None)
            if not cfgf:
                continue
            cfg = json.load(open(cfgf))
            try:
                gu = alpha_beta_to_u(complex(cfg["target_alpha"]), parse_complex(str(cfg["target_beta"])))
            except Exception:
                continue
            if max(abs(gu[i] - u[i]) for i in range(3)) < 1e-2:
                groups.append(gdir)
    else:
        target_u = None
    groups = sorted(set(groups))
    if not groups:
        sys.exit("No matching experiment groups found. Use --group or --target.")

    print(f"[+] {len(groups)} group(s):")
    for g in groups:
        print("    ", os.path.relpath(g, REPO))

    pts, ab = collect_points(groups)
    if not pts:
        sys.exit("No valid solutions found in those groups.")
    if target_u is None:
        target_u = alpha_beta_to_u(complex(ab[0]), parse_complex(str(ab[1])))
    B = gaussian_bound(target_u)
    print(f"[+] target u = ({target_u[0]:.3f}, {target_u[1]:.3f}, {target_u[2]:.3f})  "
          f"B_G = {B:.4f}  |  {len(pts)} points")

    herald = [p for p in pts if round(p["photons"]) >= 1]
    nd = nondominated(herald)
    for p in nd:
        p["nls_db"] = -10.0 * math.log10(max(p["exp"], 1e-12) / B)
    reps = select_representatives(nd, args.num)
    print(f"[+] {len(herald)} heralded points, {len(nd)} non-dominated, {len(reps)} selected")

    out = args.out or os.path.join(REPO, "pareto_reports", args.label)
    os.makedirs(out, exist_ok=True)

    records = []
    for i, p in enumerate(reps):
        rec = analyse_solution(p, B, do_hanamura=not args.no_hanamura)
        if not args.no_wigner:
            png = os.path.join(out, f"{args.label}_sol{i}_wigner.png")
            rec["wigner"] = render_wigner(rec, png, max_photons=args.wigner_max_photons)
        records.append(rec)
        han = ""
        if "han_Nc" in rec:
            han = (f" | Hanamura: Nc {rec['Nc']}->{rec['han_Nc']}, "
                   f"P x{rec['han_gain']:.1f}, sq {rec['gbs_sq_db']:.1f}->{rec['han_sq_db']:.1f} dB")
        print(f"  [{i}] xi={rec['nls_db']:.2f}dB <O>={rec['O']:.3f} P={rec['prob']:.2e} "
              f"Nc={rec['Nc']} k={rec['k_det']} sq={rec['gbs_sq_db']:.1f}dB{han}")

    # write CSV + Markdown
    cols = ["nls_db", "O", "prob", "Nc", "k_det", "leaves", "max_pnr", "gbs_sq_db",
            "geno", "han_Nc", "han_prob", "han_gain", "han_sq_db", "run", "wigner"]
    csv_path = os.path.join(out, f"{args.label}.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    md_path = os.path.join(out, f"{args.label}.md")
    with open(md_path, "w") as f:
        f.write(f"# Pareto report: {args.label}\n\n")
        f.write(f"- target Bloch vector **u = ({target_u[0]:.3f}, {target_u[1]:.3f}, {target_u[2]:.3f})**, "
                f"Gaussian bound B_G = {B:.4f}\n")
        f.write(f"- groups: {', '.join(os.path.relpath(g, REPO) for g in groups)}\n")
        f.write(f"- {len(herald)} heralded points, {len(nd)} non-dominated, {len(reps)} selected\n\n")
        f.write("| xi[dB] | <O> | P | Nc | k | leaves | sq[dB] | Hanamura Nc | P gain | sq'[dB] |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in records:
            f.write(f"| {r['nls_db']:.2f} | {r['O']:.3f} | {r['prob']:.2e} | {r['Nc']} | "
                    f"{r['k_det']} | {r['leaves']} | {r['gbs_sq_db']:.1f} | "
                    f"{r.get('han_Nc','-')} | "
                    f"{('x%.1f'%r['han_gain']) if r.get('han_gain') else '-'} | "
                    f"{r.get('han_sq_db','-')} |\n")
        wpaths = [r.get("wigner") for r in records if isinstance(r.get("wigner"), str) and r["wigner"].endswith(".png")]
        if wpaths:
            f.write("\n## Wigner functions\n\n")
            for w in wpaths:
                f.write(f"![]({os.path.basename(w)})\n\n")
    # drop the heavy _eq before any JSON dump
    for r in records:
        r.pop("_eq", None)
    with open(os.path.join(out, f"{args.label}.json"), "w") as f:
        json.dump(dict(target_u=target_u, B_G=B, groups=groups, records=records),
                  f, indent=2, default=float)
    print(f"[+] wrote {md_path}\n[+] wrote {csv_path}")


if __name__ == "__main__":
    main()
