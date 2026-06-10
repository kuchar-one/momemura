#!/usr/bin/env python3
"""
validate_pipeline.py -- end-to-end Wigner validation of the breeding -> GBS ->
Hanamura chain, for the canonical trio's Pareto solutions.

For each selected solution it reconstructs the heralded single-mode output FOUR
independent ways and compares them (numbers + Wigners):

  (A) OPTIMIZER, path-1 : utils.compute_heralded_state  (the JAX breeding sim the
                          MOME optimizer actually scored -- ground truth).
  (B) OPTIMIZER, path-2 : independent_verifier.verify_circuit  (a SEPARATE
                          thewalrus+scipy leaf-by-leaf reconstruction).  A==B
                          confirms the optimizer's own output is trustworthy.
  (C) GBS reconstruction : heralded_output on the collapsed equivalent-GBS
                          generator (compute_equivalent_gaussian -> thewalrus
                          state_vector).  C ?= A tests the GBS-equivalence claim.
  (D) HANAMURA reduced   : the reduced generator's heralded output (core state
                          when one control mode fires, else the architecture-rule
                          herald).  D ?= A (up to a Gaussian unitary) tests that
                          the Hanamura optimisation preserves the output.

Outputs (into --out):
  validate_wigners.pdf   : rows = solutions, cols = [Optimizer(A) | GBS(C) | Hanamura(D)]
  validate_report.json   : per-solution norms, n-bar, parity, effective stellar
                           rank, and all pairwise fidelities (raw + up-to-Gaussian),
                           plus cutoff dependence of A vs C (a conditioning probe).

Run on the cluster (needs jax + thewalrus + the repertoires):
  python scripts/validate_pipeline.py --out outputs/validate --max-nc 16
(--max-nc skips the very high-N_c rows whose herald is intractable/ill-conditioned;
 pass a large value or --all to include them.)
"""
from __future__ import annotations
import os, sys, json, math, argparse
import numpy as np

# float64 in the breeding sim is REQUIRED for trustworthy reconstructions
# (the cluster default is float32 and silently truncates complex128).
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import pareto_report as pr
import importlib.util as _u
_spec = _u.spec_from_file_location("ghd", os.path.join(REPO, "scripts", "gen_hanamura_data.py"))
ghd = _u.module_from_spec(_spec); _spec.loader.exec_module(ghd)

# ---- thesis Wigner style (copied from mgr/scripts/gen_wigner_pareto.py) ------ #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
from matplotlib.colorbar import ColorbarBase
from scipy.special import factorial, eval_genlaguerre
SQRT_PI = math.sqrt(math.pi)


class PlateauTwoSlopeNorm(colors.TwoSlopeNorm):
    def __init__(self, vcenter, plateau_size, vmin=None, vmax=None):
        super().__init__(vcenter=vcenter, vmin=vmin, vmax=vmax)
        self.plateau_size = plateau_size

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        pl = self.vcenter - self.plateau_size / 2
        pu = self.vcenter + self.plateau_size / 2
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, pl, pu, self.vmax], [0, 0.5, 0.5, 1],
                      left=-np.inf, right=np.inf), mask=np.ma.getmask(result))
        return np.atleast_1d(result)[0] if is_scalar else result

    def inverse(self, value):
        pl = self.vcenter - self.plateau_size / 2
        pu = self.vcenter + self.plateau_size / 2
        return np.interp(value, [0, 0.5, 0.5, 1], [self.vmin, pl, pu, self.vmax],
                         left=-np.inf, right=np.inf)


CMAP = "inferno"
NORM = PlateauTwoSlopeNorm(vcenter=0.0, plateau_size=0.03, vmin=-0.23, vmax=0.23)


def wigner_numpy(psi, xvec, yvec, tol=1e-10):
    psi = np.asarray(psi, dtype=complex).ravel()
    p2 = np.abs(psi) ** 2
    cum = np.cumsum(p2[::-1])[::-1]
    keep = int(np.searchsorted(-cum, -tol, side="left"))
    N = max(1, min(len(psi), keep + 1, 40))
    psi = psi[:N]
    X, Y = np.meshgrid(xvec, yvec)
    A = (X + 1j * Y) / math.sqrt(2.0)
    A2 = np.abs(A) ** 2
    pref = (2.0 / math.pi) * np.exp(-2.0 * A2)
    rho = np.outer(psi, np.conj(psi))
    W = np.zeros_like(X, dtype=float)
    for m in range(N):
        if abs(rho[m, m]) < 1e-12:
            continue
        W += float(rho[m, m].real) * ((-1) ** m) * eval_genlaguerre(m, 0, 4.0 * A2)
    A2c = 2.0 * np.conj(A)
    for d in range(1, N):
        Adk = A2c ** d
        for m in range(0, N - d):
            n = m + d
            r = rho[m, n]
            if abs(r) < 1e-12:
                continue
            coef = math.sqrt(factorial(m) / factorial(n))
            W += 2.0 * np.real(r * ((-1) ** m) * coef * Adk * eval_genlaguerre(m, d, 4.0 * A2))
    return pref * W


XVEC = np.linspace(-5, 5, 160)
XS = XVEC / SQRT_PI


def render(ax, psi, title):
    arr = np.asarray(psi).ravel() if psi is not None else None
    if arr is None or len(arr) <= 1 or not np.isfinite(arr).all() or np.linalg.norm(arr) < 0.5:
        ax.text(0.5, 0.5, "(unavailable)", ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
        return
    W = wigner_numpy(arr, XVEC, XVEC)
    ax.contourf(XS, XS, W, 200, cmap=CMAP, norm=NORM, zorder=-1)
    ax.set_rasterization_zorder(0); ax.set_aspect("equal"); ax.set_box_aspect(1.0)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.set_title(title, fontsize=8)


# ---- state diagnostics ------------------------------------------------------ #
def stats(psi):
    psi = np.asarray(psi).ravel()
    nrm = float(np.linalg.norm(psi))
    if nrm < 1e-9:
        return dict(norm=nrm, nbar=None, even=None)
    p = np.abs(psi) ** 2 / nrm ** 2
    return dict(norm=round(nrm, 4),
                nbar=round(float(np.sum(np.arange(len(psi)) * p)), 3),
                even=round(float(p[0::2].sum()), 3))


def fid_raw(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    L = min(len(a), len(b))
    if L == 0 or np.linalg.norm(a[:L]) < 1e-9 or np.linalg.norm(b[:L]) < 1e-9:
        return None
    a = a[:L] / np.linalg.norm(a[:L]); b = b[:L] / np.linalg.norm(b[:L])
    return float(abs(np.vdot(a, b)) ** 2)


def fid_gauss(a, b):
    """Fidelity up to a single-mode Gaussian unitary (absorbs the residual)."""
    from frontend import gbs_optimizer as go
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    L = min(len(a), len(b))
    if L == 0 or np.linalg.norm(a[:L]) < 1e-9 or np.linalg.norm(b[:L]) < 1e-9:
        return None
    try:
        f, _ = go.align_states(a[:L], b[:L], L, align_cut=min(L, 36))
        return float(f)
    except Exception:
        return None


def effective_rank(psi, max_rank=4):
    """Smallest r such that the state matches a rank-r core (up to Gaussian U) at
    fidelity > 0.97; returns (rank_or_None, best_fid_per_rank)."""
    from frontend import gbs_optimizer as go
    psi = np.asarray(psi).ravel(); L = len(psi)
    if np.linalg.norm(psi) < 1e-9:
        return None, {}
    out = {}
    for r in range(0, max_rank + 1):
        best = 0.0
        for s0 in (0.0, 0.5, 1.0, 2.0):
            for d0 in (0.0, 0.8, 1.6):
                v = ghd.core_state(s0, d0, r, L)
                try:
                    f, _ = go.align_states(psi[:L], v[:L], L, align_cut=min(L, 30))
                except Exception:
                    f = 0.0
                best = max(best, f)
        out[r] = round(best, 3)
        if best > 0.97:
            return r, out
    return None, out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO, "experiments"))
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "validate"))
    ap.add_argument("--select-from",
                    default=os.path.join(REPO, "scripts", "data", "hanamura_selection_spec.json"))
    ap.add_argument("-n", "--num", type=int, default=5)
    ap.add_argument("--cutoff", type=int, default=40)
    ap.add_argument("--skip-c", action="store_true", help="skip path C entirely")
    ap.add_argument("--cutoff2", type=int, default=0,
                    help="second path-C cutoff (conditioning probe); 0 = disabled")
    ap.add_argument("--max-nc", type=int, default=16,
                    help="skip rows with detected photons above this (herald intractable)")
    ap.add_argument("--all", action="store_true", help="include all rows regardless of N_c")
    ap.add_argument("--rank", action="store_true", help="also compute effective stellar rank (slow)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import jax.numpy as jnp
    from src.genotypes.genotypes import get_genotype_decoder
    from frontend.gaussian_decomposition import compute_equivalent_gaussian
    from frontend import gbs_optimizer as go
    from frontend import utils as futils
    from frontend.independent_verifier import verify_circuit

    select_from = json.load(open(args.select_from)) if os.path.exists(args.select_from) else None

    report = {}
    panels = []   # (label, psiA, psiC, psiD, info)
    for tgt in ghd.TARGET_ORDER:
        groups, u = ghd.match_groups(args.root, ghd.TRIO[tgt]["alpha"], ghd.TRIO[tgt]["beta"])
        if not groups:
            continue
        B = pr.gaussian_bound(u)
        reps = ghd.selected_points(tgt, groups, u, B, args.num, select_from)
        for i, p in enumerate(reps):
            Nc = int(round(p["photons"]))
            key = f"{tgt}_{i}"
            if not args.all and Nc > args.max_nc:
                report[key] = dict(skipped=f"Nc={Nc} > max_nc={args.max_nc}")
                continue
            c = p["config"]
            depth = int(c.get("depth") or 3); cutoff = int(c.get("cutoff") or 30)
            L = int(max(args.cutoff, 2 * Nc + 8))
            pnr_max = int(max(3, max(int(x) for x in [Nc]) + 1))

            dec = get_genotype_decoder(p["gname"], depth=depth, config=c)
            params = pr.to_numpy(dec.decode(jnp.asarray(np.asarray(p["g"], np.float32)), cutoff))
            eq = compute_equivalent_gaussian(params)
            n0 = [int(x) for x in eq["pnr_outcomes"]]
            pnr_max = int(max(3, (max(n0) if n0 else 0) + 1))

            rec = dict(target=tgt, row=i, Nc=Nc, nls_db=round(p["nls_db"], 2),
                       gbs_sq_db=round(float(eq["max_squeezing_db"]), 2),
                       k_control=len(eq["control_idx"]), n0=n0,
                       k_eff=int(sum(1 for x in n0 if x >= 1)))

            # (A) optimizer path-1
            psiA = None
            try:
                pa, _ = futils.compute_heralded_state(params, cutoff=L)
                psiA = np.asarray(pa).ravel()
            except Exception as e:
                rec["A_error"] = repr(e)[:120]
            # (B) optimizer path-2 (independent)
            psiB = None
            try:
                vb = verify_circuit(params, cutoff=L, pnr_max=pnr_max)
                psiB = np.asarray(vb["state"]).ravel()
            except Exception as e:
                rec["B_error"] = repr(e)[:120]
            # (C) collapsed GBS herald via reduced_herald: vacuum modes are
            # conditioned analytically and the small remaining system is built
            # with the stable Hermite recurrence -- exact, fast, well-conditioned
            # (the old heralded_output at cutoff 40/60 cost one loop hafnian per
            # amplitude and never terminated on 11-12 dB generators).
            psiC = None
            if args.skip_c:
                rec["C_skipped"] = "--skip-c"
            else:
                try:
                    pc, _ = go.reduced_herald(eq["cov"], eq["mu"], eq["signal_idx"],
                                              eq["control_idx"], n0, cutoff=L)
                    psiC = np.asarray(pc).ravel()
                except Exception as e:
                    rec["C_error"] = repr(e)[:120]
            # (D) Hanamura reduced
            psiD = None
            n1 = list(n0)
            try:
                r = go.optimize_gbs_architecture(
                    eq["cov"], eq["mu"], eq["signal_idx"], eq["control_idx"], n0,
                    reduction_factor=3.0, original_probability=p["prob"],
                    verify=False, herald_cutoff=L)
                n1 = [int(x) for x in r["n_after"]]
                rec["n1"] = n1; rec["k_eff_after"] = int(sum(1 for x in n1 if x >= 1))
                if rec["k_eff_after"] == 1:
                    ma = next(j for j, x in enumerate(n1) if x >= 1)
                    pam = r["params_after"][ma]
                    psiD = ghd.core_state(float(pam["s0"]), complex(pam["delta0"]), int(n1[ma]), L)
                    rec["D_method"] = "core"
                else:
                    pd, _ = ghd.reduced_full_state(eq, n0, n1, cutoff=L)
                    psiD = np.asarray(pd).ravel(); rec["D_method"] = "herald_fallback"
            except Exception as e:
                rec["D_error"] = repr(e)[:120]

            # diagnostics
            rec["stats"] = {k: stats(s) for k, s in
                            dict(A=psiA, B=psiB, C=psiC, D=psiD).items() if s is not None}
            rec["fid_raw"] = dict(
                A_B=fid_raw(psiA, psiB), A_C=fid_raw(psiA, psiC), A_D=fid_raw(psiA, psiD))
            rec["fid_up_to_gauss"] = dict(
                A_B=fid_gauss(psiA, psiB), A_C=fid_gauss(psiA, psiC),
                A_D=fid_gauss(psiA, psiD), C_D=fid_gauss(psiC, psiD))

            # conditioning probe: A vs C at a second cutoff (A's convergence to
            # the exact reduced_herald reference is the REAL diagnostic: the
            # breeding sim is badly truncated at the config cutoff for
            # high-squeezing solutions -- see HANAMURA_VALIDATION_FINDINGS.md)
            if args.cutoff2 and not args.skip_c and args.cutoff2 != L:
                try:
                    pa2, _ = futils.compute_heralded_state(params, cutoff=args.cutoff2)
                    pc2, _ = go.reduced_herald(eq["cov"], eq["mu"], eq["signal_idx"],
                                               eq["control_idx"], n0, cutoff=args.cutoff2)
                    rec["fid_up_to_gauss"]["A_C_cutoff2"] = fid_gauss(pa2, pc2)
                    rec["cutoff2"] = args.cutoff2
                except Exception as e:
                    rec["cutoff2_error"] = repr(e)[:120]

            if args.rank:
                rec["eff_rank_A"], rec["rank_scan_A"] = effective_rank(psiA) if psiA is not None else (None, {})

            report[key] = rec
            fg = rec["fid_up_to_gauss"]
            print(f"[{key}] Nc={Nc} k={rec['k_control']}(eff {rec['k_eff']}) sq={rec['gbs_sq_db']}dB "
                  f"| A=B {fg['A_B']} | A=C(GBS) {fg['A_C']} | A=D(Han) {fg['A_D']} "
                  f"| C=D {fg['C_D']} | D={rec.get('D_method')}")
            panels.append((key, rec, psiA, psiC, psiD))

    json.dump(report, open(os.path.join(args.out, "validate_report.json"), "w"),
              indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))

    # ---- render the three Wigners per solution -------------------------------
    if panels:
        nR = len(panels)
        fig = plt.figure(figsize=(8.4, 2.7 * nR + 0.5))
        gs = gridspec.GridSpec(nR, 3, figure=fig, hspace=0.45, wspace=0.12,
                               left=0.10, right=0.97, top=0.95, bottom=0.05)
        for ri, (key, rec, psiA, psiC, psiD) in enumerate(panels):
            fg = rec["fid_up_to_gauss"]
            axA = fig.add_subplot(gs[ri, 0]); axC = fig.add_subplot(gs[ri, 1]); axD = fig.add_subplot(gs[ri, 2])
            render(axA, psiA, f"{key}  OPTIMIZER\nNc={rec['Nc']} sq={rec['gbs_sq_db']}dB")
            render(axC, psiC, f"GBS recon\nF(opt,GBS)={fg['A_C']}")
            render(axD, psiD, f"Hanamura {rec.get('n1','')}\nF(opt,Han)={fg['A_D']} [{rec.get('D_method','')}]")
            if ri == 0:
                axA.set_ylabel("p/√π")
        plt.savefig(os.path.join(args.out, "validate_wigners.pdf"), bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"\n[+] wrote {os.path.join(args.out, 'validate_wigners.pdf')} ({nR} rows)")
    print(f"[+] wrote {os.path.join(args.out, 'validate_report.json')}")


if __name__ == "__main__":
    main()
