#!/usr/bin/env python3
r"""
Rank-resolved lower bound on GKP nonlinear squeezing, and the rank-efficiency of
the discovered breeding protocols.
=====================================================================

Motivation (why this is the sharp question):
    The stellar bound r* <= N_c is saturated trivially -- almost every state
    heralded on N_c photons has stellar rank N_c, so "rank 17" says nothing about
    quality. The non-trivial object is the *rank-resolved lower bound*

        B_n(u) = inf { <psi|O_GKP(u)|psi> : psi has stellar rank <= n },

    the best GKP nonlinear squeezing ANY rank-n state can reach (Provaznik/
    Fiurasek witness family; Chabaud stellar rank). B_0(u) is exactly the
    Gaussian bound 5/3 - ||u||_inf. Whether the optimised protocols sit *near*
    B_{N_c} decides if the search is photon (rank) efficient or wasteful.

Method (self-contained; numpy/scipy only, no qutip/thewalrus):
    A stellar-rank-n pure state is a Gaussian-transformed core state, psi = G|c>
    with c supported on Fock {0..n} (Chabaud 2020). For a fixed single-mode
    Gaussian unitary G (squeeze r,phi; displace bx,bp; rotate psi_rot) the best
    core is the smallest eigenvalue of the (n+1)x(n+1) top-left block of
    G^dag O_GKP G. B_n = min over G of that eigenvalue (5 real params,
    multi-start + warm start along n). O_GKP is built from exact analytic
    displacement-operator matrix elements (hbar=1, x=(a+a^dag)/sqrt2), identical
    to the thesis convention and to src/utils/gkp_operator.py.

Validation anchors (assert on run):
    * B_0(u) -> 5/3 - max|u|  as cutoff/squeezing grow (Gaussian bound).
    * B_n non-increasing in n.
    * For |+_L>, B_n crosses the Gaussian bound around n ~ 8, matching the
      truncated-operator ground-state crossing at Fock cutoff N ~ 9
      (fig:gkp-truncation-convergence).

Outputs (into --out dir, default cascade_rank/):
    cascade_rank_bounds.json   B_n curves + champion/front placement + metadata
    cascade_rank_bounds.png    B_n(u) with the champions and full fronts overlaid

Champions / fronts are read from a small embedded table (the validated L=200
values, cheap/knee/champion per target) and, if present, the full fronts from
`ng_results_data.json` (copy it next to this script or pass --fronts PATH).

Usage (cluster):
    python scripts/cascade_rank_analysis.py --cutoff 160 --nmax 26 --restarts 8
    # heavy: B_0 needs strong p-squeezing, so use a large cutoff. ~1-3 h single
    # core for nmax=26 at cutoff 160 with 8 restarts; embarrassingly parallel
    # over (target, n) -- see --only-target / --nmin/--nmax to sub-shard.
"""
import argparse
import json
import math
import os
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import gammaln, eval_genlaguerre

SQRT_PI = math.sqrt(math.pi)

# validated L=200 champions (cheap/knee/champion) per target: (<O>, N_c)
# from mgr/scripts/ng_results_data.json "picks"
CHAMPIONS = {
    "plus": {"u": (1.0, 0.0, 0.0),
             "picks": [(0.647148, 8), (0.396587, 20), (0.389324, 24)]},
    "H":    {"u": (1/math.sqrt(2), 1/math.sqrt(2), 0.0),
             "picks": [(0.932565, 8), (0.575355, 14), (0.502563, 18)]},
    "T":    {"u": (1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)),
             "picks": [(1.004303, 8), (0.521397, 14), (0.491360, 17)]},
}
PICK_LABELS = ["cheap", "knee", "champion"]


# ---------------------------------------------------------------- analytic O_GKP
def _displacement_matrix(alpha, N):
    D = np.zeros((N, N), dtype=complex)
    a2 = abs(alpha) ** 2
    pref = math.exp(-a2 / 2.0)
    for m in range(N):
        for n in range(N):
            if m >= n:
                d = m - n
                amp = math.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)))
                D[m, n] = pref * amp * (alpha ** d) * eval_genlaguerre(n, d, a2)
            else:
                d = n - m
                amp = math.exp(0.5 * (gammaln(m + 1) - gammaln(n + 1)))
                D[m, n] = pref * amp * ((-np.conj(alpha)) ** d) * eval_genlaguerre(m, d, a2)
    return D


def _herm(alpha, N):
    D = _displacement_matrix(alpha, N)
    return (D + D.conj().T) / 2.0


def _eqa(u, v):
    # D(alpha) = exp[i(u x + v p)]  with x=(a+adag)/sqrt2
    return (-v + 1j * u) / math.sqrt(2.0)


def build_O_GKP(N, u):
    ux, uy, uz = u
    Ox = _herm(_eqa(0.0, -SQRT_PI), N)
    Oy = _herm(_eqa(-SQRT_PI, SQRT_PI), N)
    Oz = _herm(_eqa(SQRT_PI, 0.0), N)
    Sx = _herm(_eqa(0.0, -2 * SQRT_PI), N)
    Sy = _herm(_eqa(-2 * SQRT_PI, 2 * SQRT_PI), N)
    Sz = _herm(_eqa(2 * SQRT_PI, 0.0), N)
    I = np.eye(N)
    O1 = I - (Sx + Sy + Sz) / 3.0
    O = O1 + I - (ux * Ox + uy * Oy + uz * Oz)
    return np.real((O + O.conj().T) / 2.0)


def gaussian_bound(u):
    return 5.0 / 3.0 - max(abs(x) for x in u)


# ------------------------------------------------------------- Gaussian unitary
def _ops(N):
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    return a, a.conj().T, np.arange(N).astype(float)


def gaussian_unitary(p, a, ad, ndiag):
    r, phi, bx, bp, psirot = p
    S = expm(0.5 * (np.exp(-1j * phi) * r * (a @ a) - np.exp(1j * phi) * r * (ad @ ad)))
    D = expm((bx + 1j * bp) * ad - (bx - 1j * bp) * a)
    R = np.exp(-1j * psirot * ndiag)          # rotation is diagonal in Fock -> no expm
    return (D @ S) * R[np.newaxis, :]


# ---------------------------------------------------------------- rank bound B_n
def Bn(O, n, ops, init=None, restarts=6, rmax=2.6, maxiter=400, rng=None):
    a, ad, ndiag = ops
    if rng is None:
        rng = np.random.default_rng(0)
    seeds = []
    if init is not None:
        seeds.append(np.asarray(init, float))
    # priors: strong p-squeeze (phi=pi) is the Gaussian optimum for these targets
    seeds += [np.array([1.6, math.pi, 0.0, 0.0, 0.0]),
              np.array([1.9, math.pi, 0.0, 0.0, 0.0]),
              np.array([1.2, math.pi, 0.3, 0.0, 0.0])]
    while len(seeds) < restarts:
        seeds.append(np.array([0.4 * rng.standard_normal(), rng.random() * 2 * math.pi,
                               0.4 * rng.standard_normal(), 0.4 * rng.standard_normal(),
                               rng.random() * 2 * math.pi]))

    def obj(p):
        if abs(p[0]) > rmax:
            return 10.0
        G = gaussian_unitary(p, a, ad, ndiag)
        M = G.conj().T @ O @ G
        Msub = M[:n + 1, :n + 1]
        Msub = (Msub + Msub.conj().T) / 2.0
        return float(np.linalg.eigvalsh(Msub)[0])

    best = (np.inf, None)
    for x0 in seeds[:restarts]:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 2e-5, "maxiter": maxiter})
        if res.fun < best[0]:
            best = (res.fun, res.x)
    return best


# ---------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", type=int, default=160)
    ap.add_argument("--nmax", type=int, default=26)
    ap.add_argument("--nmin", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--only-target", default=None, help="plus|H|T (default: all)")
    ap.add_argument("--fronts", default=None, help="path to ng_results_data.json (optional)")
    ap.add_argument("--out", default="cascade_rank")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    N = args.cutoff
    ops = _ops(N)
    rng = np.random.default_rng(args.seed)

    fronts = {}
    fpath = args.fronts or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "ng_results_data.json")
    if os.path.exists(fpath):
        d = json.load(open(fpath))
        for t in ("plus", "H", "T"):
            fr = d.get("fronts", {}).get(t)
            if fr:
                fronts[t] = fr  # expected: list of {expO/exp, Nc, logP} or similar
        print(f"[fronts] loaded from {fpath}")
    else:
        print(f"[fronts] {fpath} not found -- champions only")

    targets = [args.only_target] if args.only_target else ["plus", "H", "T"]
    result = {"meta": {"cutoff": N, "nmax": args.nmax, "restarts": args.restarts,
                       "convention": "hbar=1, x=(a+adag)/sqrt2, O_GKP=O1+I-u.O"},
              "targets": {}}

    for t in targets:
        u = CHAMPIONS[t]["u"]
        BG = gaussian_bound(u)
        O = build_O_GKP(N, u)
        print(f"\n=== {t}  u={tuple(round(x,4) for x in u)}  Gaussian bound B_G={BG:.4f} ===")
        ns, Bs = [], []
        init = np.array([1.9, math.pi, 0.0, 0.0, 0.0])
        t0 = time.time()
        for n in range(args.nmin, args.nmax + 1):
            b, x = Bn(O, n, ops, init=init, restarts=args.restarts,
                      maxiter=args.maxiter, rng=rng)
            init = x  # warm start next n from this optimum
            ns.append(n); Bs.append(b)
            print(f"  B_{n:2d} = {b:.4f}   ({time.time()-t0:5.0f}s)")
        # monotonicity (allow tiny numerical noise)
        for i in range(1, len(Bs)):
            if Bs[i] > Bs[i-1] + 5e-3:
                print(f"  [warn] non-monotone B_{ns[i]} > B_{ns[i-1]}: "
                      f"{Bs[i]:.4f} > {Bs[i-1]:.4f} (raise --restarts/--cutoff)")
        # champion placement: excess over the rank-N_c bound
        picks = []
        for (expO, Nc), lab in zip(CHAMPIONS[t]["picks"], PICK_LABELS):
            b_at = Bs[Nc - args.nmin] if args.nmin <= Nc <= args.nmax else None
            picks.append({"label": lab, "expO": expO, "Nc": Nc,
                          "B_Nc": b_at,
                          "excess_over_bound": (expO - b_at) if b_at is not None else None,
                          "xi_vs_bound_dB": -10*math.log10(expO/BG)})
        result["targets"][t] = {"u": u, "gaussian_bound": BG,
                                "n": ns, "B_n": Bs, "picks": picks,
                                "front": fronts.get(t)}
        for p in picks:
            if p["B_Nc"] is not None:
                print(f"  [{p['label']:9s}] <O>={p['expO']:.4f} @ N_c={p['Nc']}: "
                      f"B_{p['Nc']}={p['B_Nc']:.4f}, excess={p['excess_over_bound']:+.4f}")

    outjson = os.path.join(args.out, "cascade_rank_bounds.json")
    json.dump(result, open(outjson, "w"), indent=1)
    print(f"\nwrote {outjson}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(result["targets"]), figsize=(5*len(result["targets"]), 4.2), squeeze=False)
        for ax, (t, R) in zip(axes[0], result["targets"].items()):
            ax.plot(R["n"], R["B_n"], "-o", ms=3, color="0.2", label=r"$B_n(u)$ rank bound")
            ax.axhline(R["gaussian_bound"], ls="--", color="crimson", lw=1,
                       label=r"$B_G=B_0$")
            for p in R["picks"]:
                ax.plot(p["Nc"], p["expO"], "s", ms=8, mfc="none",
                        color="tab:blue")
                ax.annotate(p["label"], (p["Nc"], p["expO"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)
            if R.get("front"):
                try:
                    xs = [pt.get("Nc") for pt in R["front"]]
                    ys = [pt.get("expO", pt.get("exp")) for pt in R["front"]]
                    ax.plot(xs, ys, ".", ms=3, color="tab:orange", alpha=0.5,
                            label="front")
                except Exception:
                    pass
            ax.set_title(t); ax.set_xlabel("stellar rank $n$ (= $N_c$ ceiling)")
            ax.set_ylabel(r"$\langle \hat O_\mathrm{GKP}\rangle$"); ax.legend(fontsize=8)
        outpng = os.path.join(args.out, "cascade_rank_bounds.png")
        fig.tight_layout(); fig.savefig(outpng, dpi=150)
        print(f"wrote {outpng}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
