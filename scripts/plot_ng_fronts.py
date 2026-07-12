#!/usr/bin/env python3
"""plot_ng_fronts.py -- before/after Hanamura Pareto fronts for the ng
campaign (<O> vs log10 P), one panel per target.

'Before' = the validated (L=200-exact) sub-Gaussian Pareto front
(recompute_ng/pareto_fronts).  'After' = the same states with the
Hanamura-optimized success probability (hanamura_ng/hanamura_summary.csv);
<O> is unchanged by construction (the heralded core state is preserved).
Dashed line = analytic Gaussian limit G, dotted = clamped G_N.

  python scripts/plot_ng_fronts.py [--out hanamura_ng/ng_fronts.png]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

G_REF = {"a1p00_b1p00": (0.666667, 0.710782, "α=1, β=1 (plus)"),
         "a1p41_b1p41": (0.959560, 0.997924, "α=√2, β=1+1j (H-type)"),
         "a2p73_b1p41": (1.089316, 1.124977, "α=2.732, β=1+1j (magic)")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=os.path.join(REPO, "hanamura_ng",
                                                      "hanamura_summary.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "hanamura_ng",
                                                  "ng_fronts.png"))
    args = ap.parse_args(argv)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_csv(args.summary)
    s["target"] = s.group.str.extract(r"_c30_(a\dp\d+_b\dp\d+)")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (tgt, (G, GN, label)) in zip(axs, G_REF.items()):
        g = s[s.target == tgt]
        if g.empty:
            continue
        ax.scatter(np.log10(g.prob_before), g.exp_O, s=28, c="#1f77b4",
                   label="validated front (before)", zorder=3)
        v = g[g.han_valid]
        ax.scatter(np.log10(v.prob_after), v.exp_O, s=28, c="#d62728",
                   marker="^", label="after Hanamura", zorder=3)
        for _, r in v.iterrows():
            ax.annotate("", xy=(np.log10(r.prob_after), r.exp_O),
                        xytext=(np.log10(r.prob_before), r.exp_O),
                        arrowprops=dict(arrowstyle="->", color="0.75", lw=0.8))
        ax.axhline(G, ls="--", c="k", lw=1, label=f"G = {G:.4f}")
        ax.axhline(GN, ls=":", c="0.4", lw=1, label=f"G_N = {GN:.4f}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("log10 P")
        ax.set_ylabel(r"$\langle O \rangle$")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("NG-campaign validated Pareto fronts: before/after Hanamura "
                 "control-parameter optimization", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
