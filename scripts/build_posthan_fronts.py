#!/usr/bin/env python3
"""build_posthan_fronts.py -- reconstruct the TRUE post-Hanamura Pareto fronts
from the multi-factor all-states sweep (`run_hanamura_all.py`) and identify the
states that were dominated *before* the Hanamura optimization but land on a
front *after* it.

Every optimized point (state x reduction-factor) is a candidate.  For each
target (B30F and B30 merged) it builds two fronts, both entirely from the
recomputed POST-Hanamura values and sorted by them:

  * PROBABILITY front : minimize <O>_after, maximize P_after
  * SQUEEZING   front : minimize <O>_after, minimize max_sq_after
                        (max necessary squeezing, dB -- the experimental cost)

and a PRE probability front over (exp_before, prob_before) as a reference.
Every point is classified:
  * on_prob_front / on_sq_front  (post-Hanamura membership)
  * promoted : on the POST probability front but its parent state's
               (exp_before, prob_before) point was dominated -> a state the
               front-only Hanamura run could not have found.

Because a state appears at several reduction factors, the front automatically
picks each state's best factor; `reduction_factor` is carried on every row and
onto the per-factor front CSVs so pass-2 Wigner is reproducible.

Outputs (under --out, default `posthan_fronts/`):
  all_points.csv                  one row per (state, factor): before/after
                                  <O>, P, max_sq + front flags
  posthan_summary.md              per-target counts, champions, PROMOTED table,
                                  factor breakdown
  fronts_probability.png          <O>_after vs P_after, PRE front + promoted
  fronts_squeezing.png            <O>_after vs max necessary squeezing (dB)
  front_csvs/prob_<target>_rf<rf>.csv   POST prob-front points, run_hanamura
                                  _pareto.py schema (feed back per factor for
                                  the pass-2 Wigner render):
      JAX_ENABLE_X64=1 python scripts/run_hanamura_pareto.py \
          --pareto-dir posthan_fronts/front_csvs --groups 'prob_*_rf3*' \
          --reduction-factor 3.0 --out hanamura_posthan_front

  python scripts/build_posthan_fronts.py --sweep-dir hanamura_all
"""
import os
import sys
import glob
import argparse

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET_AB = {
    "a1p00_b1p00": ("1.0", "(1+0j)"),
    "a1p41_b1p41": ("1.4142135623730951", "(1+1j)"),
    "a2p73_b1p41": ("2.7320508", "(1+1j)"),
    "a0p00_b1p00": ("0.0", "(1+0j)"),
}
G_LIMIT = {"a1p00_b1p00": 0.666667, "a1p41_b1p41": 0.959560,
           "a2p73_b1p41": 1.089316}


def pareto_mask(x, y, maximize_y=True):
    """Boolean mask of Pareto-optimal points: always minimize x; maximize y if
    maximize_y else minimize y.  O(n^2); n is a few thousand, fine."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    n = len(x)
    keep = np.zeros(n, bool)
    ys = y if maximize_y else -y
    for i in range(n):
        if not finite[i]:
            continue
        better_y = ys >= ys[i]
        dom = (x <= x[i]) & better_y & ((x < x[i]) | (ys > ys[i])) & finite
        keep[i] = not np.any(dom)
    return keep


def load_sweep(sweep_dir):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(sweep_dir, "hanamura_all.shard*.jsonl")))
    if not files:
        files = sorted(glob.glob(os.path.join(sweep_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"no sweep JSONL found under {sweep_dir}")
    frames = [pd.read_json(f, lines=True) for f in files if os.path.getsize(f)]
    df = pd.concat(frames, ignore_index=True)
    subset = "rec_id" if "rec_id" in df.columns else "key"
    df = df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default=os.path.join(REPO, "hanamura_all"))
    ap.add_argument("--out", default=os.path.join(REPO, "posthan_fronts"))
    ap.add_argument("--min-fidelity", type=float, default=0.0,
                    help="drop post-front candidates below this after/before "
                         "fidelity (0 = keep all; fidelity is reported, not "
                         "gated, by default -- the reduction moves the state on "
                         "purpose)")
    args = ap.parse_args(argv)

    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "front_csvs"), exist_ok=True)
    df = load_sweep(args.sweep_dir)
    if "reduction_factor" not in df.columns:
        df["reduction_factor"] = np.nan
    print(f"loaded {len(df)} optimized points "
          f"({df['key'].nunique()} states x factors) from {args.sweep_dir}")

    for c in ["exp_before", "exp_after", "prob_before", "prob_after",
              "max_sq_before", "max_sq_after", "reduction_factor"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["logP_after"] = np.log10(df["prob_after"].where(df["prob_after"] > 0))

    all_rows, promoted_total = [], 0
    md = ["# Post-Hanamura Pareto fronts (all validated states, factor sweep)\n",
          "Fronts built entirely from the recomputed POST-Hanamura values and "
          "sorted by them. `exp_after` is ⟨O⟩ **minimized over the final Gaussian "
          "unitary** (the reduction is only defined up to one; `exp_after_raw` is "
          "the stale-frame value for contrast). Reduction factor **1.0 = "
          "damping-only**: exactly output-preserving (⟨O⟩ unchanged, fid≈1), so "
          "those points are the *lossless* probability-boost front. Every "
          "optimized point (state x factor) is a candidate; the front picks each "
          "state's best factor. **PROMOTED** = on the POST probability front but "
          "dominated before → states the front-only run could not have found.\n"]

    for target, sub in df.groupby("target"):
        sub = sub.reset_index(drop=True)

        # PRE front over UNIQUE states (before values are factor-independent)
        uniq = sub.drop_duplicates(subset="key")
        pre_keep_keys = set(uniq["key"].values[
            pareto_mask(uniq["exp_before"], uniq["prob_before"], maximize_y=True)])
        pre_mask = sub["key"].isin(pre_keep_keys).values

        ok = (np.isfinite(sub["exp_after"]) & np.isfinite(sub["prob_after"])).values
        if args.min_fidelity > 0:
            ok = ok & (pd.to_numeric(sub.get("fidelity_after_before"),
                                     errors="coerce").fillna(0).values
                       >= args.min_fidelity)

        prob_mask = np.zeros(len(sub), bool)
        sq_mask = np.zeros(len(sub), bool)
        if ok.any():
            idx = np.where(ok)[0]
            pm = pareto_mask(sub["exp_after"].values[idx],
                             sub["prob_after"].values[idx], maximize_y=True)
            prob_mask[idx[pm]] = True
            okq = idx[np.isfinite(sub["max_sq_after"].values[idx])]
            if len(okq):
                qm = pareto_mask(sub["exp_after"].values[okq],
                                 sub["max_sq_after"].values[okq], maximize_y=False)
                sq_mask[okq[qm]] = True

        # promoted (probability front): parent state dominated before
        promoted_states = set(sub["key"].values[prob_mask]) - pre_keep_keys
        promoted = sub["key"].isin(promoted_states).values & prob_mask
        promoted_total += len(promoted_states)

        sub["on_pre_front"] = pre_mask
        sub["on_prob_front"] = prob_mask
        sub["on_sq_front"] = sq_mask
        sub["promoted"] = promoted
        all_rows.append(sub)

        # per-factor POST prob-front CSVs (run_hanamura_pareto schema)
        a_str, b_str = TARGET_AB.get(target, ("", ""))
        for rf, front in sub[prob_mask].groupby("reduction_factor"):
            csv_rows = []
            for _, r in front.iterrows():
                run_rel = r["run"]
                csv_rows.append(dict(
                    root=r.get("root") or os.path.dirname(os.path.dirname(run_rel)),
                    group=r.get("group") or os.path.basename(os.path.dirname(run_rel)),
                    run=os.path.basename(run_rel), cell_idx=int(r["cell"]),
                    exp_hi=float(r["exp_after"]),
                    logP=float(r["logP_after"]) if np.isfinite(r["logP_after"]) else 0.0,
                    prob=float(r["prob_after"]),
                    total_photons=float(r["Nc_after"]), fired_modes="",
                    max_pnr=float(max(r["n1"]) if r.get("n1") else 0),
                    vs_gaussian="", vs_gs="", wigner_negvol="",
                    target_alpha=a_str, target_beta=b_str))
            rf_tag = f"{rf:g}".replace(".", "p") if np.isfinite(rf) else "na"
            pd.DataFrame(csv_rows).to_csv(
                os.path.join(args.out, "front_csvs",
                             f"prob_{target}_rf{rf_tag}.csv"), index=False)

        # markdown block
        G = G_LIMIT.get(target, float("nan"))
        md.append(f"\n## {target}  ({uniq['exp_before'].notna().sum()} states, "
                  f"{len(sub)} optimized points)\n")
        md.append(f"- PRE prob-front: **{len(pre_keep_keys)}** states | "
                  f"POST prob-front: **{int(prob_mask.sum())}** points | "
                  f"POST sq-front: **{int(sq_mask.sum())}** points | "
                  f"**PROMOTED: {len(promoted_states)}** states")
        pf = sub[prob_mask]
        if len(pf):
            fb = pf.groupby("reduction_factor").size().to_dict()
            md.append(f"- prob-front points per factor: "
                      + ", ".join(f"rf{k:g}={v}" for k, v in sorted(fb.items())))
        best = sub.loc[sub["exp_after"].idxmin()] if len(sub) else None
        if best is not None:
            md.append(f"- lowest ⟨O⟩_after: {best['exp_after']:.4f} "
                      f"(rf{best['reduction_factor']:g}, "
                      f"P {best['prob_after']:.2e}, sq {best['max_sq_after']:.1f} dB, "
                      f"⟨O⟩_before {best['exp_before']:.4f}, "
                      f"fid {best.get('fidelity_after_before', float('nan')):.3f})")
        if len(promoted_states):
            pr = (sub[promoted].sort_values("exp_after")
                  .drop_duplicates(subset="key"))
            md.append(f"\n### PROMOTED states ({target}) -- top 15 by ⟨O⟩_after\n")
            md.append("| provenance | rf | ⟨O⟩ before→after | P before→after | "
                      "Nc b→a | sqdB after | fid |")
            md.append("|---|---|---|---|---|---|---|")
            for _, r in pr.head(15).iterrows():
                prov = f"{r['group']}/{os.path.basename(r['run'])}#{int(r['cell'])}"
                md.append(
                    f"| {prov} | {r['reduction_factor']:g} | "
                    f"{r['exp_before']:.4f}→{r['exp_after']:.4f} | "
                    f"{r['prob_before']:.1e}→{r['prob_after']:.1e} | "
                    f"{int(r['Nc_before'])}→{int(r['Nc_after'])} | "
                    f"{r['max_sq_after']:.1f} | "
                    f"{r.get('fidelity_after_before', float('nan')):.3f} |")

    out_df = pd.concat(all_rows, ignore_index=True)
    keep_cols = ["target", "group", "design", "depth", "run", "cell",
                 "reduction_factor", "exp_before", "exp_after", "exp_after_raw",
                 "prob_before", "prob_after", "prob_after_stable",
                 "prob_after_herald", "prob_before_archive",
                 "max_sq_before", "max_sq_after",
                 "Nc_before", "Nc_after", "fidelity_after_before", "fidelity_raw",
                 "on_pre_front", "on_prob_front", "on_sq_front", "promoted"]
    out_df[[c for c in keep_cols if c in out_df.columns]].to_csv(
        os.path.join(args.out, "all_points.csv"), index=False)

    md.insert(1, f"\n**{promoted_total} promoted states total** (dominated "
                 f"pre-Hanamura, on the post-Hanamura probability front).\n")
    open(os.path.join(args.out, "posthan_summary.md"), "w").write(
        "\n".join(md) + "\n")

    _plot_prob(out_df, os.path.join(args.out, "fronts_probability.png"))
    _plot_sq(out_df, os.path.join(args.out, "fronts_squeezing.png"))
    print(f"DONE -> {args.out}  ({promoted_total} promoted states total)")
    return 0


def _factor_colors(df):
    import matplotlib.cm as cm
    rfs = sorted(x for x in df["reduction_factor"].dropna().unique())
    cmap = cm.get_cmap("viridis", max(len(rfs), 1))
    return {rf: cmap(i) for i, rf in enumerate(rfs)}, rfs


def _plot_prob(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = sorted(df["target"].unique())
    colors, rfs = _factor_colors(df)
    fig, axs = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), 5.2),
                            squeeze=False)
    for ax, target in zip(axs[0], targets):
        sub = df[df["target"] == target]
        for rf in rfs:
            s = sub[sub["reduction_factor"] == rf]
            ax.scatter(s["exp_after"], s["prob_after"], s=10, color=colors[rf],
                       alpha=0.35, label=f"all (rf{rf:g})", zorder=1)
        pre = sub[sub["on_pre_front"]].drop_duplicates("key").sort_values("exp_before")
        ax.plot(pre["exp_before"], pre["prob_before"], "--o", color="0.5",
                ms=3, lw=1, label="PRE front (before)", zorder=2)
        post = sub[sub["on_prob_front"]].sort_values("exp_after")
        ax.plot(post["exp_after"], post["prob_after"], "-o", color="tab:red",
                ms=4, lw=1.3, label="POST prob front", zorder=3)
        promo = sub[sub["promoted"]]
        ax.scatter(promo["exp_after"], promo["prob_after"], s=90, marker="*",
                   c="gold", edgecolors="k", linewidths=0.5,
                   label=f"promoted ({promo['key'].nunique()})", zorder=4)
        if target in G_LIMIT:
            ax.axvline(G_LIMIT[target], ls=":", c="k", lw=1,
                       label=f"G={G_LIMIT[target]:.3f}")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\langle O\rangle_{\rm after}$  (lower = more sub-Gaussian)")
        ax.set_ylabel(r"$P_{\rm after}$")
        ax.set_title(target)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Post-Hanamura probability fronts (B30F+B30 merged, factor sweep)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_sq(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    targets = sorted(df["target"].unique())
    colors, rfs = _factor_colors(df)
    fig, axs = plt.subplots(1, len(targets), figsize=(6.0 * len(targets), 5.2),
                            squeeze=False)
    for ax, target in zip(axs[0], targets):
        sub = df[df["target"] == target]
        for rf in rfs:
            s = sub[sub["reduction_factor"] == rf]
            ax.scatter(s["exp_after"], s["max_sq_after"], s=10, color=colors[rf],
                       alpha=0.35, label=f"all (rf{rf:g})", zorder=1)
        post = sub[sub["on_sq_front"]].sort_values("exp_after")
        ax.plot(post["exp_after"], post["max_sq_after"], "-o", color="tab:blue",
                ms=4, lw=1.3, label="POST squeezing front", zorder=3)
        if target in G_LIMIT:
            ax.axvline(G_LIMIT[target], ls=":", c="k", lw=1,
                       label=f"G={G_LIMIT[target]:.3f}")
        ax.set_xlabel(r"$\langle O\rangle_{\rm after}$  (lower = more sub-Gaussian)")
        ax.set_ylabel("max necessary squeezing (dB)")
        ax.set_title(target)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Post-Hanamura quality vs max-squeezing fronts "
                 "(B30F+B30 merged, factor sweep)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
