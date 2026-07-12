#!/usr/bin/env python3
"""Analyze the post-Hanamura sweep (hanamura/hanamura_all2) and emit every
number the thesis quotes, plus the combined all-recipes fronts.

FRAME: everything here lives in the GENERATOR frame (herald probability of the
equivalent GBS generator: `prob_before` for the undamped generator,
`prob_after` for the post-Hanamura one).  Quality is always on the archive
L=200 scale: `exp_before` (= exp_hi) and `exp_after_cal`.  Do not mix with the
tree-frame archive probabilities used in the in-house section.

Outputs: posthan_analysis.json (repo root) + stdout summary.
"""
import os, json, glob, math, collections
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "hanamura", "hanamura_all2")

GB = {"a1p00_b1p00": 2/3, "a1p41_b1p41": 5/3 - 1/math.sqrt(2),
      "a2p73_b1p41": 5/3 - 1/math.sqrt(3)}
LABEL = {"a1p00_b1p00": "plus", "a1p41_b1p41": "H", "a2p73_b1p41": "T"}
PLOT_P_FLOOR = 1e-16          # double-precision floor for plotted points

def xi(expO, t):
    return -10*math.log10(expO/GB[t])

def load():
    by_id = {}
    for f in glob.glob(os.path.join(SWEEP, "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
                by_id[r["rec_id"]] = r
            except Exception:
                pass
    return list(by_id.values())

def nondominated(pts, y_key, max_y=True):
    """pts: list of dicts with 'exp' and y_key; minimize exp, max/min y."""
    keep = []
    for a in pts:
        dom = False
        for b in pts:
            better_y = (b[y_key] >= a[y_key]) if max_y else (b[y_key] <= a[y_key])
            strictly = (b["exp"] < a["exp"]) or \
                       ((b[y_key] > a[y_key]) if max_y else (b[y_key] < a[y_key]))
            if b["exp"] <= a["exp"] and better_y and strictly:
                dom = True
                break
        if not dom:
            keep.append(a)
    return sorted(keep, key=lambda p: p[y_key], reverse=max_y)

def q(x, p):
    return float(np.percentile(np.asarray(x, float), p))

def gen_frame_ok(r, tol=100.0, floor=1e-16):
    """Generator-frame P_before trustworthiness.  Where the reduced-herald and
    stable density-matrix estimators both exist they agree to <0.1 in log10
    across the whole sweep (measured), except genuinely broken states (one at
    1e-70 vs 1e-33): require agreement within ``tol``.  Where the stable value
    is missing (fired tensor over budget), accept the single reduced-herald
    estimator if it is above the double-precision floor."""
    pb, ps = r.get("prob_before"), r.get("prob_before_stable")
    if not pb or pb <= 0:
        return False
    if ps is None or ps <= 0:
        return pb >= floor
    ratio = pb / ps
    return (1.0/tol) <= ratio <= tol


def after_p_ok(r, tol=100.0, floor=1e-15):
    """prob_after trustworthiness (same logic; a stable value of exactly 0
    against a tiny hafnian value means both are below precision)."""
    pa, ps = r.get("prob_after"), r.get("prob_after_stable")
    if not pa or pa <= 0:
        return False
    if ps is None:
        return pa >= floor
    if ps <= 0:
        return pa >= 1e-12   # stable underflowed; trust hafnian only if large
    ratio = pa / ps
    return (1.0/tol) <= ratio <= tol


def main():
    recs = load()
    out = {"targets": {}}
    for tgt in GB:
        R_all = [r for r in recs if r["target"] == tgt and r.get("before_ok", True)]
        # generator-frame consistency gate, per STATE (rf1 carries the
        # undamped-generator estimators)
        state_ok = {}
        for r in R_all:
            if r["reduction_factor"] == 1.0:
                state_ok[r["key"]] = gen_frame_ok(r)
        n_frame_excluded = sum(1 for v in state_ok.values() if not v)
        R = [r for r in R_all if state_ok.get(r["key"], False)]
        states = {}
        for r in R:
            states.setdefault(r["key"], []).append(r)
        n_states = len(states)
        gated = len(set(r["key"] for r in recs
                        if r["target"] == tgt and not r.get("before_ok", True)))

        # ---------- rf1 (damping-only, lossless) ----------
        rf1 = [r for r in R if r["reduction_factor"] == 1.0
               and r.get("prob_after") and r.get("prob_before")
               and r["prob_before"] > 0]
        gains = [r["prob_after"]/r["prob_before"] for r in rf1]
        dsq = [r["max_sq_after"] - r["max_sq_before"] for r in rf1
               if r.get("max_sq_after") is not None
               and r.get("max_sq_before") is not None
               and np.isfinite(r["max_sq_after"]) and np.isfinite(r["max_sq_before"])]
        rf1_stats = dict(
            n=len(rf1),
            gain_median=q(gains, 50), gain_q25=q(gains, 25), gain_q75=q(gains, 75),
            gain_max=float(max(gains)), gain_min=float(min(gains)),
            frac_gain_gt2=float(np.mean(np.asarray(gains) > 2.0)),
            frac_gain_gt10=float(np.mean(np.asarray(gains) > 10.0)),
            dsq_median=q(dsq, 50), dsq_q25=q(dsq, 25), dsq_q75=q(dsq, 75),
        )
        gmax = max(rf1, key=lambda r: r["prob_after"]/r["prob_before"])
        rf1_stats["gain_max_state"] = dict(
            run=gmax["run"], cell=gmax["cell"], n0=gmax["n0"],
            exp=gmax["exp_before"], xi=xi(gmax["exp_before"], tgt),
            P_before=gmax["prob_before"], P_after=gmax["prob_after"])

        # ---------- rf >= 2 (reduction) ----------
        red_stats = {}
        for rf in (2.0, 3.0, 4.0):
            RR = [r for r in R if r["reduction_factor"] == rf
                  and r.get("after_ok", False)
                  and r.get("exp_after_cal") is not None]
            dexp = [r["exp_after_cal"] - r["exp_before"] for r in RR]
            fid = [r["fidelity_after_before"] for r in RR
                   if r.get("fidelity_after_before") is not None]
            below = [r for r in RR if r["exp_after_cal"] < GB[tgt]]
            dnc = [r["Nc_before"] - r["Nc_after"] for r in RR]
            red_stats[f"rf{rf:g}"] = dict(
                n=len(RR), dexp_median=q(dexp, 50), dexp_q25=q(dexp, 25),
                dexp_q75=q(dexp, 75), fid_median=q(fid, 50),
                n_below_bound=len(below),
                frac_below_bound=len(below)/max(len(RR), 1),
                dNc_median=q(dnc, 50),
                best_below=(min((r["exp_after_cal"] for r in below),
                                default=None)))

        # ---------- combined all-recipes fronts (generator frame) ----------
        pts = []
        seen_state_tree = {}
        for key, rs in states.items():
            r0 = rs[0]
            if r0.get("prob_before") and r0["prob_before"] > 0:
                pts.append(dict(exp=r0["exp_before"], P=r0["prob_before"],
                                sq=r0.get("max_sq_before"), recipe="tree",
                                key=key, run=r0["run"], cell=r0["cell"],
                                Nc=r0["Nc_before"], rf=0.0))
                seen_state_tree[key] = pts[-1]
        for r in R:
            if not r.get("after_ok", False):
                continue
            if not after_p_ok(r):
                continue
            if r.get("exp_after_cal") is None:
                continue
            pts.append(dict(exp=r["exp_after_cal"], P=r["prob_after"],
                            sq=r.get("max_sq_after"), recipe=f"rf{r['reduction_factor']:g}",
                            key=r["key"], run=r["run"], cell=r["cell"],
                            Nc=r["Nc_after"], rf=r["reduction_factor"]))
        plot_pts = [p for p in pts if p["P"] >= PLOT_P_FLOOR]
        # fronts are built over CERTIFIED non-Gaussian outputs only (exp below
        # the Gaussian bound, i.e. xi > 0): a recipe whose output a Gaussian
        # state can match is not a resource trade-off worth charting
        cert_pts = [p for p in plot_pts if p["exp"] < GB[tgt]]
        front = nondominated(cert_pts, "P", max_y=True)
        tree_front = nondominated([p for p in cert_pts if p["recipe"] == "tree"],
                                  "P", max_y=True)
        tree_front_keys = set(p["key"] for p in tree_front)
        comp = collections.Counter(p["recipe"] for p in front)
        promoted = set(p["key"] for p in front) - tree_front_keys

        # squeezing front (minimize both exp and sq), certified-only
        sq_pts = [p for p in cert_pts if p.get("sq") is not None
                  and np.isfinite(p["sq"])]
        sq_front = nondominated(sq_pts, "sq", max_y=False)
        sq_comp = collections.Counter(p["recipe"] for p in sq_front)

        out["targets"][tgt] = dict(
            label=LABEL[tgt], gaussian_bound=GB[tgt],
            n_states=n_states, n_gated=gated,
            n_frame_excluded=n_frame_excluded,
            rf1=rf1_stats, reduction=red_stats,
            combined_front=dict(
                n_candidates=len(plot_pts), n_front=len(front),
                composition={k: int(v) for k, v in comp.items()},
                n_promoted_states=len(promoted),
                tree_front_size=len(tree_front),
                points=[{k: p[k] for k in ("exp", "P", "sq", "recipe", "Nc",
                                           "run", "cell")} for p in front]),
            squeezing_front=dict(
                n_front=len(sq_front),
                composition={k: int(v) for k, v in sq_comp.items()},
                points=[{k: p[k] for k in ("exp", "P", "sq", "recipe", "Nc",
                                           "run", "cell")} for p in sq_front]),
            # full candidate clouds for plotting
            cloud=dict(
                tree=[[p["exp"], p["P"], p["sq"], p["Nc"]] for p in plot_pts
                      if p["recipe"] == "tree"],
                rf1=[[p["exp"], p["P"], p["sq"], p["Nc"]] for p in plot_pts
                     if p["recipe"] == "rf1"],
                red=[[p["exp"], p["P"], p["sq"], p["Nc"]] for p in plot_pts
                     if p["recipe"] in ("rf2", "rf3", "rf4")]),
        )

        L = LABEL[tgt]
        print(f"\n== {L} ({tgt}, B_G={GB[tgt]:.4f}) — {n_states} states "
              f"(+{gated} gated, {n_frame_excluded} frame-inconsistent "
              f"excluded) ==")
        print(f" rf1 gains: median x{rf1_stats['gain_median']:.1f} "
              f"[IQR {rf1_stats['gain_q25']:.1f}-{rf1_stats['gain_q75']:.1f}], "
              f"max x{rf1_stats['gain_max']:.0f} "
              f"({rf1_stats['gain_max_state']['run'].split('/')[-1]}"
              f"#{rf1_stats['gain_max_state']['cell']}, "
              f"xi={rf1_stats['gain_max_state']['xi']:.2f} dB); "
              f"dsq median {rf1_stats['dsq_median']:+.1f} dB")
        for rf, s in red_stats.items():
            print(f" {rf}: n={s['n']}  d<O> median {s['dexp_median']:+.3f}  "
                  f"fid median {s['fid_median']:.2f}  below-bound "
                  f"{s['n_below_bound']} ({100*s['frac_below_bound']:.1f}%)  "
                  f"dNc median {s['dNc_median']:.0f}")
        cf = out["targets"][tgt]["combined_front"]
        print(f" combined front: {cf['n_front']} points "
              f"{cf['composition']}  promoted states: "
              f"{cf['n_promoted_states']} (tree-only front: "
              f"{cf['tree_front_size']})")
        sf = out["targets"][tgt]["squeezing_front"]
        print(f" squeezing front: {sf['n_front']} points {sf['composition']}")

    # ---------- story states: the thesis picks (cheap/knee/champion) ----------
    ng_path = os.environ.get("NG_DATA", os.path.expanduser(
        "~/Nextcloud/vojtech/writing/mgr/scripts/ng_results_data.json"))
    try:
        picks = json.load(open(ng_path))["picks"]
        story = {}
        idx = {}
        for r in recs:
            idx[(r["run"].split("/")[-1], r["cell"], r["reduction_factor"])] = r
        for t, name_map in (("plus", "plus"), ("H", "H"), ("T", "T")):
            for i, nm in enumerate(("cheap", "knee", "champion")):
                rec = picks[t][i]
                run = rec["run"].split("/")[-1]
                r1 = idx.get((run, rec["cell"], 1.0))
                if r1 is None:
                    continue
                entry = dict(state=f"{t}_{nm}", run=run, cell=rec["cell"],
                             n0=r1["n0"],
                             P_gen_before=r1.get("prob_before"),
                             P_gen_after=r1.get("prob_after"),
                             gain_rf1=(r1["prob_after"]/r1["prob_before"]
                                       if r1.get("prob_after") and r1.get("prob_before")
                                       else None),
                             dsq_rf1=(r1.get("max_sq_after") or float("nan"))
                                     - (r1.get("max_sq_before") or float("nan")))
                story[f"{t}_{nm}"] = entry
        out["story_states"] = story
        print("\nstory states (thesis picks, rf1 damping):")
        for k, v in story.items():
            g = v["gain_rf1"]
            print(f"  {k:14s} gain x{g:8.1f}  dsq {v['dsq_rf1']:+5.1f} dB  "
                  f"P_gen {v['P_gen_before']:.2e} -> {v['P_gen_after']:.2e}"
                  if g else f"  {k:14s} (no valid rf1 gain)")
    except Exception as e:
        print("story-state extraction skipped:", e)

    with open(os.path.join(REPO, "posthan_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote posthan_analysis.json")

if __name__ == "__main__":
    main()
