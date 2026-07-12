#!/usr/bin/env python3
"""ng_archive_stats.py -- HANDOFF_ng_results_validation §2 archive-level stats.

For every run the ng-pipeline campaign actually produced (run dirs are read
from the ``Created output directory:`` lines of the master logs, so legacy
runs sharing the same experiment folders are excluded), load the saved
repertoire and report per target / genotype / depth / phase:

  * valid cells, sub-Gaussian cells (<O> < analytic G  AND  < G_N);
  * delta_ng (descriptor axis 3), effective-photon (axis 2) and max-PNR
    (axis 1) distributions of the sub-Gaussian winners;
  * the global best <O> with its descriptors and provenance;
  * sanity: sub-Gaussian cells with ~0 effective photons (decoupled-photon
    exploit -- the scorer should have marked these invalid; any hit = bug),
    log10 P positivity, non-finite fitness rows.

Outputs: a per-run CSV, a per-stratum markdown table, and a JSON with the
top-N sub-Gaussian candidates per target (provenance = run dir + flat cell
index) to feed the §3 validation stage.

Usage:
  python scripts/ng_archive_stats.py \
      --logs ngpipe_master_a1b1.log ngpipe_master_a273b11.log \
             ngpipe_master_a141b11.log ngpipe_master_a0b1.log \
      [--top 40] [--out-prefix ng_archive_stats] \
      [--state FILE.jsonl --max-seconds 35]

RESUMABLE: with ``--state``, per-run results are appended to a JSONL file and
already-analysed runs are skipped, so the scan can be driven in short
time-boxed invocations (``--max-seconds``); when every run is in the state
file the aggregate tables are (re)written.  Requires only numpy + the
tolerant loaders (pareto_report / rescore_all_experiments); safe on CPU
boxes without qdax/jax GPU.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

# per-target references, from the campaign startup blocks (audit step);
# keys are the experiment-dir target suffixes.
REFS = {
    "a1p00_b1p00": dict(alpha="1.0", beta="1+0j", G=0.666667, GN=0.710782),
    "a1p41_b1p41": dict(alpha="1.4142136", beta="1+1j", G=0.959560, GN=0.997924),
    "a2p73_b1p41": dict(alpha="2.7320508", beta="1+1j", G=1.089316, GN=1.124977),
    "a0p00_b1p00": dict(alpha="0.0", beta="1+0j", G=0.666667, GN=0.710779),
}

TAG_RE = re.compile(r"^\[(c\d+)_(d\d+)_([A-Za-z0-9_]+)\]\s?(.*)$")
DIR_RE = re.compile(r"Created output directory: (\S+)")


def campaign_runs(log_paths):
    """Map run-dir (relative, as logged) -> (cycle, depth, phase)."""
    runs = {}
    for path in log_paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = TAG_RE.match(line)
                if not m:
                    continue
                d = DIR_RE.search(m.group(4))
                if d:
                    runs[d.group(1)] = (m.group(1), m.group(2), m.group(3))
    return runs


def target_of(run_dir):
    for suffix in REFS:
        if suffix in run_dir:
            return suffix
    return None


def genotype_of(run_dir):
    base = os.path.basename(os.path.dirname(run_dir))
    return base.split("_")[0]


def load_arrays(pkl):
    from rescore_all_experiments import load_run_arrays
    return load_run_arrays(pkl)


def analyse_run(run_dir, ref, coupling_photon_min=0.5):
    pkl = os.path.join(run_dir, "results.pkl")
    if not os.path.exists(pkl):
        return None
    gen, fit, des = load_arrays(pkl)
    n_obj = fit.shape[1]
    exp = -fit[:, 0]
    logp = fit[:, 1] if n_obj > 1 else np.zeros(len(fit))
    valid = np.isfinite(fit[:, 0]) & (fit[:, 0] > -1e9)
    v = np.where(valid)[0]
    out = dict(n_cells=int(len(fit)), n_valid=int(valid.sum()), n_obj=n_obj)
    if v.size == 0:
        return out
    ev, pv = exp[v], logp[v]
    sub = v[(ev < ref["G"]) & (ev < ref["GN"])]
    out["n_subG"] = int(sub.size)
    out["best_exp"] = float(ev.min())
    bi = v[np.argmin(ev)]
    out["best_idx"] = int(bi)
    out["best_logp"] = float(logp[bi])
    out["best_des"] = [float(x) for x in des[bi]] if des is not None else None
    out["logp_positive"] = int((pv > 1e-9).sum())
    out["logp_below_-40"] = int((pv < -40).sum())
    if sub.size:
        dsub = des[sub]
        out["sub_dng_min"] = float(dsub[:, 3].min()) if dsub.shape[1] > 3 else None
        out["sub_dng_med"] = float(np.median(dsub[:, 3])) if dsub.shape[1] > 3 else None
        out["sub_phot_min"] = float(dsub[:, 2].min())
        out["sub_phot_med"] = float(np.median(dsub[:, 2]))
        out["sub_pnr_max"] = float(dsub[:, 1].max())
        out["n_subG_zero_photon"] = int((dsub[:, 2] < coupling_photon_min).sum())
        out["n_subG_zero_dng"] = (int((dsub[:, 3] <= 0).sum())
                                  if dsub.shape[1] > 3 else None)
        # top candidates by <O>
        order = sub[np.argsort(exp[sub])]
        out["_sub_sorted"] = [(int(i), float(exp[i]), float(logp[i]),
                               [float(x) for x in des[i]]) for i in order]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--experiments-root", default=os.path.join(REPO))
    ap.add_argument("--top", type=int, default=40,
                    help="top-N sub-Gaussian candidates per target for §3")
    ap.add_argument("--out-prefix", default="ng_archive_stats")
    ap.add_argument("--state", default=None,
                    help="JSONL scan state for resumable batched runs")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="stop scanning new runs after this many seconds (0=off)")
    ap.add_argument("--sub-cap", type=int, default=400,
                    help="store at most N best subG rows per run in state")
    args = ap.parse_args(argv)

    runs = campaign_runs(args.logs)
    print(f"{len(runs)} campaign run dirs found in logs")

    # ---- resumable per-run scan ------------------------------------------
    done = {}
    if args.state and os.path.exists(args.state):
        with open(args.state) as fh:
            for line in fh:
                rec = json.loads(line)
                done[rec["run"]] = rec
    t0 = time.time()
    state_fh = open(args.state, "a") if args.state else None
    pending = [r for r in sorted(runs) if r not in done]
    for rel in pending:
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"[time-box] stopping; {len(done)}/{len(runs)} runs scanned")
            break
        cyc, dep, phase = runs[rel]
        tgt = target_of(rel)
        if tgt is None:
            rec = dict(run=rel, error="unknown target")
        else:
            try:
                r = analyse_run(os.path.join(args.experiments_root, rel),
                                REFS[tgt])
                if r is None:
                    rec = dict(run=rel, error="no results.pkl")
                else:
                    r["_sub_sorted"] = r.get("_sub_sorted", [])[: args.sub_cap]
                    rec = dict(run=rel, target=tgt,
                               genotype=genotype_of(rel), cycle=cyc,
                               depth=dep, phase=phase, **r)
            except Exception as e:  # noqa: BLE001
                rec = dict(run=rel, error=repr(e))
        done[rel] = rec
        if state_fh:
            state_fh.write(json.dumps(rec) + "\n")
            state_fh.flush()
    if state_fh:
        state_fh.close()
    if len(done) < len(runs):
        print(f"[partial] {len(done)}/{len(runs)} — rerun to continue")
        return 1

    # ---- aggregate --------------------------------------------------------
    rows = []
    strata = defaultdict(lambda: dict(n_valid=0, n_subG=0, best=math.inf,
                                      best_run=None, zero_phot=0, zero_dng=0,
                                      runs=0, missing=0))
    top_per_target = defaultdict(list)
    for rel in sorted(runs):
        rec = done[rel]
        cyc, dep, phase = runs[rel]
        tgt = rec.get("target") or target_of(rel) or "?"
        gtype = rec.get("genotype") or genotype_of(rel)
        key = (tgt, gtype, dep, phase.split("_")[0] if "_" in phase else phase)
        strata[key]["runs"] += 1
        if rec.get("error"):
            print(f"  [unloadable] {rel}: {rec['error']}")
            strata[key]["missing"] += 1
            continue
        rows.append(dict(run=rel, target=tgt, genotype=gtype, cycle=cyc,
                         depth=dep, phase=phase,
                         **{k: v for k, v in rec.items()
                            if not k.startswith("_") and k not in
                            ("run", "target", "genotype", "cycle", "depth",
                             "phase")}))
        s = strata[key]
        s["n_valid"] += rec.get("n_valid", 0)
        s["n_subG"] += rec.get("n_subG", 0) or 0
        s["zero_phot"] += rec.get("n_subG_zero_photon", 0) or 0
        s["zero_dng"] += rec.get("n_subG_zero_dng", 0) or 0
        if rec.get("best_exp", math.inf) < s["best"]:
            s["best"] = rec["best_exp"]
            s["best_run"] = rel
        for idx, e, lp, dsc in rec.get("_sub_sorted", []):
            top_per_target[tgt].append(
                dict(run=rel, cell=idx, exp=e, logp=lp, des=dsc,
                     genotype=gtype, depth=dep, cycle=cyc, phase=phase))

    # ---- outputs ----------------------------------------------------------
    csv_path = f"{args.out_prefix}_runs.csv"
    if rows:
        keys = sorted({k for row in rows for k in row})
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    md = [f"# NG campaign archive stats\n",
          "| target | genotype | depth | phase | runs | valid cells | subG cells "
          "| subG 0-photon | subG dng<=0 | best <O> | best run |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(strata):
        s = strata[key]
        best = "-" if s["best"] is math.inf else f"{s['best']:.4f}"
        md.append("| " + " | ".join([*key, str(s["runs"]), str(s["n_valid"]),
                                     str(s["n_subG"]), str(s["zero_phot"]),
                                     str(s["zero_dng"]), best,
                                     str(s["best_run"])]) + " |")
    for tgt, cands in top_per_target.items():
        cands.sort(key=lambda c: c["exp"])
        ref = REFS[tgt]
        md.append(f"\n## {tgt}: analytic G = {ref['G']}, G_N = {ref['GN']}, "
                  f"total subG rows (pre-dedupe) = {len(cands)}\n")
        md.append("| # | <O> | log10P | dng | photons | maxPNR | active | "
                  "geno | depth | phase | run | cell |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        seen = set()
        shown = 0
        for c in cands:
            sig = (round(c["exp"], 6), round(c["logp"], 6))
            if sig in seen:
                continue
            seen.add(sig)
            d = c["des"]
            dng = f"{d[3]:.3f}" if len(d) > 3 else "-"
            md.append(f"| {shown} | {c['exp']:.5f} | {c['logp']:.2f} | {dng} | "
                      f"{d[2]:.2f} | {d[1]:.0f} | {d[0]:.0f} | {c['genotype']} | "
                      f"{c['depth']} | {c['phase']} | {c['run']} | {c['cell']} |")
            shown += 1
            if shown >= args.top:
                break
    md_path = f"{args.out_prefix}.md"
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")
    json_path = f"{args.out_prefix}_top.json"
    with open(json_path, "w") as fh:
        json.dump({t: c[: max(args.top * 5, 200)]
                   for t, c in top_per_target.items()}, fh, indent=1)
    print(f"Wrote {csv_path}, {md_path}, {json_path}")


if __name__ == "__main__":
    main()
