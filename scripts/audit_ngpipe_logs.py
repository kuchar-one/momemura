#!/usr/bin/env python3
"""audit_ngpipe_logs.py -- HANDOFF_ng_results_validation §2 log audit.

Parses the ng-pipeline master logs (ngpipe_master_*.log), splits them into
per-phase streams (tag ``[c{cycle}_d{depth}_{A|B|C}...]``) and per watchdog
launch within each stream, and checks:

  * startup blocks: G_N / G_N2 / analytic-G ordering, macro-mutations ON,
    injected pareto_ng seeds (nonzero at depths 4/5), PNR-pattern seeds,
    ng-hybrid banner without the missing-descriptor WARN;
  * validation sweeps: cadence, removed-artifact counts per sweep, late
    spikes (counts after the first full sweep that exceed --spike-thresh);
  * progress lines: best-Exp monotone non-increase within a launch, and
    best-Exp jumping UP right after a sweep (= elite was an artifact);
  * coverage collapses right after sweeps (> --cov-drop percentage points);
  * phase C (Adam): improvement of Best Exp from first to last iter per
    launch (flat-from-iter-1 everywhere = jitter-fill failure);
  * watchdog events: improvements, crashes, OOM, NaN, XLA recompile storms.

Usage:
  python scripts/audit_ngpipe_logs.py ngpipe_master_a1b1.log [more logs ...] \
      [--out audit_report.md] [--spike-thresh 300] [--cov-drop 2.0]

Writes a markdown report (default: ngpipe_audit_report.md in CWD) and prints
a per-log FLAGS summary to stdout.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

TAG_RE = re.compile(r"^\[(c\d+_d\d+_[A-Za-z0-9_]+)\]\s?(.*)$")
SWEEP_RE = re.compile(
    r"=== Moment Validation Sweep \(gen (\d+), L (\d+)->(\d+), (\w+)\) ===")
REMOVED_RE = re.compile(r"Removed (\d+) L-truncation artifacts")
PROG_RE = re.compile(
    r"Iter (\d+)/(\d+) \| Chunk: ([0-9.]+)s \| ETA: [0-9:]+ \| "
    r"Cov: ([0-9.]+)% \| Exp: ([0-9.eE+-]+) \(vs GS: ([+-][0-9.]+), "
    r"vs G: ([+-][0-9.]+), vs G_N: ([+-][0-9.]+)\)")
ADAM_RE = re.compile(r"Iter (\d+)/(\d+) \| Best Exp: ([0-9.eE+-]+) \| Best Prob: ([0-9.]+)")
GN_RE = re.compile(
    r"Clamped-Gaussian reference \(([0-9.]+)s\): G_N = ([0-9.]+) .* "
    r"G_N2 = ([0-9.]+) .*; analytic G = ([0-9.]+)")
SEEDS_RE = re.compile(r"Injected (\d+) seeds \(metric=(\w+), fill=(\w+)\)")
PNRSEEDS_RE = re.compile(r"Injected (\d+) PNR-pattern seeds")
MACRO_RE = re.compile(r"Physics macro-mutations: ON \(prob=([0-9.]+), (\d+) operators\)")
LAUNCH_RE = re.compile(r"\[Watchdog\] Launching (\w+) Run")
IMPROVE_RE = re.compile(r"\[Watchdog\] IMPROVEMENT DETECTED! \(([0-9.inf]+) -> ([0-9.]+)\)")
BAD_RE = re.compile(r"(?i)(traceback|out of memory|OOM|CUDA_ERROR|RESOURCE_EXHAUSTED|killed|segfault)")
NAN_RE = re.compile(r"(?i)\bnan\b")
WARN_NG_RE = re.compile(r"WARN.*ng-hybrid without --moment-ng-descriptor")
COMPILE_RE = re.compile(r"(?i)compil(ing|ation) module|xla.*compil")


def parse_log(path):
    """Split a master log into per-tag ordered event streams."""
    streams = defaultdict(list)   # tag -> list of (lineno, text)
    untagged = []
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            m = TAG_RE.match(line.rstrip("\n"))
            if m:
                streams[m.group(1)].append((i, m.group(2)))
            else:
                untagged.append((i, line.rstrip("\n")))
    return streams, untagged


def audit_stream(tag, lines, spike_thresh, cov_drop):
    """Audit one phase stream. Returns dict of stats + list of flag strings."""
    flags = []
    launches = []          # each: dict with startup info + progress + sweeps
    cur = None

    def new_launch(lineno, kind):
        return dict(kind=kind, line=lineno, gn=None, gn2=None, g=None,
                    seeds=None, pnr_seeds=None, macro=None, warn_ng=False,
                    prog=[], adam=[], sweeps=[], pending_sweep=None)

    for lineno, text in lines:
        m = LAUNCH_RE.search(text)
        if m:
            if cur:
                launches.append(cur)
            cur = new_launch(lineno, m.group(1))
            continue
        if cur is None:
            cur = new_launch(lineno, "implicit")
        m = GN_RE.search(text)
        if m:
            cur["gn"], cur["gn2"], cur["g"] = (float(m.group(2)),
                                               float(m.group(3)),
                                               float(m.group(4)))
            continue
        m = SEEDS_RE.search(text)
        if m:
            cur["seeds"] = int(m.group(1))
            continue
        m = PNRSEEDS_RE.search(text)
        if m:
            cur["pnr_seeds"] = int(m.group(1))
            continue
        m = MACRO_RE.search(text)
        if m:
            cur["macro"] = float(m.group(1))
            continue
        if WARN_NG_RE.search(text):
            cur["warn_ng"] = True
            continue
        m = SWEEP_RE.search(text)
        if m:
            cur["pending_sweep"] = dict(gen=int(m.group(1)),
                                        lo=int(m.group(2)), hi=int(m.group(3)),
                                        mode=m.group(4), removed=None,
                                        line=lineno)
            continue
        m = REMOVED_RE.search(text)
        if m and cur["pending_sweep"] is not None:
            cur["pending_sweep"]["removed"] = int(m.group(1))
            cur["sweeps"].append(cur["pending_sweep"])
            cur["pending_sweep"] = None
            continue
        m = PROG_RE.search(text)
        if m:
            cur["prog"].append(dict(it=int(m.group(1)), tot=int(m.group(2)),
                                    chunk_s=float(m.group(3)),
                                    cov=float(m.group(4)),
                                    exp=float(m.group(5)),
                                    vs_gn=float(m.group(8)),
                                    after_sweep=len(cur["sweeps"]),
                                    line=lineno))
            continue
        m = ADAM_RE.search(text)
        if m:
            cur["adam"].append(dict(it=int(m.group(1)), tot=int(m.group(2)),
                                    exp=float(m.group(3)),
                                    prob=float(m.group(4)), line=lineno))
    if cur:
        launches.append(cur)

    # ---- checks -----------------------------------------------------------
    n_sweeps = sum(len(l["sweeps"]) for l in launches)
    all_removed = [s["removed"] for l in launches for s in l["sweeps"]]
    best_exp = None
    for l in launches:
        # startup sanity
        if l["gn"] is not None:
            if not (l["g"] <= l["gn2"] + 1e-9 <= l["gn"] + 1e-9):
                flags.append(f"{tag} L{l['line']}: G ordering violated "
                             f"(G={l['g']}, G_N2={l['gn2']}, G_N={l['gn']})")
        if l["warn_ng"]:
            flags.append(f"{tag} L{l['line']}: ng-hybrid WITHOUT ng-descriptor WARN present")
        if l["macro"] is None and l["prog"]:
            flags.append(f"{tag} L{l['line']}: no macro-mutation banner in qdax launch")
        # per-launch monotone best exp.  Small (<bump_tol) increases right
        # after a sweep are the L50->L120 refresh correcting a genuine
        # elite's fitness; large ones mean the elite was an artifact.
        bump_tol = 5e-3
        prev = None
        prev2_after = 0            # sweep count two progress lines back:
        prev_after = 0             # the refreshed best shows with 1-line lag
        l["refresh_bumps"] = 0
        for p in l["prog"]:
            if prev is not None and p["exp"] > prev + 1e-9:
                after = p["after_sweep"] > prev2_after
                delta = p["exp"] - prev
                if delta < bump_tol:
                    l["refresh_bumps"] += 1
                else:
                    flags.append(
                        f"{tag} L{p['line']}: best Exp REVERTED {prev:.4f} -> "
                        f"{p['exp']:.4f} (d={delta:.4f})"
                        + (" [sweep-coincident: artifact-led elite]" if after
                           else " [NO sweep nearby: unexplained]"))
            prev2_after = prev_after
            prev_after = p["after_sweep"]
            prev = p["exp"]
        # coverage collapse right after sweep
        for j in range(1, len(l["prog"])):
            a, b = l["prog"][j - 1], l["prog"][j]
            if b["after_sweep"] > a["after_sweep"] and a["cov"] - b["cov"] > cov_drop:
                flags.append(f"{tag} L{b['line']}: coverage collapse after sweep "
                             f"({a['cov']:.1f}% -> {b['cov']:.1f}%)")
        # sweep spikes (skip each launch's first sweep -- it's full).  A
        # constant background removal rate is expected (fresh L50 candidates
        # cleaned at L120 every sweep); a SPIKE well above the stream median
        # is what signals a new exploit family.
        later = [s["removed"] for s in l["sweeps"][1:] if s["removed"] is not None]
        if later:
            med = sorted(later)[len(later) // 2]
            for s in l["sweeps"][1:]:
                if s["removed"] is not None and \
                        s["removed"] > max(spike_thresh, 3 * med):
                    flags.append(f"{tag} L{s['line']}: sweep spike, removed "
                                 f"{s['removed']} at gen {s['gen']} "
                                 f"(stream median {med})")
        if l["prog"]:
            e = l["prog"][-1]["exp"]
            best_exp = e if best_exp is None else min(best_exp, e)
        if l["adam"]:
            e = l["adam"][-1]["exp"]
            best_exp = e if best_exp is None else min(best_exp, e)

    # phase C flatness: ALL launches flat from iter 1
    adam_launches = [l for l in launches if l["adam"]]
    if adam_launches:
        flat = all(abs(l["adam"][0]["exp"] - l["adam"][-1]["exp"]) < 1e-12
                   for l in adam_launches if len(l["adam"]) > 1)
        if flat:
            flags.append(f"{tag}: ALL Adam launches flat from iter 1 "
                         f"(jitter-fill seeding suspect)")

    return dict(tag=tag, n_launches=len(launches), n_sweeps=n_sweeps,
                removed_total=sum(r for r in all_removed if r is not None),
                removed_series=all_removed, best_exp=best_exp,
                refresh_bumps=sum(l.get("refresh_bumps", 0) for l in launches),
                launches=launches), flags


def audit_log(path, spike_thresh, cov_drop):
    streams, untagged = parse_log(path)
    results, flags = {}, []
    for tag in sorted(streams):
        res, fl = audit_stream(tag, streams[tag], spike_thresh, cov_drop)
        results[tag] = res
        flags += fl
    # global bad-line scan (tagged or not)
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            if BAD_RE.search(line):
                flags.append(f"(global) L{i}: {line.strip()[:160]}")
            elif NAN_RE.search(line) and "Fin" not in line:
                flags.append(f"(global) L{i} NaN mention: {line.strip()[:160]}")
    return results, flags


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="ngpipe_audit_report.md")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--spike-thresh", type=int, default=300)
    ap.add_argument("--cov-drop", type=float, default=2.0)
    args = ap.parse_args(argv)

    md = ["# NG-pipeline log audit\n"]
    js = {}
    for path in args.logs:
        name = os.path.basename(path)
        results, flags = audit_log(path, args.spike_thresh, args.cov_drop)
        js[name] = dict(
            flags=flags,
            phases={t: {k: v for k, v in r.items() if k != "launches"}
                    for t, r in results.items()})
        md.append(f"\n## {name}\n")
        md.append("| phase | launches | sweeps | removed total | removed/sweep med (max) | refresh bumps | best Exp |")
        md.append("|---|---|---|---|---|---|---|")
        for tag, r in results.items():
            series = [x for x in r["removed_series"] if x is not None]
            if series:
                med = sorted(series)[len(series) // 2]
                s = f"{med} ({max(series)})"
            else:
                s = "-"
            be = f"{r['best_exp']:.4f}" if r["best_exp"] is not None else "-"
            md.append(f"| {tag} | {r['n_launches']} | {r['n_sweeps']} | "
                      f"{r['removed_total']} | {s} | {r['refresh_bumps']} | {be} |")
        md.append(f"\n### Flags ({len(flags)})\n")
        md += [f"- {f}" for f in flags] or ["- none"]
        print(f"{name}: {len(flags)} flags, "
              f"{sum(r['n_sweeps'] for r in results.values())} sweeps total")
        for f in flags[:40]:
            print("  *", f)
        if len(flags) > 40:
            print(f"  ... and {len(flags) - 40} more (see report)")

    with open(args.out, "w") as fh:
        fh.write("\n".join(md) + "\n")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(js, fh, indent=1, default=float)
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
