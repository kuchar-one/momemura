"""NG Pipeline: physics-motivated, NG-aware, dynamic-depth breeding search.

Per cycle, per depth (default 3 -> 4 -> 5):

  [A] DISCOVERY      run_mome qdax, genotype B30F (forced heralding: the
                     Gaussian manifold is unrepresentable), ng-hybrid emitter
                     (uniform + fitness-elite + NG-elite), physics
                     macro-mutations, canonical PNR-pattern seeds, log-binned
                     NG descriptor, NG exploration reward ON.

  [B] CONSOLIDATION  run_mome qdax, genotype B30 (free), seeded from the
                     B30F discovery archives (lossless conversion) + everything
                     shallower (exact depth embedding), NG reward annealed low.

  [C] POLISH         short parallel Adam runs (run_mome single) seeded from the
                     NG-stratified Pareto set with jitter-fill: pure continuous
                     exploitation of every basin the MOME phases found
                     (discrete genes have zero gradient, so structure is
                     frozen -- by design).

Results of every phase land in output/experiments/ and are picked up by the
next phase's global seed scan; depth d+1 embeds depth-d elites exactly, so no
progress is ever lost when the tree grows.

Usage (pass-through args go to run_mome.py):
    python run_pipeline_ng.py --pop 64 --cutoff 30 --modes 3 --pnr-max 10 \
        --target-alpha 1.0 --target-beta 1.0 [--dry-run] [...]

Controls:  --depth-schedule 3,4,5   --cycles 2
           --iters-schedule 4000,1500,800   (MOME iters per depth)
           --adam-iters 300 --adam-parallel 2 --adam-lr 0.02
           --ng-high 0.3 --ng-low 0.1
           --skip-phases A,B,C subsets via --phases (default ABC)
SIGUSR1 skips the current phase gracefully (children save + exit).
"""

import argparse
import os
import select
import signal
import subprocess
import sys
import time
from typing import List

current_processes: List[subprocess.Popen] = []
skip_requested = False


def _signal_skip(signum, frame):
    global skip_requested
    print("\n[NGPipe] SIGUSR1 received -- skipping current phase...")
    skip_requested = True
    for p in current_processes:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass


def _launch(cmd_args: List[str], log_prefix: str, extra_env: dict = None):
    full_cmd = [sys.executable, "-u", "watchdog_restart.py"] + cmd_args
    log_file = f"ngpipe_{log_prefix}_{int(time.time())}.log"
    env = dict(os.environ)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    print(f"[{log_prefix}] Logging to {log_file}")
    p = subprocess.Popen(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, bufsize=0, env=env,
    )
    return p, log_file, log_prefix


def _monitor(process_info):
    """Stream child output to console (prefixed) and per-process logs."""
    if not process_info:
        return
    log_handles, proc_map, readers = {}, {}, []
    try:
        for p, log_path, prefix in process_info:
            f = open(log_path, "wb")
            log_handles[p.stdout.fileno()] = f
            proc_map[p.stdout.fileno()] = (p, prefix)
            readers.append(p.stdout.fileno())
        while readers:
            readable, _, _ = select.select(readers, [], [], 1.0)
            for fd in readable:
                p, prefix = proc_map[fd]
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    data = b""
                if not data:
                    readers.remove(fd)
                    continue
                log_handles[fd].write(data)
                log_handles[fd].flush()
                try:
                    for line in data.decode("utf-8", "replace").splitlines():
                        if line.strip():
                            print(f"[{prefix}] {line}")
                except Exception:
                    pass
    finally:
        for f in log_handles.values():
            f.close()
    for p, _, _ in process_info:
        p.wait()


def _run_phase(cmd_args, prefix, extra_env=None, dry=False):
    global current_processes, skip_requested
    skip_requested = False
    if dry:
        print(f"[DRY] {prefix}: watchdog_restart.py {' '.join(cmd_args)}")
        return
    info = _launch(cmd_args, prefix, extra_env)
    current_processes = [info[0]]
    _monitor([info])
    current_processes = []


def _run_parallel(cmd_list, prefixes, extra_env=None, dry=False):
    global current_processes
    if dry:
        for args, pre in zip(cmd_list, prefixes):
            print(f"[DRY] {pre}: watchdog_restart.py {' '.join(args)}")
        return
    infos = []
    for args, pre in zip(cmd_list, prefixes):
        infos.append(_launch(args, pre, extra_env))
        time.sleep(10)  # unique output-dir timestamps
    current_processes = [i[0] for i in infos]
    _monitor(infos)
    current_processes = []


def main():
    signal.signal(signal.SIGUSR1, _signal_skip)

    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth-schedule", default="3,4,5",
                        help="Comma list of tree depths, shallow -> deep.")
    parser.add_argument("--iters-schedule", default="4000,1500,800",
                        help="MOME iterations per depth (matched to schedule).")
    parser.add_argument("--cycles", type=int, default=2,
                        help="Full growth cycles (later cycles re-seed from "
                             "everything the earlier ones found).")
    parser.add_argument("--phases", default="ABC",
                        help="Which phases to run per depth: A=discovery(B30F), "
                             "B=consolidation(B30), C=Adam polish.")
    parser.add_argument("--adam-iters", type=int, default=300)
    parser.add_argument("--adam-parallel", type=int, default=2)
    parser.add_argument("--adam-lr", type=float, default=0.02)
    parser.add_argument("--ng-high", type=float, default=0.3,
                        help="alpha-nongauss for discovery phases.")
    parser.add_argument("--ng-low", type=float, default=0.1,
                        help="alpha-nongauss for consolidation phases.")
    parser.add_argument("--macro-prob-discovery", type=float, default=0.35)
    parser.add_argument("--macro-prob-consolidation", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command instead of running.")
    # swallowed if present in the pass-through (pipeline controls these)
    parser.add_argument("--mode", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--genotype", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--depth", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--iters", type=int, help=argparse.SUPPRESS)
    args, base = parser.parse_known_args()

    depths = [int(d) for d in args.depth_schedule.split(",") if d]
    iters = [int(i) for i in args.iters_schedule.split(",") if i]
    if len(iters) < len(depths):
        iters = iters + [iters[-1]] * (len(depths) - len(iters))

    # pnr seeds scale with the population if provided (default 64)
    pop = 64
    if "--pop" in base:
        pop = int(base[base.index("--pop") + 1])

    common = base + [
        "--seed-scan", "--global-seed-scan",
        "--moment-ng-descriptor",
        "--seed-metric", "pareto_ng",
    ]

    print("=" * 66)
    print("NG PIPELINE  (forced-heralding discovery -> consolidation -> polish)")
    print(f"  depths {depths} | iters {iters[:len(depths)]} | cycles {args.cycles}")
    print(f"  phases {args.phases} | NG reward {args.ng_high}->{args.ng_low}->0")
    print(f"  pass-through: {' '.join(base)}")
    print("=" * 66)

    # restart-free long runs inside the watchdog
    wd_env = {"STAGNATION_LIMIT": os.environ.get("STAGNATION_LIMIT", "100000"),
              "WATCHDOG_START_LONG": os.environ.get("WATCHDOG_START_LONG", "1")}

    for cycle in range(args.cycles):
        for di, depth in enumerate(depths):
            n_it = iters[di]
            tag = f"c{cycle}_d{depth}"

            if "A" in args.phases:
                print(f"\n[NGPipe] === {tag} PHASE A: DISCOVERY "
                      f"(B30F, forced heralding, {n_it} iters) ===\n")
                disc = common + [
                    "--mode", "qdax",
                    "--genotype", "B30F",
                    "--depth", str(depth),
                    "--iters", str(n_it),
                    "--emitter", "ng-hybrid",
                    "--macro-prob", str(args.macro_prob_discovery),
                    "--pnr-seeds", str(max(pop // 2, 8)),
                    "--seed-accept", "B30",
                    "--alpha-nongauss", str(args.ng_high),
                ]
                _run_phase(disc, f"{tag}_A_b30f", wd_env, args.dry_run)

            if "B" in args.phases:
                print(f"\n[NGPipe] === {tag} PHASE B: CONSOLIDATION "
                      f"(B30, free, {n_it} iters) ===\n")
                cons = common + [
                    "--mode", "qdax",
                    "--genotype", "B30",
                    "--depth", str(depth),
                    "--iters", str(n_it),
                    "--emitter", "ng-hybrid",
                    "--macro-prob", str(args.macro_prob_consolidation),
                    "--pnr-seeds", str(max(pop // 4, 4)),
                    "--seed-accept", "B30F",
                    "--alpha-nongauss", str(args.ng_low),
                ]
                _run_phase(cons, f"{tag}_B_b30", wd_env, args.dry_run)

            if "C" in args.phases:
                print(f"\n[NGPipe] === {tag} PHASE C: ADAM POLISH "
                      f"({args.adam_parallel} x {args.adam_iters} iters, "
                      f"NG-stratified Pareto multi-start) ===\n")
                cmds, prefixes = [], []
                for k in range(args.adam_parallel):
                    pol = common + [
                        "--mode", "single",
                        "--genotype", "B30",
                        "--depth", str(depth),
                        "--iters", str(args.adam_iters),
                        "--seed-accept", "B30F",
                        "--seed-fill", "jitter",
                        "--seed-jitter", str(0.02 * (k + 1)),
                        "--adam-lr", str(args.adam_lr),
                        "--alpha-expectation", "1.0",
                        "--alpha-probability", "0.0",
                        "--alpha-nongauss", "0.0",
                        "--seed", str(int(time.time()) + 1000 * k),
                    ]
                    cmds.append(pol)
                    prefixes.append(f"{tag}_C_adam{k}")
                _run_parallel(cmds, prefixes, wd_env, args.dry_run)

    print("\n[NGPipe] All cycles complete.")


if __name__ == "__main__":
    main()
