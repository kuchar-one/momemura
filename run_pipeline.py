"""
Pipeline for running a sequence of MOME and Single-Objective optimizations.

Sequence:
1. 100% Exp, 0% Prob (Single Obj x2)
2. QDAX MOME (Full)
3. 90% Exp, 10% Prob (Single Obj x2)
4. QDAX MOME (Full)
...
21. 0% Exp, 100% Prob (Single Obj x2)
22. QDAX MOME (Full)

Features:
- Validates SIGUSR1 to skip the current step gracefully (triggers save in subprocesses).
- Runs Single Objective phases in parallel (2 processes).
- Runs QDAX MOME phases sequentially.
- Uses watchdog_restart.py for robustness.
"""

import subprocess
import os
import sys
import time
import signal
import numpy as np
import argparse
import select
from typing import List

# --- Configuration ---
STEPS = 5  # 5 steps (matching scalarization)
SINGLE_OBJ_PARALLELISM = 2

# Global tracking for signal handling
current_processes: List[subprocess.Popen] = []
skip_requested = False


def signal_handler_skip(signum, frame):
    """
    Handles SIGUSR1 to skip the current step.
    Sends SIGTERM to all child processes to trigger their save-and-exit logic.
    """
    global skip_requested
    print("\n[Pipeline] SIGUSR1 Received! Requesting skip of current step...")
    skip_requested = True

    # Terminate known child processes
    for p in current_processes:
        if p.poll() is None:  # If still running
            print(f"[Pipeline] Sending SIGTERM to child PID {p.pid}...")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"[Pipeline] Error sending SIGTERM: {e}")


def run_watchdog_command(cmd_args: List[str], log_prefix: str) -> tuple:
    """
    Launches watchdog_restart.py with the given arguments.
    Returns (process, log_filename, log_prefix).
    """
    full_cmd = [sys.executable, "-u", "watchdog_restart.py"] + cmd_args

    log_file = f"pipeline_{log_prefix}_{int(time.time())}.log"
    print(f"[{log_prefix}] Logging to {log_file}")

    # Start process with pipes for stdout/stderr
    p = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        bufsize=0,  # Unbuffered
    )
    return p, log_file, log_prefix


def monitor_processes(process_info: List[tuple]):
    """
    Monitors a list of (process, log_file, prefix) tuples.
    Streaming output to console (prefixed) and to the respective log files.
    """
    if not process_info:
        return

    # Open log files
    log_handles = {}
    proc_map = {}
    readers = []

    try:
        for p, log_path, prefix in process_info:
            f = open(log_path, "wb")  # Binary mode for direct writing
            log_handles[p.stdout.fileno()] = f
            proc_map[p.stdout.fileno()] = (p, prefix)
            readers.append(p.stdout.fileno())

        while readers:
            # Check for readable pipes
            readable, _, _ = select.select(readers, [], [], 1.0)

            for fd in readable:
                p, prefix = proc_map[fd]

                try:
                    # Read chunk
                    data = os.read(fd, 4096)
                except OSError:
                    data = b""

                if not data:
                    # EOF
                    readers.remove(fd)
                    continue

                # Write to Log
                log_handles[fd].write(data)
                log_handles[fd].flush()

                # Print to Console (Line based is nicer but chunked is safer for pipes)
                # Decode for console
                try:
                    text_chunk = data.decode("utf-8", errors="replace")
                    # Ideally we want line buffering for console
                    # But raw streaming with prefix is acceptable
                    lines = text_chunk.splitlines()
                    for line in lines:
                        if line.strip():
                            print(f"[{prefix}] {line}")
                except Exception:
                    pass

            # Check for process exits
            # If a process is done and pipe is empty, we are good.
            # But the 'data' check above handles EOF.

            # Prune readers for dead processes if needed?
            # select loop handles pipe closure (EOF returns empty bytes)

    finally:
        # Close all logs
        for f in log_handles.values():
            f.close()

    # Wait for final exit codes?
    for p, _, _ in process_info:
        p.wait()


def main():
    global current_processes, skip_requested

    # ... (Argparse logic same)

    # Argparse to strip arguments we control and pass the rest
    parser = argparse.ArgumentParser(
        description="MOME Optimization Pipeline", add_help=False
    )
    parser.add_argument("--mode", type=str, help="Ignored (Pipeline controls mode)")
    parser.add_argument(
        "--alpha-expectation", type=float, help="Ignored (Pipeline controls weights)"
    )
    parser.add_argument(
        "--alpha-probability", type=float, help="Ignored (Pipeline controls weights)"
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run only one single-objective instance instead of two parallely",
    )
    args, remainder = parser.parse_known_args()

    # If user asked for help
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print(
            "Usage: python run_pipeline.py [--single-run] [any run_mome.py arguments...]"
        )
        sys.exit(0)

    base_mome_args = remainder

    # Determine Parallelism
    num_parallel = 1 if args.single_run else SINGLE_OBJ_PARALLELISM

    print("=" * 60)
    print("Starting Optimization Pipeline")
    print(f"Pass-through Arguments: {' '.join(base_mome_args)}")
    print(f"Single-Objective Parallelism: {num_parallel}")
    # --- Curriculum (exp-first) -------------------------------------------- #
    # The non-Gaussian advantage lives in <O> (exp) at LOW probability; starting
    # prob-heavy (the old 0.9->0.0 schedule) herds the early search into the
    # Gaussian corner (prob->1) and it never escapes. So sweep the Pareto front
    # from the <O> end: EXP-first (prob 0 early), ramping prob in later. Env
    # overrides: ALPHA_PROB_START / ALPHA_PROB_END. And anneal a non-Gaussianity
    # exploration reward (ALPHA_NONGAUSS_MAX -> 0) that bootstraps escape from the
    # Gaussian basin early and vanishes so the final optimum stays honest.
    p_start = float(os.environ.get("ALPHA_PROB_START", 0.0))
    p_end = float(os.environ.get("ALPHA_PROB_END", 0.9))
    ng_max = float(os.environ.get("ALPHA_NONGAUSS_MAX", 0.3))
    alpha_probs = np.linspace(p_start, p_end, STEPS)
    alpha_ngs = np.linspace(ng_max, 0.0, STEPS)
    print(f"Steps: {STEPS} (exp-first: prob {p_start:.2f}->{p_end:.2f}; "
          f"non-Gaussian reward {ng_max:.2f}->0)")
    print("=" * 60)

    # PIPELINE_QDAX_FIRST=1 runs the exploratory QDAX MOME phase BEFORE the
    # single-objective phase, so the single-obj runs seed from a diverse archive
    # of firing structures the emitter found (instead of the single-obj -- which
    # cannot explore firing at all -- going first from cold). Pair with
    # STAGNATION_LIMIT=<large> / WATCHDOG_START_LONG=1 for restart-free exploration.
    qdax_first = os.environ.get("PIPELINE_QDAX_FIRST", "0") == "1"

    for i, alpha_p in enumerate(alpha_probs):
        alpha_e = 1.0 - alpha_p
        alpha_ng = float(alpha_ngs[i])

        desc = f"Split {i + 1}/{STEPS}: Exp={alpha_e:.1f}, Prob={alpha_p:.1f}, NG={alpha_ng:.2f}"
        print(f"\n[Pipeline] === Starting {desc} ===\n")

        def run_single_phase():
            print(
                f"[Pipeline] Launching {num_parallel} "
                f"{'Parallel' if num_parallel > 1 else 'Single'} Single-Objective Run(s)..."
            )
            single_obj_args = base_mome_args + [
                "--mode", "single",
                "--alpha-expectation", str(alpha_e),
                "--alpha-probability", str(alpha_p),
                "--alpha-nongauss", str(alpha_ng),
                "--global-seed-scan",
            ]
            process_info_list = []
            for k in range(num_parallel):
                worker_seed = int(time.time()) + i * 100 + k
                worker_args = list(single_obj_args)
                if "--seed" in worker_args:
                    try:
                        idx = worker_args.index("--seed")
                        worker_args.pop(idx); worker_args.pop(idx)
                    except Exception:
                        pass
                worker_args.extend(["--seed", str(worker_seed)])
                proc_tuple = run_watchdog_command(worker_args, f"step_{i}_single_{k}")
                process_info_list.append(proc_tuple)
                time.sleep(10)  # unique output-dir timestamps
            print("[Pipeline] Streaming output for Single Objective runs...")
            monitor_processes(process_info_list)
            print("[Pipeline] Single Objective Phase Complete.")

        def run_qdax_phase():
            print("\n[Pipeline] Launching QDAX MOME Run...")
            qdax_args = base_mome_args + ["--mode", "qdax", "--global-seed-scan",
                                          "--alpha-nongauss", str(alpha_ng)]
            proc_tuple = run_watchdog_command(qdax_args, f"step_{i}_qdax")
            monitor_processes([proc_tuple])
            print("[Pipeline] QDAX Phase Complete.")

        if qdax_first:
            run_qdax_phase()
            run_single_phase()
        else:
            run_single_phase()
            run_qdax_phase()

    print("\n[Pipeline] All Steps Completed.")


if __name__ == "__main__":
    main()
