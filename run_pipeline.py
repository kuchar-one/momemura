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
    args, remainder = parser.parse_known_args()

    # If user asked for help
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("Usage: python run_pipeline.py [any run_mome.py arguments...]")
        sys.exit(0)

    base_mome_args = remainder

    print("=" * 60)
    print("Starting Optimization Pipeline")
    print(f"Pass-through Arguments: {' '.join(base_mome_args)}")
    print(f"Steps: {STEPS} (Splits from 90-10 to 0-100)")
    print("=" * 60)

    # 0.9 to 0.0 inclusive (Start: 90% Prob, End: Full Exp)
    alpha_probs = np.linspace(0.9, 0.0, STEPS)

    for i, alpha_p in enumerate(alpha_probs):
        alpha_e = 1.0 - alpha_p

        desc = f"Split {i + 1}/{STEPS}: Exp={alpha_e:.1f}, Prob={alpha_p:.1f}"
        print(f"\n[Pipeline] === Starting {desc} ===\n")

        # --- Phase 1: Single Objective (Parallel) ---
        print(
            f"[Pipeline] Launching {SINGLE_OBJ_PARALLELISM} Parallel Single-Objective Runs..."
        )

        current_processes = []
        process_info_list = []
        skip_requested = False

        single_obj_args = base_mome_args + [
            "--mode",
            "single",
            "--alpha-expectation",
            str(alpha_e),
            "--alpha-probability",
            str(alpha_p),
            "--global-seed-scan",
        ]

        for k in range(SINGLE_OBJ_PARALLELISM):
            # Unique Seed Logic
            worker_seed = int(time.time()) + i * 100 + k
            worker_args = list(single_obj_args)
            if "--seed" in worker_args:
                try:
                    idx = worker_args.index("--seed")
                    worker_args.pop(idx)
                    worker_args.pop(idx)
                except Exception:
                    pass
            worker_args.extend(["--seed", str(worker_seed)])

            proc_tuple = run_watchdog_command(worker_args, f"step_{i}_single_{k}")
            process_info_list.append(proc_tuple)
            current_processes.append(proc_tuple[0])

            # Add delay to ensure unique timestamps for output directories
            time.sleep(10)

        # Monitor Parallel Process
        print("[Pipeline] Streaming output for Single Objective runs...")
        monitor_processes(process_info_list)
        print("[Pipeline] Single Objective Phase Complete.")

        if skip_requested:
            print("[Pipeline] Resume normal flow after skip.")
            skip_requested = False

        # --- Phase 2: QDAX MOME ---
        print("\n[Pipeline] Launching QDAX MOME Run...")

        qdax_args = base_mome_args + ["--mode", "qdax", "--global-seed-scan"]

        current_processes = []
        proc_tuple = run_watchdog_command(qdax_args, f"step_{i}_qdax")
        current_processes.append(proc_tuple[0])

        # Monitor Sequential Process
        monitor_processes([proc_tuple])

        print("[Pipeline] QDAX Phase Complete.")

        if skip_requested:
            print("[Pipeline] Resume normal flow after skip.")
            skip_requested = False

    print("\n[Pipeline] All Steps Completed.")


if __name__ == "__main__":
    main()
