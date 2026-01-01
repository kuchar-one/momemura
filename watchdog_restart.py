import subprocess
import time
import sys
import re
import os
import signal
import fcntl
import select

# Logic Configuration
STAGNATION_LIMIT = 1000  # Iterations without improvement to trigger restart
POLL_INTERVAL = 1.0  # Seconds
CONSECUTIVE_NO_GAINS_LIMIT = 2  # Threshold to switch to Long Run


def set_non_blocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def run_optimization(args, run_type="SHORT", global_best=float("inf")):
    """
    Runs the optimization process.
    Returns: (final_best_exp, did_finish_naturally)
    """
    cmd = [sys.executable, "-u", "run_mome.py"] + args

    # Ensure seed-scan is present
    if "--seed-scan" not in cmd:
        cmd.append("--seed-scan")

    print(f"\n[Watchdog] Launching {run_type} Run...")
    print(f"[Watchdog] Command: {' '.join(cmd)}\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    set_non_blocking(proc.stdout.fileno())

    current_best = float("inf")
    last_imp_iter = 0
    current_iter = 0

    # Track logic
    time.time()

    try:
        while True:
            # Check if process exited
            ret = proc.poll()
            if ret is not None:
                print(f"\n[Watchdog] Process finished with code {ret}.")
                return current_best, True  # Finished naturally

            # Read Output
            reads = [proc.stdout.fileno()]
            ready = select.select(reads, [], [], 1.0)

            if ready[0]:
                try:
                    raw = os.read(proc.stdout.fileno(), 4096)
                except OSError:
                    raw = b""

                if raw:
                    text = raw.decode("utf-8", errors="replace")
                    sys.stdout.write(text)
                    sys.stdout.flush()

                    # Parse Iteration and Expectation
                    # Format: "[===---] Iter 1250/10000 ... Exp: 0.5364 ..."
                    # We might get multiple chunks. Scan all.

                    # Regex for Iteration
                    iter_matches = re.findall(r"Iter (\d+)/(\d+)", text)
                    if iter_matches:
                        last_match = iter_matches[-1]
                        current_iter = int(last_match[0])

                    # Regex for Expectation
                    # Exp: 0.5364
                    exp_matches = re.findall(r"Exp:\s*(-?[\d.]+)", text)
                    if exp_matches:
                        for val_str in exp_matches:
                            try:
                                val = float(val_str)
                                if val < current_best:
                                    current_best = val
                                    last_imp_iter = current_iter
                                    # Update global immediately for logging?
                                    # No, keep it local to this function logic.
                            except ValueError:
                                pass

            # Dynamic Reversion: If LONG run improves global best, treat as SHORT (enable stagnation check)
            if run_type == "LONG" and current_best < global_best:
                print(
                    f"[Watchdog] Improvement in FINAL RUN detected ({global_best:.4f} -> {current_best:.4f})!"
                )
                print("[Watchdog] Reverting to Stagnation Check capability.")
                run_type = "SHORT"
                global_best = current_best
                last_imp_iter = current_iter
                # We continue the loop, but now 'run_type' is SHORT, so the check below applies.

            # Stagnation Check (Only for SHORT runs)
            if run_type == "SHORT":
                stagnation_duration = current_iter - last_imp_iter
                if stagnation_duration > STAGNATION_LIMIT:
                    print(
                        f"\n[Watchdog] STAGNATION DETECTED: No improvement for {stagnation_duration} iters."
                    )
                    print(
                        f"[Watchdog] Current Best: {current_best:.4f} (Last Imp at {last_imp_iter})"
                    )
                    print("[Watchdog] Stopping process to restart...")

                    # Graceful Stop (SIGINT) to allow saving
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)

                    # Wait for exit
                    try:
                        proc.wait(timeout=30)  # Give 30s to save
                    except subprocess.TimeoutExpired:
                        print("[Watchdog] Save timed out. Forcing kill.")
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

                    return current_best, False  # Stopped early

    except KeyboardInterrupt:
        print("\n[Watchdog] Interrupted by User. Stopping.")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        sys.exit(0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python watchdog_restart.py [run_mome.py arguments...]")
        sys.exit(1)

    mome_args = sys.argv[1:]

    # Check for Global Seed Scan flag and strip it (we control when to pass it)
    use_global_scan = False
    if "--global-seed-scan" in mome_args:
        use_global_scan = True
        mome_args.remove("--global-seed-scan")

    global_best_exp = float("inf")
    consecutive_no_gains = 0

    while True:
        # Determine Run Type
        run_type = (
            "LONG" if consecutive_no_gains >= CONSECUTIVE_NO_GAINS_LIMIT else "SHORT"
        )

        print("\n" + "=" * 60)
        print(f"[Watchdog] Starting Cycle (Type: {run_type})")
        print(
            f"[Watchdog] Global Best: {global_best_exp if global_best_exp != float('inf') else 'None'}"
        )
        print(f"[Watchdog] Consecutive No Gains: {consecutive_no_gains}")
        print("=" * 60 + "\n")

        current_args = list(mome_args)
        if use_global_scan and global_best_exp == float("inf"):
            # This is effectively the first run
            current_args.append("--global-seed-scan")

        run_best, finished_naturally = run_optimization(
            current_args, run_type=run_type, global_best=global_best_exp
        )

        # Analyze Result
        print(f"\n[Watchdog] Run Finished. Best Local: {run_best:.4f}")

        improved = False
        if run_best < global_best_exp:
            diff = global_best_exp - run_best
            # Check for meaningful improvement? Assuming strictly < is enough as float
            # Or use a tiny epsilon? 1e-6
            if global_best_exp == float("inf") or diff > 1e-6:
                print(
                    f"[Watchdog] IMPROVEMENT DETECTED! ({global_best_exp:.4f} -> {run_best:.4f})"
                )
                global_best_exp = run_best
                improved = True
                consecutive_no_gains = 0
            else:
                print(
                    f"[Watchdog] Marginal/No improvement vs Global Best ({global_best_exp:.4f})."
                )
        else:
            print(f"[Watchdog] No improvement vs Global Best ({global_best_exp:.4f}).")

        if not improved:
            consecutive_no_gains += 1

        # Decision Logic
        if run_type == "LONG":
            if improved:
                print(
                    "[Watchdog] Long Run triggered improvement! Reverting to Restart Logic."
                )
                # Loop continues, type will be SHORT next time because consecutive_no_gains reset to 0
            else:
                print(
                    "[Watchdog] Long Run finished without improvement. Optimization Complete."
                )
                break
        else:
            # SHORT Run
            if improved:
                print("[Watchdog] Improvement found. Restarting cycle immediately.")
            else:
                print(
                    f"[Watchdog] Stagnation confirm ({consecutive_no_gains}/{CONSECUTIVE_NO_GAINS_LIMIT}). Restarting."
                )

        # Wait a moment before restart to ensure filesystem sync?
        time.sleep(2)


if __name__ == "__main__":
    main()
