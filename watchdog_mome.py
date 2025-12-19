import subprocess
import time
import sys
import re
import os
import signal
import fcntl
import select

# Configuration
TIMEOUT_SECONDS = (
    600  # 10 minutes (user's stuck time) -> let's say 10 mins without output is a hang
)
POLL_INTERVAL = 1


def set_non_blocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def main():
    if len(sys.argv) < 2:
        print("Usage: python watchdog_mome.py [run_mome.py arguments...]")
        sys.exit(1)

    mome_args = sys.argv[1:]
    cmd = [sys.executable, "-u", "run_mome.py"] + mome_args

    current_output_dir = None

    # Enable resume if output dir known and not passed explicitly?
    # Loop for restarts
    restart_count = 0

    while True:
        print(f"\n[Watchdog] Launching process (Attempt {restart_count + 1})...")
        print(f"[Watchdog] Command: {' '.join(cmd)}\n")

        # Start Process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,  # No input
            preexec_fn=os.setsid,  # New session group
        )

        # Set non-blocking stdout
        set_non_blocking(proc.stdout.fileno())

        last_output_time = time.time()

        try:
            while True:
                # Check status
                ret = proc.poll()
                if ret is not None:
                    print(f"\n[Watchdog] Process finished with code {ret}.")
                    if ret == 0:
                        sys.exit(0)  # Success
                    else:
                        print(
                            "[Watchdog] Process failed. Not restarting immediately (loops danger). Exiting."
                        )
                        sys.exit(ret)

                # IO Check
                reads = [proc.stdout.fileno()]
                ret = select.select(reads, [], [], 1.0)  # 1 sec timeout

                if reads[0] in ret[0]:
                    try:
                        # Read chunk
                        # raw = proc.stdout.read1(1024) # read1 for buffered?
                        # Use os.read for raw fd access
                        raw = os.read(proc.stdout.fileno(), 4096)
                    except OSError:
                        raw = b""

                    if raw:
                        text = raw.decode("utf-8", errors="replace")
                        sys.stdout.write(text)
                        sys.stdout.flush()

                        last_output_time = time.time()

                        # Parsing for resuming info
                        if "Created output directory:" in text:
                            match = re.search(r"Created output directory: (.+)", text)
                            if match:
                                current_output_dir = match.group(1).strip()
                                print(
                                    f"\n[Watchdog] Captured Output Dir: {current_output_dir}"
                                )

                # Check Timeout
                elapsed_since_output = time.time() - last_output_time
                if elapsed_since_output > TIMEOUT_SECONDS:
                    print(
                        f"\n[Watchdog] NO OUTPUT for {TIMEOUT_SECONDS}s. DETECTED HANG."
                    )
                    print("[Watchdog] Killing process group...")
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    time.sleep(2)
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

                    print("[Watchdog] Process killed. Initiating Restart...")
                    restart_count += 1

                    # Prepare Resume Args
                    # If we captured output dir, inject --resume
                    if current_output_dir:
                        # Check if --resume already in args
                        if "--resume" not in cmd:
                            print(
                                f"[Watchdog] Adding --resume {current_output_dir} to args."
                            )
                            cmd.extend(["--resume", current_output_dir])
                        else:
                            # Update existing? Assuming we append, argparse takes last?
                            # Usually yes.
                            pass

                    break  # Break inner loop to restart

                # Check total time limit? (Optional)

        except KeyboardInterrupt:
            print("\n[Watchdog] Interrupted by User. Stopping.")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            sys.exit(0)


if __name__ == "__main__":
    main()
