#!/usr/bin/env python
"""
Frontend runner script for Momemura visualization.

Usage:
    python run_frontend.py          # Uses default JAX backend (GPU if available)
    python run_frontend.py --cpu    # Forces CPU backend
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Momemura frontend")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force JAX to use CPU backend (avoids GPU/CUDA issues)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )
    args = parser.parse_args()

    # Set environment before launching Streamlit
    env = os.environ.copy()
    if args.cpu:
        env["JAX_PLATFORMS"] = "cpu"
        print("Running frontend with CPU backend")
    else:
        print("Running frontend with default JAX backend")

    # Get path to frontend app
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "frontend", "app.py")

    # Launch Streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(args.port),
    ]

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nFrontend stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Frontend exited with error: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
