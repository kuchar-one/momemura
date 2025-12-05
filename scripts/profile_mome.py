import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cProfile
import pstats
import run_mome


def profile_run():
    # Set arguments for run_mome
    sys.argv = [
        "run_mome.py",
        "--mode",
        "qdax",
        "--pop",
        "50",
        "--iters",
        "20",
        "--cutoff",
        "25",
        "--backend",
        "jax",
        "--no-plot",
        "--low-mem",
    ]

    print("Starting profiling run...")
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        run_mome.main()
    except SystemExit:
        pass

    profiler.disable()
    print("Profiling complete.")

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(50)
    stats.dump_stats("mome_profile.prof")


if __name__ == "__main__":
    profile_run()
