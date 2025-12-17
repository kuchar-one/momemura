import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from src.simulation.jax.herald import jax_pure_state_amplitude, vacuum_covariance


def verify():
    print("Verifying Herald Recurrence Optimization...")

    # Setup N=3 case (1 Signal + 2 Control)
    cutoff = 25
    mu = jnp.zeros(6)  # 3 modes * 2 (x,p)
    cov = vacuum_covariance(3)
    pnr = (1, 1)  # 2 controls

    # JIT compile
    print("Compiling...")
    start = time.time()
    jit_fn = jax.jit(partial(jax_pure_state_amplitude, pnr_outcome=pnr, cutoff=cutoff))

    # Warmup
    # We pass pnr_outcome as static?
    # jax_pure_state_amplitude has static_argnames=("cutoff", "pnr_outcome")
    # So we call it directly.

    # Note: pure_state_amplitude is already decorated with @partial(jax.jit, static_argnames=...)
    # But let's call it.

    # We need to make sure we trigger _recurrence_3d.
    # N = 3 (mu size 6).

    res, prob = jax_pure_state_amplitude(mu, cov, pnr, cutoff)
    res.block_until_ready()
    print(f"Compilation + 1st run: {time.time() - start:.4f}s")

    # Benchmark
    print("Benchmarking 100 iterations...")
    start = time.time()
    for _ in range(100):
        res, prob = jax_pure_state_amplitude(mu, cov, pnr, cutoff)
        res.block_until_ready()
    avg = (time.time() - start) / 100
    print(f"Avg time per call: {avg:.6f}s")

    print(f"Result sum: {jnp.sum(jnp.abs(res))}")
    print(f"Prob: {prob}")

    # Check for NaNs
    if jnp.isnan(res).any():
        print("FAIL: NaNs detected!")
    else:
        print("SUCCESS: No NaNs.")


if __name__ == "__main__":
    verify()
