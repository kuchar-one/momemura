import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402
from jax.scipy.linalg import expm  # noqa: E402

# ... (existing imports)


# Current implementation
@partial(jax.jit, static_argnames=("cutoff",))
def jax_u_bs_expm(theta: float, phi: float, cutoff: int) -> jnp.ndarray:
    n = jnp.arange(1, cutoff)
    sqrt_n = jnp.sqrt(n)
    a_op = jnp.diag(sqrt_n, 1)
    Identity = jnp.eye(cutoff)
    a_big = jnp.kron(a_op, Identity)
    b_big = jnp.kron(Identity, a_op)
    bdag_big = b_big.conj().T
    term = jnp.exp(-1j * phi) * a_big @ bdag_big
    G = theta * (term - term.conj().T)
    U = expm(G)
    return U


# Optimized implementation using Block Diagonal Expm
@partial(jax.jit, static_argnames=("cutoff",))
def jax_u_bs_block_diag(theta: float, phi: float, cutoff: int) -> jnp.ndarray:
    """
    Constructs BS unitary by computing expm of blocks of G.
    G is block diagonal in total photon number N = n1 + n2.
    """
    import numpy as np

    # Number of blocks
    num_blocks = 2 * cutoff - 1

    # Max block size is cutoff.
    # We construct a batch of matrices (num_blocks, cutoff, cutoff).
    # Initialize with zeros.
    blocks = jnp.zeros((num_blocks, cutoff, cutoff), dtype=jnp.complex128)

    # Vectorized construction
    # We need to compute diagonals for each block.
    # N ranges from 0 to num_blocks-1.
    # Use JAX arrays for the computation of matrix elements (which depend on theta/phi - dynamic)
    Ns_jax = jnp.arange(num_blocks)

    # For each N, valid n1 ranges from n_min to n_max.
    # n_min = max(0, N - cutoff + 1)
    # n_max = min(N, cutoff - 1)
    n_min_jax = jnp.maximum(0, Ns_jax - cutoff + 1)
    n_max_jax = jnp.minimum(Ns_jax, cutoff - 1)

    # We want to fill the superdiagonal (k-1, k) and subdiagonal (k, k-1).
    # k goes from 1 to dim-1.
    # dim = n_max - n_min + 1.
    # Let's map k to n1: n1 = n_min + k.
    # The coupling is between n1 and n1-1.
    # So we iterate n1 from n_min+1 to n_max.

    # We can use a grid of (N, k) where k is index within block.
    # k goes from 0 to cutoff-1.
    # n1 = n_min + k.
    # Valid if n1 <= n_max.
    # Actually, coupling is at k (between k and k-1).
    # So valid for k=1 to cutoff-1, and n_min+k <= n_max.

    ks = jnp.arange(cutoff)

    # Broadcast
    Ns_grid = Ns_jax[:, None]  # (num_blocks, 1)
    ks_grid = ks[None, :]  # (1, cutoff)
    n_min_grid = n_min_jax[:, None]
    n_max_grid = n_max_jax[:, None]

    # Current n1 corresponding to index k
    n1 = n_min_grid + ks_grid

    # Validity mask for the coupling element (k-1, k)
    # We need index k and k-1 to be valid.
    # k starts at 1.
    # n1 must be <= n_max_grid.
    # And k >= 1.
    valid_mask = (ks_grid >= 1) & (n1 <= n_max_grid)

    # The coupling term corresponds to a b^dag between |n1-1, N-(n1-1)> and |n1, N-n1>.
    # <n1-1, ... | a b^dag | n1, ...> = sqrt(n1 * (N - n1 + 1))
    # This is element (k-1, k).

    val = jnp.sqrt(n1 * (Ns_grid - n1 + 1))

    # G_k-1,k = theta * e^{-i phi} * val
    # G_k,k-1 = -theta * e^{i phi} * val

    term = theta * val * valid_mask
    diag_super = term * jnp.exp(-1j * phi)
    diag_sub = -term * jnp.exp(1j * phi)

    # Fill blocks
    # We use .at with index arrays.
    # We want to set blocks[N, k-1, k] and blocks[N, k, k-1].

    # Indices for scatter
    # We need to flatten or use vmap scatter?
    # Let's use vmap over N? No, we have the grid.

    # blocks is (num_blocks, cutoff, cutoff).
    # We update at [:, k-1, k]

    # We can loop over k from 1 to cutoff-1 (unrolled loop of size 10 is fine).
    # This avoids advanced indexing issues.

    for k in range(1, cutoff):
        # Slice for this k across all blocks
        # valid_mask[:, k]
        # diag_super[:, k]

        # We update blocks[:, k-1, k]
        blocks = blocks.at[:, k - 1, k].set(diag_super[:, k])
        blocks = blocks.at[:, k, k - 1].set(diag_sub[:, k])

    # Compute expm of all blocks
    # jax.scipy.linalg.expm supports batching?
    # Documentation says "Compute the matrix exponential of an array."
    # Usually supports batching.
    U_blocks = jax.scipy.linalg.expm(blocks)

    # Now scatter U_blocks into the full U matrix.
    U = jnp.zeros((cutoff**2, cutoff**2), dtype=jnp.complex128)

    # We need to map (N, i, j) -> (row, col) in U.
    # i, j are indices within block N.
    # n1_row = n_min[N] + i
    # n2_row = N - n1_row
    # row = n1_row * cutoff + n2_row

    # n1_col = n_min[N] + j
    # n2_col = N - n1_col
    # col = n1_col * cutoff + n2_col

    # We need static indices for slicing.
    # Compute n_min, n_max using numpy (static).
    Ns_np = np.arange(num_blocks)
    n_min_np = np.maximum(0, Ns_np - cutoff + 1)
    n_max_np = np.minimum(Ns_np, cutoff - 1)

    # Collect indices and values for single scatter
    all_ii = []
    all_jj = []
    all_vals = []

    for N in range(num_blocks):
        nm = int(n_min_np[N])
        nM = int(n_max_np[N])
        dim = nM - nm + 1

        # Indices in U
        n1s = jnp.arange(nm, nM + 1)
        n2s = N - n1s
        indices = n1s * cutoff + n2s

        # Extract valid subblock
        block = U_blocks[N, :dim, :dim]

        # Grid indices
        ii, jj = jnp.meshgrid(indices, indices, indexing="ij")

        all_ii.append(ii.flatten())
        all_jj.append(jj.flatten())
        all_vals.append(block.flatten())

    # Concatenate
    total_ii = jnp.concatenate(all_ii)
    total_jj = jnp.concatenate(all_jj)
    total_vals = jnp.concatenate(all_vals)

    # Single scatter
    U = U.at[total_ii, total_jj].set(total_vals)

    return U


# Optimized implementation using Decomposition + Precomputed Generators
# U = D(-phi) @ exp(-2i theta Jy) @ D(phi)
# Jy blocks are constant. We can precompute them.
# But for now, let's just compute them on the fly to test the decomposition speedup (real vs complex).


@partial(jax.jit, static_argnames=("cutoff",))
def jax_u_bs_decomposed(theta: float, phi: float, cutoff: int) -> jnp.ndarray:
    import numpy as np

    num_blocks = 2 * cutoff - 1

    # Precompute Jy blocks (real, antisymmetric)
    # In a real implementation, these would be cached/static.
    # Here we construct them.

    # We need to construct the blocks of Jy.
    # Jy = (a^dag b - a b^dag) / 2i
    # -2i theta Jy = -theta (a^dag b - a b^dag)
    # This is real.
    # Element (k, k-1) is -theta * sqrt(...)
    # Element (k-1, k) is theta * sqrt(...)

    # We can reuse the block construction logic but for real matrix.

    blocks = jnp.zeros((num_blocks, cutoff, cutoff), dtype=jnp.float64)  # Real!

    Ns_jax = jnp.arange(num_blocks)
    n_min_jax = jnp.maximum(0, Ns_jax - cutoff + 1)
    n_max_jax = jnp.minimum(Ns_jax, cutoff - 1)

    ks = jnp.arange(cutoff)

    Ns_grid = Ns_jax[:, None]
    ks_grid = ks[None, :]
    n_min_grid = n_min_jax[:, None]
    n_max_grid = n_max_jax[:, None]

    n1 = n_min_grid + ks_grid
    valid_mask = (ks_grid >= 1) & (n1 <= n_max_grid)

    val = jnp.sqrt(n1 * (Ns_grid - n1 + 1))

    # Generator for exp(-2i theta Jy) is -theta * (a^dag b - a b^dag)
    # = theta * a b^dag - theta * a^dag b
    # (k-1, k) is a b^dag term -> +theta
    # (k, k-1) is a^dag b term -> -theta

    term = theta * val * valid_mask

    # Fill blocks
    for k in range(1, cutoff):
        blocks = blocks.at[:, k - 1, k].set(term[:, k])
        blocks = blocks.at[:, k, k - 1].set(-term[:, k])

    # Compute expm (REAL)
    Ry_blocks = jax.scipy.linalg.expm(blocks)

    # Now apply phases D(phi)
    # D = exp(-i phi Jz)
    # Jz = (n1 - n2)/2
    # Diagonal elements.

    # We can apply phases to the blocks before scattering?
    # U_block = D_block^dag @ Ry_block @ D_block
    # (U)_{ij} = exp(i phi m_i) (Ry)_{ij} exp(-i phi m_j)
    # m_i = (n1_i - n2_i)/2

    # Let's compute m values for the grid.
    # n1 is n_min + k
    # n2 is N - n1
    # m = (n1 - n2)/2 = n1 - N/2

    m_vals = n1 - Ns_grid / 2.0

    # Phases
    # We want e^{-i phi Jz} Ry e^{i phi Jz}
    # Left mult by exp(-i phi m_i)
    # Right mult by exp(i phi m_j)

    phases = jnp.exp(-1j * phi * m_vals)  # (num_blocks, cutoff)

    # Broadcast multiply
    # Ry_blocks is (num_blocks, cutoff, cutoff)
    # phases[:, :, None] * Ry_blocks * phases.conj()[:, None, :]

    U_blocks_complex = phases[:, :, None] * Ry_blocks * phases.conj()[:, None, :]

    # Scatter
    U = jnp.zeros((cutoff**2, cutoff**2), dtype=jnp.complex128)

    Ns_np = np.arange(num_blocks)
    n_min_np = np.maximum(0, Ns_np - cutoff + 1)
    n_max_np = np.minimum(Ns_np, cutoff - 1)

    all_ii = []
    all_jj = []
    all_vals = []

    for N in range(num_blocks):
        nm = int(n_min_np[N])
        nM = int(n_max_np[N])
        dim = nM - nm + 1

        n1s = jnp.arange(nm, nM + 1)
        n2s = N - n1s
        indices = n1s * cutoff + n2s

        block = U_blocks_complex[N, :dim, :dim]

        ii, jj = jnp.meshgrid(indices, indices, indexing="ij")

        all_ii.append(ii.flatten())
        all_jj.append(jj.flatten())
        all_vals.append(block.flatten())

    total_ii = jnp.concatenate(all_ii)
    total_jj = jnp.concatenate(all_jj)
    total_vals = jnp.concatenate(all_vals)

    U = U.at[total_ii, total_jj].set(total_vals)

    return U


def _compute_d_block(N, n1s, theta, log_facts):
    # Computes Wigner d-matrix elements for j=N/2
    # Rows: n1 (bra), Cols: m1 (ket)
    # n1s is vector of n1 values.
    # We need all pairs (n1, m1) from n1s.

    beta = -2 * theta

    # Grid of n1, m1
    n1, m1 = jnp.meshgrid(n1s, n1s, indexing="ij")

    n2 = N - n1
    m2 = N - m1

    # m, m_prime in Wigner formula
    # m (ket) -> associated with m1
    # m' (bra) -> associated with n1
    # m_val = (m1 - m2)/2 = m1 - j
    # mp_val = (n1 - n2)/2 = n1 - j

    # Formula:
    # d^j_{m', m} = sum_k (-1)^k ...

    # Range of k:
    # k >= 0
    # k >= m - m' = (m1 - j) - (n1 - j) = m1 - n1
    # k <= j + m = j + m1 - j = m1
    # k <= j - m' = j - (n1 - j) = 2j - n1 = N - n1 = n2

    # So max(0, m1 - n1) <= k <= min(m1, n2)

    # We can iterate k from 0 to N. Terms outside valid range are 0 (handled by log_fact logic or mask).
    # Since N is small, we can sum over k.

    # Vectorized sum over k?
    # We can add a dimension for k.

    # Let's define a safe log_fact that returns -inf for negative args?
    # Or just use bounds.

    # k range is dynamic per element.
    # But we can sum k from 0 to N and mask.

    ks = jnp.arange(N + 1)  # (K,)

    # Broadcast shapes: (Dim, Dim, K)
    n1 = n1[..., None]
    m1 = m1[..., None]
    n2 = n2[..., None]
    m2 = m2[..., None]
    # j is scalar

    # Terms
    # sqrt((j+m')!(j-m')!(j+m)!(j-m)!)
    # = sqrt(n1! n2! m1! m2!)

    # We use log factorial
    def lf(x):
        # x is array. map to log_facts.
        # Clip negative to 0 (will be masked anyway)
        return log_facts[jnp.clip(x, 0, None).astype(jnp.int32)]

    # Log numerator
    log_num = 0.5 * (lf(n1) + lf(n2) + lf(m1) + lf(m2))

    # Log denominator
    # (j+m'-k)! (j-m-k)! k! (k+m-m')!
    # j+m' = n1
    # j-m = n2
    # m-m' = m1 - n1
    # Denom: (n1-k)! (n2-k)! k! (k+m1-n1)!

    # k is broadcasted

    # Valid mask for k
    # n1-k >= 0, n2-k >= 0, k >= 0, k+m1-n1 >= 0
    mask = (n1 >= ks) & (n2 >= ks) & (ks >= 0) & (ks + m1 - n1 >= 0)

    log_denom = lf(n1 - ks) + lf(n2 - ks) + lf(ks) + lf(ks + m1 - n1)

    # Phase (-1)^k
    # sign = (-1)**k
    # We handle sign separately.

    # Trig terms
    # (cos b/2)^(2j - 2k + m - m') = (cos)^(N - 2k + m1 - n1)
    # (sin b/2)^(2k + m' - m) = (sin)^(2k + n1 - m1)

    cb = jnp.cos(beta / 2)
    sb = jnp.sin(beta / 2)

    # Exponents
    exp_c = N - 2 * ks + m1 - n1
    exp_s = 2 * ks + n1 - m1

    # Combine log terms
    log_term = log_num - log_denom

    # Term magnitude
    mag = jnp.exp(log_term) * (cb**exp_c) * (sb**exp_s)

    # Apply sign and mask
    term = mag * ((-1.0) ** ks) * mask

    # Sum over k
    d_mat = jnp.sum(term, axis=-1)

    return d_mat


# Explicit implementation (Placeholder for now, using expm to verify baseline)
# We will replace this with the optimized version.
@partial(jax.jit, static_argnames=("cutoff",))
def jax_u_bs_optimized(theta: float, phi: float, cutoff: int) -> jnp.ndarray:
    # TODO: Implement explicit block-diagonal construction
    return jax_u_bs_expm(theta, phi, cutoff)


def benchmark():
    # Debug case
    cutoff = 2
    theta = 0.5
    phi = 0.0  # Try real case first

    print(f"Debugging BS Unitary (Cutoff={cutoff}, phi={phi})...")

    u1 = jax_u_bs_expm(theta, phi, cutoff)
    u2 = jax_u_bs_decomposed(theta, phi, cutoff)

    print("Expm U:")
    print(u1)
    print("Decomposed U:")
    print(u2)

    diff = jnp.linalg.norm(u1 - u2)
    print(f"Difference norm: {diff:.2e}")

    if diff > 1e-5:
        print("Mismatch!")
        return

    # Performance benchmark
    cutoff = 10
    phi = 0.2
    print(f"\nBenchmarking Performance (Cutoff={cutoff})...")

    # Warmup
    _ = jax_u_bs_expm(theta, phi, cutoff).block_until_ready()
    _ = jax_u_bs_decomposed(theta, phi, cutoff).block_until_ready()

    # Time Expm
    start = time.time()
    for _ in range(100):
        _ = jax_u_bs_expm(theta, phi, cutoff).block_until_ready()
    end = time.time()
    print(f"Expm: {(end - start) * 10000:.2f} ms per 1000 calls (extrapolated)")

    # Time Decomposed
    start = time.time()
    for _ in range(100):
        _ = jax_u_bs_decomposed(theta, phi, cutoff).block_until_ready()
    end = time.time()
    print(f"Decomposed: {(end - start) * 10000:.2f} ms per 1000 calls (extrapolated)")

    # Performance benchmark C=25
    cutoff = 25
    print(f"\nBenchmarking Performance (Cutoff={cutoff})...")

    # Warmup
    _ = jax_u_bs_expm(theta, phi, cutoff).block_until_ready()
    _ = jax_u_bs_decomposed(theta, phi, cutoff).block_until_ready()

    # Time Expm
    start = time.time()
    for _ in range(100):
        _ = jax_u_bs_expm(theta, phi, cutoff).block_until_ready()
    end = time.time()
    print(f"Expm: {(end - start) * 10000:.2f} ms per 1000 calls (extrapolated)")

    # Time Decomposed
    start = time.time()
    for _ in range(100):
        _ = jax_u_bs_decomposed(theta, phi, cutoff).block_until_ready()
    end = time.time()
    print(f"Decomposed: {(end - start) * 10000:.2f} ms per 1000 calls (extrapolated)")


if __name__ == "__main__":
    benchmark()
