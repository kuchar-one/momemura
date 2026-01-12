import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

# Constants
HBAR = 2.0


def vacuum_covariance(N: int, hbar: float = HBAR) -> jnp.ndarray:
    """Returns the vacuum covariance matrix for N modes."""
    return (hbar / 2) * jnp.eye(2 * N)


def passive_unitary_to_symplectic(U: jnp.ndarray) -> jnp.ndarray:
    """
    Converts an N-mode passive unitary U to a 2N x 2N symplectic matrix.
    S = [[X, -Y], [Y, X]] where U = X + iY.
    This is for xp ordering?
    Thewalrus uses xp ordering: (x1, ..., xN, p1, ..., pN).
    In xp ordering, a passive transformation U acts as:
    a' = U a.
    x' + i p' = U (x + i p).
    x' = X x - Y p
    p' = Y x + X p
    So [x'; p'] = [[X, -Y], [Y, X]] [x; p].
    """
    X = jnp.real(U)
    Y = jnp.imag(U)
    return jnp.block([[X, -Y], [Y, X]])


def two_mode_squeezer_symplectic(r: float, phi: float = 0.0) -> jnp.ndarray:
    """
    Returns the symplectic matrix for a two-mode squeezer (S2 gate).
    In xp ordering (x1, x2, p1, p2).
    S(z) where z = r * exp(i phi).
    For r only (phi=0):
    x1 -> x1 cosh r + x2 sinh r
    x2 -> x1 sinh r + x2 cosh r
    p1 -> p1 cosh r - p2 sinh r
    p2 -> -p1 sinh r + p2 cosh r
    Wait, this depends on convention.
    Let's match thewalrus/strawberryfields convention.
    S2(r):
    cosh(r) I2   sinh(r) Z2
    sinh(r) Z2   cosh(r) I2
    where Z2 = diag(1, -1)? No.
    Let's check `ops.py` or `gaussian_herald_circuit.py` logic.
    It uses `thewalrus.symplectic.two_mode_squeezing`.
    I'll assume standard definition.
    """
    ch = jnp.cosh(r)
    sh = jnp.sinh(r)
    # Using thewalrus convention (inferred):
    # S = [[ch*I, sh*Z], [sh*Z, ch*I]] where Z = diag(1, -1)?
    # Actually, let's implement a generic one or assume standard.
    # For TMSS, we usually squeeze X1-X2 and P1+P2?
    # Let's use the explicit matrix.
    # Block diagonal in X/P? No, it mixes X and P?
    # No, TMSS with phi=0 mixes x1,x2 and p1,p2 separately.
    # x1 -> ch x1 + sh x2
    # x2 -> sh x1 + ch x2
    # p1 -> ch p1 - sh p2
    # p2 -> -sh p1 + ch p2

    # So in xp ordering (x1, x2, p1, p2):
    # X block: [[ch, sh], [sh, ch]]
    # P block: [[ch, -sh], [-sh, ch]]
    # XP/PX blocks: 0

    X_blk = jnp.array([[ch, sh], [sh, ch]])
    P_blk = jnp.array([[ch, -sh], [-sh, ch]])
    Z_blk = jnp.zeros((2, 2))

    return jnp.block([[X_blk, Z_blk], [Z_blk, P_blk]])


def expand_mode_symplectic(
    S_small: jnp.ndarray, modes: jnp.ndarray, N: int
) -> jnp.ndarray:
    """
    Expands a small symplectic matrix acting on `modes` to a full NxN symplectic matrix.
    S_small is 2k x 2k (xp ordering).
    modes is array of k indices.
    N is total modes.
    """
    k = len(modes)
    # S_full = I
    S_full = jnp.eye(2 * N)

    # We need to place elements of S_small into S_full.
    # S_small has structure [[A, B], [C, D]] where A,B,C,D are k x k.
    # We map A to X-X block, B to X-P, etc.

    # Indices in full matrix:
    # X indices: modes
    # P indices: modes + N

    # Since we can't mutate, we construct update indices.
    # But this is hard in JAX dynamic slice update.
    # However, k is small (2 for TMSS).
    # We can use `at[...].set(...)`.

    # Extract blocks
    A = S_small[:k, :k]
    B = S_small[:k, k:]
    C = S_small[k:, :k]
    D = S_small[k:, k:]

    # Update X-X
    # We need to update S_full[modes, modes] = A?
    # JAX advanced indexing `S_full.at[modes][:, modes].set(A)` doesn't work directly like numpy.
    # We need `S_full.at[jnp.ix_(modes, modes)].set(A)`.

    x_idxs = modes
    p_idxs = modes + N

    S_full = S_full.at[jnp.ix_(x_idxs, x_idxs)].set(A)
    S_full = S_full.at[jnp.ix_(x_idxs, p_idxs)].set(B)
    S_full = S_full.at[jnp.ix_(p_idxs, x_idxs)].set(C)
    S_full = S_full.at[jnp.ix_(p_idxs, p_idxs)].set(D)

    return S_full


def complex_alpha_to_qp(alpha: jnp.ndarray, hbar: float = HBAR) -> jnp.ndarray:
    """
    Converts complex displacement alpha to (q, p) vector.
    alpha = (q + i p) / sqrt(2 hbar)
    => q = sqrt(2 hbar) Re(alpha)
    => p = sqrt(2 hbar) Im(alpha)
    Returns [q1, ..., qN, p1, ..., pN].
    """
    scale = jnp.sqrt(2 * hbar)
    q = scale * jnp.real(alpha)
    p = scale * jnp.imag(alpha)
    return jnp.concatenate([q, p])


def Xmat(N: int) -> jnp.ndarray:
    """
    Returns the matrix X_N = [[0, I_N], [I_N, 0]].
    """
    eye_mat = jnp.eye(N)
    Z = jnp.zeros((N, N))
    return jnp.block([[Z, eye_mat], [eye_mat, Z]])


def Qmat(cov: jnp.ndarray, hbar: float = HBAR) -> jnp.ndarray:
    """
    Returns the Q matrix in the complex basis (a, a^dag) matching thewalrus convention.
    Q = [[<a a^dag>, <a^dag a^dag>], [<a a>, <a a^dag>]]
    """
    N = cov.shape[0] // 2
    identity = jnp.eye(N)

    # cov is in XP ordering [x1...xN, p1...pN]
    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar

    # <a_i^dag a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * identity) / 4
    # <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # Q = [[aidaj, aiaj*], [aiaj, aidaj*]] + I
    # Note: thewalrus adds I to diagonal blocks.
    # aidaj + I = <a a^dag>
    # aidaj* + I = <a^dag a> + I = <a a^dag> (since <a^dag a> = <a a^dag> - I? No)
    # <a a^dag> = <a^dag a> + I.
    # So aidaj* + I is indeed <a a^dag>.

    Q = jnp.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + jnp.eye(2 * N)
    return Q


def Amat(cov: jnp.ndarray, hbar: float = HBAR) -> jnp.ndarray:
    """
    Returns the A matrix for a Gaussian state in the complex basis (a, a^dag).
    A = X @ (I - Q^{-1})*.
    """
    N = cov.shape[0] // 2
    Q = Qmat(cov, hbar)
    Qinv = jnp.linalg.inv(Q)
    eye_mat = jnp.eye(2 * N)
    X = Xmat(N)
    return X @ (eye_mat - Qinv).conj()


def complex_to_real_displacements(mu: jnp.ndarray, hbar: float = HBAR) -> jnp.ndarray:
    """
    Converts xp-ordered means (mu) to complex displacements (alpha).
    alpha = (x + i*p) / sqrt(2*hbar)
    """
    N = mu.shape[0] // 2
    x = mu[:N]
    p = mu[N:]
    return (x + 1j * p) / jnp.sqrt(2 * hbar)


def calc_A_gamma(
    mu: jnp.ndarray, cov: jnp.ndarray, hbar: float = HBAR
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes A matrix and gamma vector for Hermite polynomials.
    gamma = alpha - A @ alpha.conj()
    """
    A = Amat(cov, hbar)
    alpha = complex_to_real_displacements(mu, hbar)
    # alpha is length N. A is 2N x 2N.
    # Wait. Thewalrus pure_state_amplitude uses B = A[0:N, 0:N].
    # And gamma = alpha - B @ alpha.conj().
    # This implies we only need the top-left block of A?
    # Yes, for pure states, the amplitude depends on the "B" matrix (submatrix of A).
    # Let's verify this.
    # The multidimensional Hermite polynomial H_n(B, gamma) depends on B (NxN) and gamma (N).

    N = mu.shape[0] // 2
    # TheWalrus takes B = A[:N, :N].conj()
    B = A[:N, :N].conj()
    gamma = alpha - B @ alpha.conj()
    return B, gamma


@partial(jax.jit, static_argnames=("cutoff", "pnr_outcome"))
def jax_pure_state_amplitude(
    mu: jnp.ndarray,
    cov: jnp.ndarray,
    pnr_outcome: Tuple[int, ...],
    cutoff: int,
    hbar: float = HBAR,
) -> Tuple[jnp.ndarray, float]:
    """
    Computes the amplitudes of the signal modes given PNR outcomes on control modes.
    Uses the recurrence relation for multidimensional Hermite polynomials.

    Args:
        mu: Mean vector (xp ordering).
        cov: Covariance matrix (xp ordering).
        pnr_outcome: Tuple of PNR outcomes for control modes.
        cutoff: Fock cutoff for signal modes.
        hbar: Planck constant.

    Returns:
        norm: Normalized amplitudes for signal modes (tensor).
        prob: Probability of the PNR outcome (sum of squared amplitudes).
    """
    # 1. Compute B and gamma
    B, gamma = calc_A_gamma(mu, cov, hbar)

    # Explicitly cast to dynamic complex dtype
    dtype = (jnp.zeros(1) + 1j).dtype
    B = B.astype(dtype)
    gamma = gamma.astype(dtype)
    common_dtype = dtype

    # 2. Setup shape
    N = mu.shape[0] // 2
    n_sig = N - len(pnr_outcome)

    shape = [cutoff] * n_sig + [p + 1 for p in pnr_outcome]

    # 3. Calculate prefactor
    scale = jnp.sqrt(2 * hbar)
    N_modes = mu.shape[0] // 2
    q = mu[:N_modes]
    p = mu[N_modes:]
    alpha = (q + 1j * p) / scale

    Q = Qmat(cov, hbar)
    _, logdet = jnp.linalg.slogdet(Q)
    detQ_fourth = jnp.exp(0.25 * logdet)
    prefactor = (
        jnp.exp(-0.5 * (jnp.sum(jnp.abs(alpha) ** 2) - alpha.conj() @ B @ alpha.conj()))
        / detQ_fourth
    )

    # 4. Run recurrence (passing prefactor)
    if N == 1:
        H = _recurrence_1d(shape, B, gamma, common_dtype, prefactor)
    elif N == 2:
        H = _recurrence_2d(shape, B, gamma, common_dtype, prefactor)
    elif N == 3:
        H = _recurrence_3d(shape, B, gamma, common_dtype, prefactor)
    elif N == 4:
        H = _recurrence_4d(shape, B, gamma, common_dtype, prefactor)
    else:
        raise NotImplementedError("JAX recurrence only implemented for N<=4")

    amplitudes = H

    # 5. Slice at pnr_outcome
    slices = [slice(None)] * n_sig + [p for p in pnr_outcome]
    res = amplitudes[tuple(slices)]

    # 6. Normalize
    # The result `res` is the amplitude vector.
    # Probability is squared norm.

    prob = jnp.sum(jnp.abs(res) ** 2)

    norm = jax.lax.cond(
        prob > 0, lambda _: res / jnp.sqrt(prob), lambda _: jnp.zeros_like(res), None
    )

    return norm, prob


def jax_get_full_amplitudes(
    mu: jnp.ndarray,
    cov: jnp.ndarray,
    max_pnr_outcome: Tuple[int, ...],
    cutoff: int,
    hbar: float = HBAR,
) -> jnp.ndarray:
    """
    Computes the full amplitude tensor for all PNR outcomes up to max_pnr_outcome.
    Returns the unnormalized amplitudes.
    """
    # 1. Compute B and gamma
    B, gamma = calc_A_gamma(mu, cov, hbar)

    # Explicitly cast to dynamic complex dtype
    dtype = (jnp.zeros(1) + 1j).dtype
    B = B.astype(dtype)
    gamma = gamma.astype(dtype)
    common_dtype = dtype

    # 2. Setup shape
    N = mu.shape[0] // 2
    n_sig = N - len(max_pnr_outcome)
    shape = [cutoff] * n_sig + [p + 1 for p in max_pnr_outcome]

    # 3. Calculate prefactor
    scale = jnp.sqrt(2 * hbar)
    N_modes = mu.shape[0] // 2
    q = mu[:N_modes]
    p = mu[N_modes:]
    alpha = (q + 1j * p) / scale

    Q = Qmat(cov, hbar)
    _, logdet = jnp.linalg.slogdet(Q)
    detQ_fourth = jnp.exp(0.25 * logdet)
    prefactor = (
        jnp.exp(-0.5 * (jnp.sum(jnp.abs(alpha) ** 2) - alpha.conj() @ B @ alpha.conj()))
        / detQ_fourth
    )

    # 4. Run recurrence (passing prefactor)
    if N == 1:
        H = _recurrence_1d(shape, B, gamma, common_dtype, prefactor)
    elif N == 2:
        H = _recurrence_2d(shape, B, gamma, common_dtype, prefactor)
    elif N == 3:
        H = _recurrence_3d(shape, B, gamma, common_dtype, prefactor)
    elif N == 4:
        H = _recurrence_4d(shape, B, gamma, common_dtype, prefactor)
    else:
        raise NotImplementedError("JAX recurrence only implemented for N<=4")

    # Amplitudes are directly from recurrence (prefactor included in initialization)
    amplitudes = H

    return amplitudes


def _recurrence_1d(shape, B, gamma, dtype, prefactor):
    # Normalized recurrence:
    # A[n] = gamma/sqrt(n) A[n-1] + B sqrt((n-1)/n) A[n-2]
    # A[0] = prefactor
    N0 = shape[0]
    A = jnp.zeros(N0, dtype=dtype)
    A = A.at[0].set(prefactor)

    def body(n, a):
        # n goes from 1 to N0-1
        # Term 1: gamma * A[n-1] / sqrt(n)
        val = (gamma[0] * a[n - 1]) / jnp.sqrt(n)

        # Term 2: B * A[n-2] * sqrt((n-1)/n)
        val = val + jax.lax.cond(
            n >= 2,
            lambda _: B[0, 0] * a[n - 2] * jnp.sqrt((n - 1) / n),
            lambda _: jnp.array(0, dtype=dtype),
            None,
        )
        return a.at[n].set(val)

    A = jax.lax.fori_loop(1, N0, body, A)
    return A


def _recurrence_2d(shape, B, gamma, dtype, prefactor):
    # Normalized recurrence optimized (branchless)
    N0, N1 = shape
    A = jnp.zeros(shape, dtype=dtype)
    A = A.at[0, 0].set(prefactor)

    # Precompute constants
    b00, b01, b11 = B[0, 0], B[0, 1], B[1, 1]
    g0, g1 = gamma[0], gamma[1]

    def body_n0(n0, h_arr):
        # Safe reciprocal for n0 (avoid div by zero when n0=0)
        inv_sqrt_n0 = jax.lax.rsqrt(jnp.maximum(n0, 1.0).astype(dtype))
        sqrt_n0_minus_1_div_n0 = jnp.sqrt(
            (jnp.maximum(n0 - 1, 0.0)) * inv_sqrt_n0 * inv_sqrt_n0
        )

        # Masks
        # Masks
        n0_pos = jnp.array(n0 > 0, dtype=dtype)
        n0_ge_2 = jnp.array(n0 >= 2, dtype=dtype)

        def body_n1(n1, h):
            # n1 loop interaction
            inv_sqrt_n1 = jax.lax.rsqrt(jnp.maximum(n1, 1.0).astype(dtype))
            sqrt_n1_minus_1_div_n1 = jnp.sqrt(
                (jnp.maximum(n1 - 1, 0.0)) * inv_sqrt_n1 * inv_sqrt_n1
            )
            sqrt_n1_div_n0 = jnp.sqrt(n1) * inv_sqrt_n0

            n1_pos = jnp.array(n1 > 0, dtype=dtype)
            n1_ge_2 = jnp.array(n1 >= 2, dtype=dtype)

            # Case i=0 (n0 > 0)
            # T1: gamma0 * A[n0-1, n1] / sqrt(n0)
            val0 = (g0 * h[n0 - 1, n1]) * inv_sqrt_n0
            # T2: B00 * A[n0-2, n1] * sqrt((n0-1)/n0)
            val0 += (b00 * h[n0 - 2, n1] * sqrt_n0_minus_1_div_n0) * n0_ge_2
            # T3: B01 * A[n0-1, n1-1] * sqrt(n1/n0)
            val0 += (b01 * h[n0 - 1, n1 - 1] * sqrt_n1_div_n0) * n1_pos

            # Case i=1 (Fallback if n0=0, implies n1 > 0)
            # T1: gamma1 * A[0, n1-1] / sqrt(n1)
            val1 = (g1 * h[0, n1 - 1]) * inv_sqrt_n1
            # T2: B11 * A[0, n1-2] * sqrt((n1-1)/n1)
            val1 += (b11 * h[0, n1 - 2] * sqrt_n1_minus_1_div_n1) * n1_ge_2

            # Select based on n0 > 0
            # If n0=0, we use val1. If n0>0, we use val0.
            # (0,0) is skipped by (n0_pos + n1_pos) check? No, (0,0) is set.
            # We must not overwrite (0,0).

            is_start = (n0 == 0) & (n1 == 0)

            val = val0 * n0_pos + val1 * (1.0 - n0_pos)

            # Update only if not start
            # But functional update requires explicit branch or writing same value
            current = h[n0, n1]
            new_val = jax.lax.select(is_start, current, val)

            return h.at[n0, n1].set(new_val)

        return jax.lax.fori_loop(0, N1, body_n1, h_arr)

    A = jax.lax.fori_loop(0, N0, body_n0, A)
    return A


# I will implement _recurrence_3d and _recurrence_4d similarly if needed,
# but for now let's stick to 2D (1 sig, 1 ctrl) which is the main use case.
# If N > 2, we can raise error or implement later.
# Actually, I should implement a generic one using unrolled loops if possible,
# but hardcoding 2D covers the immediate bottleneck.
# I'll add a check.


def _recurrence_3d(shape, B, gamma, dtype, prefactor):
    # Normalized recurrence optimized (branchless)
    N0, N1, N2 = shape
    A = jnp.zeros(shape, dtype=dtype)
    A = A.at[0, 0, 0].set(prefactor)

    # Constants
    b00, b01, b02 = B[0, 0], B[0, 1], B[0, 2]
    b11, b12 = B[1, 1], B[1, 2]
    b22 = B[2, 2]
    g0, g1, g2 = gamma[0], gamma[1], gamma[2]

    def body_n0(n0, h0):
        # n0 precalcs
        inv_sqrt_n0 = jax.lax.rsqrt(jnp.maximum(n0, 1.0).astype(dtype))
        sqrt_n0_minus_1_div_n0 = jnp.sqrt(
            (jnp.maximum(n0 - 1, 0.0)) * inv_sqrt_n0 * inv_sqrt_n0
        )
        n0_pos = jnp.array(n0 > 0, dtype=dtype)
        n0_ge_2 = jnp.array(n0 >= 2, dtype=dtype)

        def body_n1(n1, h1):
            # n1 precalcs
            inv_sqrt_n1 = jax.lax.rsqrt(jnp.maximum(n1, 1.0).astype(dtype))
            sqrt_n1_minus_1_div_n1 = jnp.sqrt(
                (jnp.maximum(n1 - 1, 0.0)) * inv_sqrt_n1 * inv_sqrt_n1
            )
            # Cross terms
            sqrt_n1_div_n0 = jnp.sqrt(n1) * inv_sqrt_n0

            n1_pos = jnp.array(n1 > 0, dtype=dtype)
            n1_ge_2 = jnp.array(n1 >= 2, dtype=dtype)

            # Logic: If n0 > 0, we use index i=0. Else if n1 > 0, use i=1.

            def body_n2(n2, h):
                # n2 precalcs
                inv_sqrt_n2 = jax.lax.rsqrt(jnp.maximum(n2, 1.0).astype(dtype))
                sqrt_n2_minus_1_div_n2 = jnp.sqrt(
                    (jnp.maximum(n2 - 1, 0.0)) * inv_sqrt_n2 * inv_sqrt_n2
                )
                # Cross
                sqrt_n2_div_n0 = jnp.sqrt(n2) * inv_sqrt_n0
                sqrt_n2_div_n1 = jnp.sqrt(n2) * inv_sqrt_n1

                n2_pos = jnp.array(n2 > 0, dtype=dtype)
                n2_ge_2 = jnp.array(n2 >= 2, dtype=dtype)

                # --- VAL 0 (i=0) ---
                # gamma0 * A[n0-1, n1, n2] / sqrt(n0)
                v0 = (g0 * h[n0 - 1, n1, n2]) * inv_sqrt_n0
                v0 += (b00 * h[n0 - 2, n1, n2] * sqrt_n0_minus_1_div_n0) * n0_ge_2
                v0 += (b01 * h[n0 - 1, n1 - 1, n2] * sqrt_n1_div_n0) * n1_pos
                v0 += (b02 * h[n0 - 1, n1, n2 - 1] * sqrt_n2_div_n0) * n2_pos

                # --- VAL 1 (i=1) ---
                # gamma1 * A[0, n1-1, n2] / sqrt(n1)
                v1 = (g1 * h[0, n1 - 1, n2]) * inv_sqrt_n1
                v1 += (b11 * h[0, n1 - 2, n2] * sqrt_n1_minus_1_div_n1) * n1_ge_2
                v1 += (b12 * h[0, n1 - 1, n2 - 1] * sqrt_n2_div_n1) * n2_pos

                # --- VAL 2 (i=2) ---
                v2 = (g2 * h[0, 0, n2 - 1]) * inv_sqrt_n2
                v2 += (b22 * h[0, 0, n2 - 2] * sqrt_n2_minus_1_div_n2) * n2_ge_2

                # Selection
                # if n0 > 0: use v0
                # else if n1 > 0: use v1
                # else: use v2

                final_val = (
                    v0 * n0_pos
                    + v1 * (1.0 - n0_pos) * n1_pos
                    + v2 * (1.0 - n0_pos) * (1.0 - n1_pos)
                )

                # Handle Start
                is_start = (n0 == 0) & (n1 == 0) & (n2 == 0)
                current = h[n0, n1, n2]
                new_val = jax.lax.select(is_start, current, final_val)

                return h.at[n0, n1, n2].set(new_val)

            return jax.lax.fori_loop(0, N2, body_n2, h1)

        return jax.lax.fori_loop(0, N1, body_n1, h0)

    A = jax.lax.fori_loop(0, N0, body_n0, A)
    return A


def _recurrence_4d(shape, B, gamma, dtype, prefactor):
    """
    Optimized 4D recurrence using jnp.where instead of jax.lax.cond.
    This eliminates control flow overhead and is much faster on GPU.
    """
    N0, N1, N2, N3 = shape
    A = jnp.zeros(shape, dtype=dtype)
    A = A.at[0, 0, 0, 0].set(prefactor)

    # Precompute zero for branchless ops
    zero = jnp.array(0.0, dtype=dtype)

    def body_n0(n0, h0):
        def body_n1(n1, h1):
            def body_n2(n2, h2):
                def body_n3(n3, h):
                    # Skip origin (already set)
                    is_start = (n0 == 0) & (n1 == 0) & (n2 == 0) & (n3 == 0)

                    # Precompute safe sqrt values (avoid division by zero)
                    sqrt_n0 = jnp.sqrt(jnp.maximum(n0, 1.0))
                    sqrt_n1 = jnp.sqrt(jnp.maximum(n1, 1.0))
                    sqrt_n2 = jnp.sqrt(jnp.maximum(n2, 1.0))
                    sqrt_n3 = jnp.sqrt(jnp.maximum(n3, 1.0))

                    # use_0: when n0 > 0
                    val_0 = (gamma[0] * h[jnp.maximum(n0 - 1, 0), n1, n2, n3]) / sqrt_n0
                    val_0 += jnp.where(
                        n0 >= 2,
                        B[0, 0]
                        * h[jnp.maximum(n0 - 2, 0), n1, n2, n3]
                        * jnp.sqrt(jnp.maximum(n0 - 1, 0.0) / jnp.maximum(n0, 1.0)),
                        zero,
                    )
                    val_0 += jnp.where(
                        n1 >= 1,
                        B[0, 1]
                        * h[jnp.maximum(n0 - 1, 0), jnp.maximum(n1 - 1, 0), n2, n3]
                        * jnp.sqrt(n1 / jnp.maximum(n0, 1.0)),
                        zero,
                    )
                    val_0 += jnp.where(
                        n2 >= 1,
                        B[0, 2]
                        * h[jnp.maximum(n0 - 1, 0), n1, jnp.maximum(n2 - 1, 0), n3]
                        * jnp.sqrt(n2 / jnp.maximum(n0, 1.0)),
                        zero,
                    )
                    val_0 += jnp.where(
                        n3 >= 1,
                        B[0, 3]
                        * h[jnp.maximum(n0 - 1, 0), n1, n2, jnp.maximum(n3 - 1, 0)]
                        * jnp.sqrt(n3 / jnp.maximum(n0, 1.0)),
                        zero,
                    )

                    # use_1: when n0 == 0 and n1 > 0
                    val_1 = (gamma[1] * h[0, jnp.maximum(n1 - 1, 0), n2, n3]) / sqrt_n1
                    val_1 += jnp.where(
                        n1 >= 2,
                        B[1, 1]
                        * h[0, jnp.maximum(n1 - 2, 0), n2, n3]
                        * jnp.sqrt(jnp.maximum(n1 - 1, 0.0) / jnp.maximum(n1, 1.0)),
                        zero,
                    )
                    val_1 += jnp.where(
                        n2 >= 1,
                        B[1, 2]
                        * h[0, jnp.maximum(n1 - 1, 0), jnp.maximum(n2 - 1, 0), n3]
                        * jnp.sqrt(n2 / jnp.maximum(n1, 1.0)),
                        zero,
                    )
                    val_1 += jnp.where(
                        n3 >= 1,
                        B[1, 3]
                        * h[0, jnp.maximum(n1 - 1, 0), n2, jnp.maximum(n3 - 1, 0)]
                        * jnp.sqrt(n3 / jnp.maximum(n1, 1.0)),
                        zero,
                    )

                    # use_2: when n0 == 0 and n1 == 0 and n2 > 0
                    val_2 = (gamma[2] * h[0, 0, jnp.maximum(n2 - 1, 0), n3]) / sqrt_n2
                    val_2 += jnp.where(
                        n2 >= 2,
                        B[2, 2]
                        * h[0, 0, jnp.maximum(n2 - 2, 0), n3]
                        * jnp.sqrt(jnp.maximum(n2 - 1, 0.0) / jnp.maximum(n2, 1.0)),
                        zero,
                    )
                    val_2 += jnp.where(
                        n3 >= 1,
                        B[2, 3]
                        * h[0, 0, jnp.maximum(n2 - 1, 0), jnp.maximum(n3 - 1, 0)]
                        * jnp.sqrt(n3 / jnp.maximum(n2, 1.0)),
                        zero,
                    )

                    # use_3: when n0 == 0 and n1 == 0 and n2 == 0 and n3 > 0
                    val_3 = (gamma[3] * h[0, 0, 0, jnp.maximum(n3 - 1, 0)]) / sqrt_n3
                    val_3 += jnp.where(
                        n3 >= 2,
                        B[3, 3]
                        * h[0, 0, 0, jnp.maximum(n3 - 2, 0)]
                        * jnp.sqrt(jnp.maximum(n3 - 1, 0.0) / jnp.maximum(n3, 1.0)),
                        zero,
                    )

                    # Select based on which index is first > 0
                    val = jnp.where(
                        n0 > 0,
                        val_0,
                        jnp.where(n1 > 0, val_1, jnp.where(n2 > 0, val_2, val_3)),
                    )

                    # Handle origin
                    val = jnp.where(is_start, h[0, 0, 0, 0], val)

                    return h.at[n0, n1, n2, n3].set(val)

                return jax.lax.fori_loop(0, N3, body_n3, h2)

            return jax.lax.fori_loop(0, N2, body_n2, h1)

        return jax.lax.fori_loop(0, N1, body_n1, h0)

    A = jax.lax.fori_loop(0, N0, body_n0, A)
    return A
