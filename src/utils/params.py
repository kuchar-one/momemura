import numpy as np
from typing import Sequence, Tuple
from src.utils.accel import njit_wrapper as njit


# -----------------------------
# Low-level helpers
# -----------------------------
@njit
def omega_matrix(num_modes: int) -> np.ndarray:
    """
    Return the symplectic form Omega (2N x 2N) in q,p ordering:
    Omega = block_diag([ [0,1], [-1,0] , ... ]).
    """
    N = int(num_modes)
    Omega = np.zeros((2 * N, 2 * N), dtype=np.float64)
    for m in range(N):
        i = 2 * m
        j = i + 1
        Omega[i, j] = 1.0
        Omega[j, i] = -1.0
    return Omega


def vacuum_covariance(num_modes: int, hbar: float = 2.0) -> np.ndarray:
    """
    Return vacuum covariance in q,p ordering. For the chosen convention (hbar=2)
    this equals the identity(2N).
    """
    return np.eye(2 * int(num_modes), dtype=float)


def two_mode_squeezer_symplectic(r: float) -> np.ndarray:
    """
    Return the 4x4 real symplectic matrix for a two-mode squeezer in q,p ordering
    acting on modes (1,2) arranged as (q1,p1,q2,p2).
    """
    cr = np.cosh(r)
    sr = np.sinh(r)
    S = np.array(
        [
            [cr, 0.0, sr, 0.0],
            [0.0, cr, 0.0, -sr],
            [sr, 0.0, cr, 0.0],
            [0.0, -sr, 0.0, cr],
        ],
        dtype=float,
    )
    return S


@njit
def expand_mode_symplectic(
    S_small: np.ndarray, mode_indices: np.ndarray, total_modes: int
) -> np.ndarray:
    """
    Embed a small (2m x 2m) symplectic matrix S_small, acting on modes listed in
    mode_indices (length m), into a 2N x 2N identity-like matrix with the small
    block placed in the corresponding q,p subindices.

    mode_indices are integers in 0..N-1; total_modes = N.
    """
    N = int(total_modes)
    big = np.eye(2 * N, dtype=np.float64)
    # small_m = len(mode_indices)
    for a_i, ma in enumerate(mode_indices):
        for a_j, mb in enumerate(mode_indices):
            for qp_i in (0, 1):
                for qp_j in (0, 1):
                    big_idx_i = 2 * ma + qp_i
                    big_idx_j = 2 * mb + qp_j
                    small_idx_i = 2 * a_i + qp_i
                    small_idx_j = 2 * a_j + qp_j
                    big[big_idx_i, big_idx_j] = S_small[small_idx_i, small_idx_j]
    return big


@njit
def passive_unitary_to_symplectic(U: np.ndarray) -> np.ndarray:
    """
    Convert an n x n complex passive unitary U (acting on annihilation operators)
    to its 2n x 2n real symplectic representation in (q,p) ordering:

        S = [[Re(U), -Im(U)],
             [Im(U),  Re(U)]]

    This implements the passive Gaussian action (no squeezing).
    """
    U = np.asarray(U, dtype=np.complex128)
    Re = np.real(U)
    Im = np.imag(U)
    # Numba supports concatenate
    top = np.concatenate((Re, -Im), axis=1)
    bot = np.concatenate((Im, Re), axis=1)
    return np.concatenate((top, bot), axis=0)


@njit
def complex_alpha_to_qp(alpha: Sequence[complex]) -> np.ndarray:
    """
    Convert complex displacement (annihilation basis) alpha -> real qp vector.
    Convention: a = (q + i p)/2, hbar = 2, so
        q = 2 Re(alpha), p = 2 Im(alpha).
    Returns array shaped (2*N,), ordering [q0,p0,q1,p1,...].
    """
    alpha = np.asarray(alpha, dtype=np.complex128)
    qp = np.empty(2 * alpha.size, dtype=np.float64)
    for i, a in enumerate(alpha):
        qp[2 * i] = 2.0 * np.real(a)
        qp[2 * i + 1] = 2.0 * np.imag(a)
    return qp


@njit
def interleaved_to_xp(
    mu_ip: np.ndarray, cov_ip: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert mu and cov from interleaved ordering [q1,p1,q2,p2,...]
    to xp ordering [q1,q2,...,p1,p2,...].

    Returns (mu_xp, cov_xp).
    """
    mu_ip = np.asarray(mu_ip)
    cov_ip = np.asarray(cov_ip)
    if mu_ip.ndim != 1:
        raise ValueError("mu must be a 1D array")
    N2 = mu_ip.size
    if N2 % 2 != 0:
        raise ValueError("mu length must be even")
    N = N2 // 2

    # Direct permutation without intermediate list
    mu_xp = np.empty_like(mu_ip)
    # q's: 0, 2, 4... -> 0, 1, 2...
    for m in range(N):
        mu_xp[m] = mu_ip[2 * m]
    # p's: 1, 3, 5... -> N, N+1, N+2...
    for m in range(N):
        mu_xp[N + m] = mu_ip[2 * m + 1]

    # For covariance, we need permutation matrix P such that cov_xp = P cov_ip P^T
    P = np.zeros((2 * N, 2 * N), dtype=np.float64)
    # q's
    for m in range(N):
        # source index 2*m maps to target index m
        P[m, 2 * m] = 1.0
    # p's
    for m in range(N):
        # source index 2*m+1 maps to target index N+m
        P[N + m, 2 * m + 1] = 1.0

    cov_xp = P @ cov_ip @ P.T
    return mu_xp, cov_xp


@njit
def xp_to_interleaved(S_xp: np.ndarray) -> np.ndarray:
    """
    Convert a 2N x 2N symplectic / real matrix from xp ordering
    [q0,q1,...,p0,p1,...] to interleaved ordering [q0,p0,q1,p1,...].

    Returns the converted matrix S_ip.
    """
    S_xp = np.asarray(S_xp)
    if S_xp.ndim != 2 or S_xp.shape[0] != S_xp.shape[1]:
        raise ValueError("S_xp must be a square 2N x 2N matrix.")
    N2 = S_xp.shape[0]
    if N2 % 2 != 0:
        raise ValueError("S_xp size must be even.")
    N = N2 // 2

    # Build permutation matrix P with P[i, perm[i]] = 1
    # perm maps target index i (in xp) to source index perm[i] (in interleaved)
    # Wait, previous logic was: perm[i] is source index for target i.
    # mu_xp[i] = mu_ip[perm[i]]
    # So P[i, perm[i]] = 1 is correct for mu_xp = P @ mu_ip.

    P = np.zeros((2 * N, 2 * N), dtype=np.float64)
    # q's: target m (0..N-1) comes from source 2*m
    for m in range(N):
        P[m, 2 * m] = 1.0
    # p's: target N+m (N..2N-1) comes from source 2*m+1
    for m in range(N):
        P[N + m, 2 * m + 1] = 1.0

    # We derived: S_ip = P.T @ S_xp @ P
    S_ip = P.T @ S_xp @ P
    return S_ip
