"""
ops.py

Common quantum operations and helpers for Hanamura circuits.
Includes:
- Constants (HBAR)
- Quadrature functions
- Single-mode operators (annihilation, creation)
- Beamsplitter builders (Fock basis and 2x2 matrix)
- State vector helpers
"""

import numpy as np
import math
from typing import Optional, Tuple
from scipy.special import eval_hermite, gammaln
from scipy.linalg import expm
from src.utils.accel import njit_wrapper as njit
from src.utils.cache_manager import CacheManager
from threading import Lock

# -----------------------------------------------------------
# Global numeric conventions
# -----------------------------------------------------------
HBAR = 2.0  # standardize to hbar=2 everywhere (Walrus-friendly choice)


# -----------------------------------------------------------
# Quadrature helpers (consistent with general hbar)
# -----------------------------------------------------------
def quadrature_wavefunction(n: int, x: float, hbar: float = HBAR) -> float:
    """
    Harmonic oscillator quadrature wavefunction psi_n(x) for general hbar.
    Formula:
        psi_n(x) = (1 / (pi*hbar)^(1/4)) * (1/sqrt(2^n n!)) * H_n(x/sqrt(hbar)) * exp(-x^2/(2 hbar))
    Returns a Python float.
    """
    arg = x / math.sqrt(hbar)
    pref = (math.pi * hbar) ** (-0.25) / math.sqrt((2.0**n) * math.exp(gammaln(n + 1)))
    Hn = eval_hermite(n, arg)
    return float(pref * Hn * math.exp(-0.5 * arg * arg))


def quadrature_vector(cutoff: int, x: float, hbar: float = HBAR) -> np.ndarray:
    """
    Build the quadrature vector phi_n(x) (length=cutoff) for given x and hbar.
    DEPRECATED: Use get_phi_matrix_cached for performance.
    """
    return np.array(
        [quadrature_wavefunction(n, x, hbar=hbar) for n in range(cutoff)], dtype=float
    )


# -----------------------------------------------------------
# Vectorized & Cached Quadrature (Stage 3 Optimization)
# -----------------------------------------------------------
_QUAD_CACHE = {}
_QUAD_CACHE_LOCK = Lock()


def quadrature_prefactors(cutoff: int, hbar: float = HBAR) -> np.ndarray:
    """
    Precompute normalization prefactors for Hermite functions.
    pref_n = (pi*hbar)^(-1/4) / sqrt(2^n n!)
    """
    # compute in log domain for stability
    log_pref0 = -0.25 * math.log(math.pi * hbar)
    n = np.arange(cutoff)
    log_denom = 0.5 * (n * math.log(2.0) + gammaln(n + 1.0))
    log_pref = log_pref0 - log_denom
    pref = np.exp(log_pref)  # shape (cutoff,)
    return pref


@njit
def _hermite_phi_matrix(
    cutoff: int, xs: np.ndarray, prefactors: np.ndarray, hbar: float
) -> np.ndarray:
    """
    Compute quadrature matrix Phi[n, i] = phi_n(xs[i]) using stable recurrence.
    JIT-compiled for speed.
    """
    N = xs.shape[0]
    phi = np.zeros((cutoff, N), dtype=np.float64)
    # transform argument
    arg = np.empty(N, dtype=np.float64)
    sqrt_h = math.sqrt(hbar)
    for i in range(N):
        arg[i] = xs[i] / sqrt_h

    # compute H_0 and H_1 arrays
    if cutoff > 0:
        # H0(x) = 1
        for i in range(N):
            phi[0, i] = prefactors[0] * 1.0 * math.exp(-0.5 * arg[i] * arg[i])
    if cutoff > 1:
        for i in range(N):
            H1 = 2.0 * arg[i]
            phi[1, i] = prefactors[1] * H1 * math.exp(-0.5 * arg[i] * arg[i])
    # recurrence for n >= 2
    if cutoff > 2:
        # maintain arrays for H_{n-2} and H_{n-1} at each x
        H_nm2 = np.empty(N, dtype=np.float64)
        H_nm1 = np.empty(N, dtype=np.float64)
        # initialize
        for i in range(N):
            H_nm2[i] = 1.0
            H_nm1[i] = 2.0 * arg[i]
        for n in range(2, cutoff):
            for i in range(N):
                Hn = 2.0 * arg[i] * H_nm1[i] - 2.0 * (n - 1) * H_nm2[i]
                phi[n, i] = prefactors[n] * Hn * math.exp(-0.5 * arg[i] * arg[i])
                # shift
                H_nm2[i] = H_nm1[i]
                H_nm1[i] = Hn
    return phi


def get_phi_matrix_cached(
    cutoff: int, xs: np.ndarray, hbar: float = HBAR
) -> np.ndarray:
    """
    Get the quadrature matrix Phi[n, i] from cache or compute it.
    Thread-safe.
    """
    # Build a stable cache key: quantize xs to 1e-10 or use bytes hash
    # For window quadrature we often reuse the same xs exactly, so using bytes is fine.
    # Ensure xs is float64 and contiguous for consistent bytes
    # Round to 8 decimals to ensure cache hits for slightly varying inputs
    xs_rounded = np.round(xs, decimals=8)
    xs_f64 = np.ascontiguousarray(xs_rounded, dtype=np.float64)
    key = (int(cutoff), float(hbar), xs_f64.tobytes())

    with _QUAD_CACHE_LOCK:
        res = _QUAD_CACHE.get(key)

    if res is not None:
        return res

    pref = quadrature_prefactors(cutoff, hbar)
    Phi = _hermite_phi_matrix(cutoff, xs_f64, pref, hbar)

    with _QUAD_CACHE_LOCK:
        _QUAD_CACHE[key] = Phi

    return Phi


# -----------------------------------------------------------
# Single-mode ladder operators (truncated)
# -----------------------------------------------------------
def annihilation_operator(cutoff: int) -> np.ndarray:
    "Return the single-mode annihilation operator (cutoff x cutoff)."
    a = np.zeros((cutoff, cutoff), dtype=complex)
    for n in range(1, cutoff):
        a[n - 1, n] = math.sqrt(n)
    return a


def creation_operator(cutoff: int) -> np.ndarray:
    "Return single-mode creation operator (cutoff x cutoff)."
    return annihilation_operator(cutoff).conj().T


# -----------------------------------------------------------
# Beamsplitter builder (expm generator) with caching
# -----------------------------------------------------------
def _bs_cache_key(
    cutoff: int, theta: float, phi: float
) -> Tuple[str, int, float, float]:
    return ("U_bs", int(cutoff), float(theta), float(phi))


def build_beamsplitter_unitary(
    cutoff: int, theta: float, phi: float = 0.0, cache: Optional[CacheManager] = None
) -> np.ndarray:
    r"""
    Build the full two-mode beamsplitter unitary U_bs of shape (c^2, c^2) in the truncated Fock basis,
    using the second-quantized generator and matrix exponential:

        K = theta * ( e^{i phi} a1^\dag a2 - e^{-i phi} a1 a2^\dag )
        U_bs = expm(K)

    This construction is numerically stable (matrix exponential of a sparse operator).
    - cutoff: single-mode cutoff dimension
    - theta: mixing angle (so that for phi=0 the transformation of annihilation ops is the usual rotation)
    - phi: beamsplitter internal phase (real)
    The result is cached (if cache is provided) using CacheManager.
    """
    key = _bs_cache_key(cutoff, theta, phi)

    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    # build single-mode operators
    a = annihilation_operator(cutoff)
    identity = np.eye(cutoff, dtype=complex)

    # two-mode operators in full Hilbert space
    a1 = np.kron(a, identity)
    a2 = np.kron(identity, a)
    adag1 = a1.conj().T
    adag2 = a2.conj().T

    # generator K
    K = np.exp(1j * phi) * np.dot(adag1, a2) - np.exp(-1j * phi) * np.dot(a1, adag2)
    K = float(theta) * K  # scale

    # exponentiate
    U = expm(K)

    # cache and return
    if cache is not None:
        cache.set(key, U)
    return U


@njit
def beamsplitter_2x2(theta: float, phi: float) -> np.ndarray:
    """
    2x2 complex beam-splitter matrix B(theta, phi) acting on [a_i, a_j]^T such that
    B = [[cosθ, -e^{-iφ} sinθ],
         [e^{iφ} sinθ, cosθ]].

    This choice matches a common parameterization used in interferometer templates.
    """
    t = np.cos(theta)
    r = np.sin(theta)
    return np.array(
        [[t, -np.exp(-1j * phi) * r], [np.exp(1j * phi) * r, t]], dtype=np.complex128
    )


# -----------------------------------------------------------
# Utility: kron of vector -> full two-mode state vector
# -----------------------------------------------------------
def kron_state_vector(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Return the Kronecker (tensor) product vector for two single-mode states.
    Both f1 and f2 may be 1D vectors (length = cutoff).
    """
    return np.kron(f1, f2)


# -----------------------------------------------------------
# Homodyne helpers
# -----------------------------------------------------------
def contract_rho_with_phi(rho_full: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Given rho_full shaped (c^2, c^2) and phi (length c), compute unnormalized reduced
    rho for mode1 after projecting mode2 onto |x> via quadrature vector phi.
    Implementation uses tensordot contraction:
       rho_t[n0,n1,m0,m1] -> contract n1 and m1 with phi -> new_rho[n0,m0]
    """
    c = int(round(np.sqrt(rho_full.shape[0])))
    rho_t = rho_full.reshape((c, c, c, c))
    tmp = np.tensordot(rho_t, phi, axes=([1], [0]))  # shape (n0,m0,m1)
    new_rho = np.tensordot(tmp, phi, axes=([2], [0]))  # shape (n0,m0)
    return new_rho
